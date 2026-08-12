"""Train hierarchical YOLOv11 for IHC tumor gating and intensity grading.

Example:
    /home/user/anaconda3/envs/urban/bin/python train_hierarchical.py \
        --stain her2 --model-size m --epochs 300
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets import nn
from utils import util
from utils.hierarchical import HierarchicalComputeLoss, load_flat_checkpoint_into_hierarchical
from utils.ihc_dataset import (
    IHCHierarchicalDataset,
    discover_records,
    split_records_by_slide,
)
from utils.valid import compute_point_label_metrics_single


CLASS_NAMES = {0: 'class0', 1: 'class1', 2: 'class2', 3: 'class3', 4: 'other'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stain', choices=('her2', 'er_pr'), default='her2')
    parser.add_argument('--data-root', type=Path, default=Path('../../data/precise_BC_cell_scoring'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--pretrained-flat', type=Path, default=None,
                        help='Existing five-class best_model.pt used to warm-start the model')
    parser.add_argument('--resume', type=Path, default=None,
                        help='Hierarchical last_model.pt to resume')
    parser.add_argument('--model-size', choices=('n', 't', 's', 'm', 'l', 'x'), default='m')
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--min-learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--false-tumor-weight', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=242)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_model(size):
    constructor = getattr(nn, f'yolo_v11_{size}_hierarchical')
    return constructor(num_grades=4)


def make_checkpoint(epoch, model, optimizer, scheduler, scaler, losses, metrics,
                    best_score, args):
    return {
        'architecture': 'hierarchical_cell_tumor_grade',
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'amp_scale_state_dict': scaler.state_dict(),
        'losses': losses,
        'metrics': metrics,
        'best_score': best_score,
        'class_names': CLASS_NAMES,
        'args': vars(args),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    stain_root = args.data_root / args.stain
    records = discover_records(stain_root / 'patch_images', stain_root / 'labels')
    train_records, val_records = split_records_by_slide(
        records, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f'Data: {len(train_records)} train / {len(val_records)} val patches (slide-level split)')

    train_dataset = IHCHierarchicalDataset(
        train_records, input_size=args.input_size, augment=True
    )
    val_dataset = IHCHierarchicalDataset(
        val_records, input_size=args.input_size, augment=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        drop_last=True,
        collate_fn=IHCHierarchicalDataset.collate_fn,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
        collate_fn=IHCHierarchicalDataset.collate_fn,
        persistent_workers=args.workers > 0,
    )

    model = build_model(args.model_size).to(device)
    output_dir = args.output_dir or Path(
        f'../../model/precise_BC_cell_scoring/{args.stain}_hierarchical_yolov11'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        util.set_params(model, args.weight_decay),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')
    start_epoch = 0
    best_score = float('-inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['amp_scale_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_score = float(checkpoint.get('best_score', best_score))
        print(f'Resumed hierarchical checkpoint at epoch {start_epoch}')
    else:
        pretrained = args.pretrained_flat
        if pretrained is None:
            candidate = Path(
                f'../../model/precise_BC_cell_scoring/'
                f'{"her2_yolov11" if args.stain == "her2" else "ER_PR_yolov11"}/best_model.pt'
            )
            pretrained = candidate if candidate.exists() else None
        if pretrained:
            report = load_flat_checkpoint_into_hierarchical(model, pretrained, map_location=device)
            print(f'Warm-started from {pretrained}: {report}')

    loss_params = {
        'box': 7.5,
        'dfl': 0.5,
        'objectness': 1.0,
        'tumor': 1.0,
        'grade': 1.0,
        'false_tumor_weight': args.false_tumor_weight,
        'top_k': 10,
        'assigner_alpha': 0.5,
        'assigner_beta': 6.0,
    }
    criterion = HierarchicalComputeLoss(model, loss_params)
    metric_params = {'names': CLASS_NAMES}
    accumulation_steps = max(round(64 / args.batch_size), 1)

    with open(output_dir / 'split_summary.json', 'w', encoding='utf-8') as file:
        json.dump({
            'train_patches': len(train_records),
            'val_patches': len(val_records),
            'train_slides': sorted({record[0].stem.rsplit('_', 2)[0] for record in train_records}),
            'val_slides': sorted({record[0].stem.rsplit('_', 2)[0] for record in val_records}),
        }, file, indent=2)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        meters = {name: util.AverageMeter() for name in (
            'box', 'objectness', 'tumor', 'grade', 'dfl', 'total'
        )}
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')

        for step, (images, targets) in enumerate(progress):
            images = images.to(device, non_blocking=True).float() / 255.0
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(images)
                losses = criterion(outputs, targets)
                scaled_loss = losses['total'] / accumulation_steps

            scaler.scale(scaled_loss).backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for name, value in losses.items():
                meters[name].update(value.detach().item(), images.shape[0])
            progress.set_postfix({
                'obj': f'{meters["objectness"].avg:.3f}',
                'tumor': f'{meters["tumor"].avg:.3f}',
                'grade': f'{meters["grade"].avg:.3f}',
            })

        scheduler.step()
        average_losses = {name: meter.avg for name, meter in meters.items()}
        metrics = {}

        should_validate = (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs
        if should_validate:
            metrics = compute_point_label_metrics_single(
                model, val_loader, device, metric_params, distance_threshold=16,
                class_agnostic_nms=True,
            )
            other_recall = metrics.get('class_stats', {}).get('other', {}).get('recall', 0.0)
            macro_f1 = metrics.get('macro_f1', 0.0)
            # Prioritize correct rejection of non-tumor cells while retaining
            # overall grade performance as a secondary objective.
            selection_score = 0.7 * other_recall + 0.3 * macro_f1
            print(
                f'Validation: other recall={other_recall:.4f}, '
                f'macro F1={macro_f1:.4f}, selection={selection_score:.4f}'
            )
        else:
            selection_score = float('-inf')

        is_best = selection_score > best_score
        if is_best:
            best_score = selection_score
        checkpoint = make_checkpoint(
            epoch, model, optimizer, scheduler, scaler, average_losses, metrics,
            best_score, args
        )
        torch.save(checkpoint, output_dir / 'last_model.pt')
        if is_best:
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f'New best checkpoint: {best_score:.4f}')

    print(f'Training complete. Checkpoints: {output_dir}')


if __name__ == '__main__':
    main()
