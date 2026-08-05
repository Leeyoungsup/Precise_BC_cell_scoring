# Model evaluation results

Four YOLOv11 checkpoints were evaluated on the same validation split used by
their corresponding training notebook (`test_size=0.1`, `random_state=242`).

- Confidence threshold: 0.1
- Class-agnostic NMS IoU threshold: 0.3
- Hungarian match IoU threshold: 0.3
- Checkpoint: `best_model.pt`

The consolidated comparison is in `model_evaluation_summary.csv`. Each model
also has a per-class table (`*_table1_per_class_metrics.csv`) and an overall
summary (`*_table2_point_detection_summary.csv`).

The Notion-ready ER/PR report is in `er_pr_notion.md`, with its eight image
assets under `er_pr_figures/`.

These numbers are validation results because this split was used during model
selection. A separate patient/slide-level holdout set is required for an
unbiased final test result.
