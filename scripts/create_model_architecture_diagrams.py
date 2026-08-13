"""Create publication-ready architecture diagrams from the implemented models.

The diagram embeds an unmodified 512x512 ER/PR IHC patch from the project
dataset.  Network blocks and probability equations mirror ``nets/nn.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_IMAGE = (
    ROOT
    / "../../data/precise_BC_cell_scoring/er_pr/patch_images"
    / "TS25-003604_001_001_ER_111619.svs_29220_9740.png"
).resolve()
OUTPUT_DIR = ROOT / "docs/results/model_architecture"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

COLORS = {
    "navy": "#173B57",
    "blue": "#DCEEFF",
    "blue_edge": "#3A78A1",
    "teal": "#DDF4F0",
    "teal_edge": "#25877A",
    "orange": "#FFF0D8",
    "orange_edge": "#D47A13",
    "purple": "#EEE5FF",
    "purple_edge": "#7957B8",
    "red": "#FFE1E5",
    "red_edge": "#C6475A",
    "yellow": "#FFF5C8",
    "yellow_edge": "#B68A00",
    "green": "#E2F4E8",
    "green_edge": "#358454",
    "gray": "#F2F4F6",
    "gray_edge": "#7B8792",
    "muted": "#52616B",
}


def setup_font() -> None:
    if Path(FONT_REGULAR).exists():
        font_manager.fontManager.addfont(FONT_REGULAR)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=FONT_REGULAR
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    *,
    face: str = "gray",
    edge: str = "gray_edge",
    title_size: float = 12,
    body_size: float = 9.2,
    linewidth: float = 1.7,
    radius: float = 0.12,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.06,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=COLORS.get(edge, edge),
        facecolor=COLORS.get(face, face),
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.66 if body else 0.5),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["navy"],
        zorder=zorder + 1,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.31,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            linespacing=1.35,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#48616F",
    style: str = "-|>",
    linewidth: float = 1.8,
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=linewidth,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def input_patch(ax, image, x: float, y: float, size: float) -> None:
    ax.imshow(image, extent=(x, x + size, y, y + size), zorder=3)
    ax.add_patch(
        Rectangle(
            (x, y), size, size, fill=False, linewidth=1.8,
            edgecolor=COLORS["navy"], zorder=4
        )
    )
    ax.text(
        x + size / 2,
        y - 0.16,
        "실제 ER/PR IHC patch · 512×512",
        ha="center",
        va="top",
        fontsize=8.3,
        color=COLORS["muted"],
    )


def scales_box(ax, x: float, y: float, w: float, h: float) -> None:
    box(
        ax, x, y, w, h, "Multi-scale features", "P3 / 8  ·  64×64×256\nP4 / 16 · 32×32×512\nP5 / 32 · 16×16×512",
        face="gray", edge="gray_edge", title_size=10.5, body_size=8.1
    )


def draw_common(ax, image, y: float) -> dict[str, tuple[float, float, float, float]]:
    coords = {
        "image": (0.55, y + 0.30, 2.05, 2.05),
        "backbone": (3.05, y + 0.55, 2.25, 1.55),
        "fpn": (5.75, y + 0.55, 2.10, 1.55),
        "scale": (8.30, y + 0.40, 1.65, 1.85),
    }
    input_patch(ax, image, *coords["image"][:2], coords["image"][2])
    box(
        ax, *coords["backbone"], "YOLOv11-m Backbone",
        "DarkNet\nConv + CSP + SPP + PSA",
        face="blue", edge="blue_edge", title_size=11.3, body_size=9.1
    )
    box(
        ax, *coords["fpn"], "DarkFPN",
        "Top-down + Bottom-up\nfeature aggregation",
        face="teal", edge="teal_edge", title_size=11.3, body_size=9.1
    )
    scales_box(ax, *coords["scale"])
    arrow(ax, (2.62, y + 1.32), (3.00, y + 1.32))
    arrow(ax, (5.32, y + 1.32), (5.70, y + 1.32))
    arrow(ax, (7.87, y + 1.32), (8.25, y + 1.32))
    return coords


def draw_flat(ax, image, y: float, *, row_label: bool = True) -> None:
    if row_label:
        ax.text(
            0.35, y + 2.82, "A. 기존 모델 — Flat 5-class YOLOv11-m",
            fontsize=14.2, fontweight="bold", color=COLORS["orange_edge"], va="center"
        )
    draw_common(ax, image, y)
    box(
        ax, 10.42, y + 0.28, 2.55, 2.10, "Flat Detection Head ×3",
        "Box tower\n4 × 16 DFL logits\n\nClass tower\n5 independent sigmoid logits",
        face="orange", edge="orange_edge", title_size=11.1, body_size=8.5
    )
    box(
        ax, 13.46, y + 0.48, 2.18, 1.70, "Flat class scores",
        "P(0+), P(1+), P(2+),\nP(3+), P(other)\n\nCell 여부와 class가 결합",
        face="orange", edge="orange_edge", title_size=10.7, body_size=8.4
    )
    box(
        ax, 16.12, y + 0.48, 1.52, 1.70, "Output",
        "[box + 5 scores]\n\nFlat-compatible tensor",
        face="green", edge="green_edge", title_size=11, body_size=8.5
    )
    box(
        ax, 18.08, y + 0.48, 1.45, 1.70, "Detections",
        "Class-agnostic NMS\n\n0+ / 1+ / 2+ / 3+ / other",
        face="green", edge="green_edge", title_size=10.6, body_size=7.7
    )
    arrow(ax, (9.98, y + 1.32), (10.37, y + 1.32))
    arrow(ax, (13.02, y + 1.32), (13.41, y + 1.32))
    arrow(ax, (15.68, y + 1.32), (16.07, y + 1.32))
    arrow(ax, (17.68, y + 1.32), (18.03, y + 1.32))
    ax.text(
        11.70, y + 0.03,
        "Training loss: Box + Classification + DFL",
        ha="center", va="top", fontsize=8.7, color=COLORS["muted"]
    )


def draw_hierarchical(ax, image, y: float, *, row_label: bool = True) -> None:
    if row_label:
        ax.text(
            0.35, y + 2.82, "B. Hierarchical 모델 — Cell → Tumor gate → Grade",
            fontsize=14.2, fontweight="bold", color=COLORS["purple_edge"], va="center"
        )
    draw_common(ax, image, y)
    # Box and semantic towers are parallel branches from every P3/P4/P5
    # feature map.  The layout deliberately keeps them vertically separated
    # so the diagram does not imply a sequential connection.
    box(
        ax, 10.34, y + 1.48, 1.50, 0.86, "Box tower",
        "4 × 16 DFL logits",
        face="blue", edge="blue_edge", title_size=9.8, body_size=7.9
    )
    box(
        ax, 10.34, y + 0.16, 1.50, 1.00, "Semantic tower",
        "shared feature\nDWConv + Conv",
        face="purple", edge="purple_edge", title_size=9.5, body_size=7.7
    )
    box(
        ax, 12.18, y + 1.72, 1.72, 0.64, "Objectness · sigmoid",
        "P(cell)", face="green", edge="green_edge", title_size=8.5, body_size=7.3
    )
    box(
        ax, 12.18, y + 0.96, 1.72, 0.64, "Tumor gate · sigmoid",
        "P(tumor | cell)", face="red", edge="red_edge", title_size=8.5, body_size=7.3
    )
    box(
        ax, 12.18, y + 0.20, 1.72, 0.64, "Grade · softmax",
        "P(0+/1+/2+/3+ | tumor)", face="yellow", edge="yellow_edge",
        title_size=8.5, body_size=6.8
    )
    box(
        ax, 14.38, y + 0.20, 2.36, 2.16, "Explicit fusion",
        "Box + semantic scores\n\nTumor grade score\n= cell × tumor × grade\n\nOther score\n= cell × (1 − tumor)\n\nHard gate: tumor ≥ 0.50",
        face="gray", edge="gray_edge", title_size=10.0, body_size=7.3
    )
    box(
        ax, 17.24, y + 0.43, 2.18, 1.70, "Output",
        "[box + 5 scores]\n\nClass-agnostic NMS\n0+ / 1+ / 2+ / 3+ / other",
        face="green", edge="green_edge", title_size=10.6, body_size=7.7
    )
    # Multi-scale feature split.
    arrow(ax, (9.98, y + 1.32), (10.29, y + 1.91), connectionstyle="arc3,rad=-0.12")
    arrow(ax, (9.98, y + 1.32), (10.29, y + 0.66), connectionstyle="arc3,rad=0.12")
    # Semantic tower split into three decisions.
    arrow(ax, (11.88, y + 0.66), (12.13, y + 2.04), connectionstyle="arc3,rad=-0.16")
    arrow(ax, (11.88, y + 0.66), (12.13, y + 1.28), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (11.88, y + 0.66), (12.13, y + 0.52))
    # Box and semantic outputs are fused into the flat-compatible tensor.
    arrow(ax, (11.88, y + 1.91), (14.33, y + 2.04), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (13.94, y + 2.04), (14.33, y + 1.78), connectionstyle="arc3,rad=0.08")
    arrow(ax, (13.94, y + 1.28), (14.33, y + 1.28))
    arrow(ax, (13.94, y + 0.52), (14.33, y + 0.78), connectionstyle="arc3,rad=-0.08")
    arrow(ax, (16.78, y + 1.28), (17.19, y + 1.28))
    ax.text(
        13.30, y - 0.02,
        "Training loss: Box + Objectness + Tumor + Grade + DFL",
        ha="center", va="top", fontsize=8.6, color=COLORS["muted"]
    )


def style_canvas(ax, width: float, height: float) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_facecolor("white")


def save(fig, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def create_combined(image) -> None:
    fig = plt.figure(figsize=(20, 9.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    style_canvas(ax, 20, 9.8)
    ax.text(
        10, 9.48,
        "기존 Flat 모델 vs Hierarchical 모델 아키텍처",
        ha="center", va="center", fontsize=21, fontweight="bold", color=COLORS["navy"]
    )
    ax.text(
        10, 9.10,
        "동일한 YOLOv11-m Backbone/FPN · Detection head의 semantic decision 구조만 변경",
        ha="center", va="center", fontsize=11.2, color=COLORS["muted"]
    )
    ax.plot([0.35, 19.65], [4.83, 4.83], color="#CAD2D8", lw=1.2)
    draw_flat(ax, image, 5.45)
    draw_hierarchical(ax, image, 0.58)
    ax.text(
        10, 0.12,
        "Implementation source: nets/nn.py · Hierarchical inference: utils/hierarchical.py",
        ha="center", va="center", fontsize=8.5, color="#77858E"
    )
    save(fig, "flat_vs_hierarchical_architecture")


def create_flat(image) -> None:
    fig = plt.figure(figsize=(20, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    style_canvas(ax, 20, 5.0)
    ax.text(
        10, 4.62, "기존 Flat 5-class YOLOv11-m 아키텍처",
        ha="center", fontsize=20, fontweight="bold", color=COLORS["navy"]
    )
    ax.text(
        10, 4.27, "Cell detection confidence와 5개 class decision이 하나의 flat classification head에 결합",
        ha="center", fontsize=10.8, color=COLORS["muted"]
    )
    draw_flat(ax, image, 1.03, row_label=False)
    save(fig, "flat_model_architecture")


def create_hierarchical(image) -> None:
    fig = plt.figure(figsize=(20, 5.0), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    style_canvas(ax, 20, 5.0)
    ax.text(
        10, 4.62, "Hierarchical YOLOv11-m 아키텍처",
        ha="center", fontsize=20, fontweight="bold", color=COLORS["navy"]
    )
    ax.text(
        10, 4.27, "Cell 존재 여부, Tumor/Other gate, Tumor grade를 명시적으로 분리",
        ha="center", fontsize=10.8, color=COLORS["muted"]
    )
    draw_hierarchical(ax, image, 1.03, row_label=False)
    save(fig, "hierarchical_model_architecture")


def main() -> None:
    setup_font()
    if not SOURCE_IMAGE.exists():
        raise FileNotFoundError(f"Source patch not found: {SOURCE_IMAGE}")
    image = Image.open(SOURCE_IMAGE).convert("RGB")
    create_combined(image)
    create_flat(image)
    create_hierarchical(image)
    print(f"Created architecture diagrams in: {OUTPUT_DIR}")
    print(f"Embedded original patch: {SOURCE_IMAGE}")


if __name__ == "__main__":
    main()
