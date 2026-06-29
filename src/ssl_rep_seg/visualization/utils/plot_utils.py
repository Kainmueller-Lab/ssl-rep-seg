###############################
# funcs for the major plotting of qualitative elements inside the protocol
###############################
import numpy as np
import matplotlib.pyplot as plt
# ── helpers ────────────────────────────────────────────────────────────────────

def colorize_segmentation(mask, color_lut_rgb):
    """Map integer class mask -> RGB image using a {class_id: (r,g,b)} LUT."""
    rgb = np.zeros((*mask.shape, 3), dtype=float)
    for cid, color in color_lut_rgb.items():
        rgb[mask == cid] = color
    return rgb


def colorize_avg_map(avg_map, selected_ids, color_lut_rgb):
    """
    Blend per-class average frequency maps into a single RGB image.
    Each pixel color is a weighted average of the selected class colors,
    weighted by how often that class appears at that position.
    """
    weights    = avg_map[selected_ids].astype(float)
    total      = weights.sum(axis=0, keepdims=True)
    total_safe = np.where(total == 0, 1.0, total)
    weights    = weights / total_safe

    colors = np.array([color_lut_rgb[int(c)] for c in selected_ids], dtype=float)
    rgb    = np.tensordot(weights, colors, axes=(0, 0))
    rgb[total.squeeze(0) == 0] = 0.0
    return np.clip(rgb, 0, 1)


def normalize_image(img):
    """CHW or HWC float image -> HWC normalized to [0, 1]."""
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.moveaxis(img, 0, -1)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


# ── axes fillers ─────────────────────────────────────────────────────────────

def _fill_sample_columns(axes, num_samples, model_keys, cluster_preds,
                         imgs, gt_masks, color_lut_rgb, start_row):
    for c in range(num_samples):
        if imgs is not None and gt_masks is not None:
            axes[0, c].imshow(normalize_image(imgs[c]))
            axes[0, c].axis("off")
            axes[1, c].imshow(colorize_segmentation(gt_masks[c], color_lut_rgb), interpolation="nearest")
            axes[1, c].axis("off")

        for r, k in enumerate(model_keys, start=start_row):
            axes[r, c].imshow(colorize_segmentation(cluster_preds[k][c], color_lut_rgb), interpolation="nearest")
            axes[r, c].axis("off")


def _fill_overview_column(axes, ov_col, model_keys, overview_maps, start_row,
                          has_img_rows):
    if has_img_rows:
        axes[0, ov_col].axis("off")
        axes[1, ov_col].imshow(overview_maps["gt"])
        axes[1, ov_col].axis("off")

    for r, k in enumerate(model_keys, start=start_row):
        axes[r, ov_col].imshow(overview_maps[k])
        axes[r, ov_col].axis("off")


# ── layout helpers ─────────────────────────────────────────────────────────────

def _apply_row_labels(axes, model_keys, has_img_rows):
    row_labels = (["image", "label"] + model_keys) if has_img_rows else model_keys
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, rotation=90, fontsize=10)


def _shift_overview_column(axes, ov_col, n_rows, gap=0.05):
    for r in range(n_rows):
        p = axes[r, ov_col].get_position()
        axes[r, ov_col].set_position([p.x0 + gap, p.y0, p.width, p.height])

def _shift_overview_row(axes, n_cols, n_rows, gap=0.03):
    for rr in range(2, n_rows):
        for cc in range(n_cols):
            p = axes[rr, cc].get_position()
            axes[rr, cc].set_position([p.x0, p.y0 - gap, p.width, p.height])



# ── shared axis styling ────────────────────────────────────────────────────────

def _style_bar_axes(ax, x_on_top=False):
    """Apply shared cosmetics: spines, ticks, twin axis."""
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_xlim(left=0)
    ax.set_yticks([])
    ax.margins(y=0.02)
    ax.spines["right"].set_visible(False)

    if x_on_top:
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", labelsize=8, top=True, labeltop=True,
                       bottom=False, labelbottom=False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
    else:
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()
        ax.tick_params(axis="x", labelsize=8, top=False, labeltop=False,
                       bottom=True, labelbottom=True)
        ax.spines["top"].set_visible(False)



def _add_group_separators(ax, sep_lines_pos):
    """Draw dashed horizontal separator lines between model groups."""
    for g, sep in enumerate(sep_lines_pos):
        ax.axhline(sep, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)




# ── color helpers ──────────────────────────────────────────────────────────────

def build_color_luts(dataset_cmap_rgb255, n_classes):
    """
    From a (n_classes, 3) uint8 colormap array, build all color lookup tables
    needed for segmentation visualization.

    Returns:
        class_color_lut_rgb    : {class_id: (r, g, b)} in [0, 1]
        class_color_lut_dimmed : {class_id: hex string}
        class_cmap             : ListedColormap (full brightness)
    """
    hex_colors    = [plt.matplotlib.colors.to_hex(c / 255.0) for c in dataset_cmap_rgb255]
    class_cmap    = plt.matplotlib.colors.ListedColormap(hex_colors, name="seg_cmap")
    color_lut_rgb = {cid: tuple(dataset_cmap_rgb255[cid] / 255.0) for cid in range(n_classes)}
    return color_lut_rgb,  class_cmap

# ── attn helpers ─────────────────────────────────────────────

def binarize_attn(attn: np.ndarray, head_q: float = 0.95, agg_q: float = 0.95) -> np.ndarray:
    binarized = np.zeros_like(attn, dtype=np.uint8)
    for h in range(len(attn)):
        q = np.quantile(attn[h], head_q, axis=-1, keepdims=True)
        binarized[h] = (attn[h] >= q).astype(np.uint8)
    agg = binarized.sum(axis=0)
    return (agg >= np.quantile(agg, agg_q, axis=-1, keepdims=True)).astype(np.uint8)


