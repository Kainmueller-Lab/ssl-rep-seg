
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from .utils.plot_utils import (colorize_avg_map, 
                                _fill_overview_column, 
                                _fill_sample_columns, 
                                _apply_row_labels, 
                                _shift_overview_column,
                                _shift_overview_row,
                                build_color_luts)



def plot_segmentation_results(
    num_rep_samples,
    model_info_list,
    data,
    imgs=None,
    gt_masks=None,
    class_color_lut_rgb=None,
    n_classes=27,
    selected_classes=None,
    save_path=None,
):
    """
    Plot representative segmentation predictions alongside spatial average maps.

    Layout:
        columns : one per representative sample + one overview column
        rows    : [image, GT label] (optional) + one row per model in model_info_list

    Args:
        num_rep_samples   : number of representative sample images to show
        model_info_list   : list of model run keys (used to index data and label rows)
        data              : dict with keys "cluster_preds" and "avg_maps", each
                            indexed by dataset_name then model key / "gt"
        imgs              : (N, C, H, W) array of representative images, optional
        gt_masks          : (N, H, W) integer array of GT labels, optional
        class_color_lut_rgb    : {class_id: [r,g,b]} for overview blending
        n_classes         : total number of classes
        selected_classes  : list of class ids to include in overview; defaults to all
        save_path         : directory to save the figure under assets/
    """
    cluster_preds = data["cluster_preds"]
    avg_maps      = data["avg_maps"]
    model_keys    = model_info_list
    has_img_rows  = imgs is not None and gt_masks is not None
    selected_ids  = np.array(selected_classes if selected_classes is not None
                             else range(n_classes), dtype=int)

    # Build overview maps
    overview_maps = {
        "gt": colorize_avg_map(avg_maps["gt"], selected_ids, class_color_lut_rgb),
        **{k: colorize_avg_map(avg_maps[k], selected_ids, class_color_lut_rgb)
           for k in model_keys},
    }

    # Figure layout
    n_rows    = (2 if has_img_rows else 0) + len(model_keys)
    n_cols    = num_rep_samples + 1
    start_row = 2 if has_img_rows else 0
    ov_col    = n_cols - 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.8 * n_rows))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # Ensure axes is always 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    _fill_sample_columns(axes, num_rep_samples, model_keys, cluster_preds,
                         imgs, gt_masks, class_color_lut_rgb, start_row)
    # overview_maps are the aggregated avg maps for GT and each model
    _fill_overview_column(axes, ov_col, model_keys, overview_maps, start_row, has_img_rows)
    # optionally add row labels like "GT", "b_dino_k_l11", etc.
    _apply_row_labels(axes, model_keys, has_img_rows)
    if has_img_rows:
        _shift_overview_row(axes, n_cols, n_rows)
    _shift_overview_column(axes, ov_col, n_rows)
    

    os.makedirs(os.path.join(save_path, "assets"), exist_ok=True)
    out_path = os.path.join(save_path, f"assets/seg_results.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)

    return 


def load_qualitative_data(root, dataset_name, model_info_list):
    data = {
        "avg_maps":        defaultdict(dict),
        "cluster_preds":   defaultdict(dict),
    }

    for model_info in ["gt"] + model_info_list:
       
        data["avg_maps"][model_info] = np.load(
            os.path.join(root, f"avg_maps/{dataset_name}_{model_info}_avg_map.npy")
        )

        if model_info != "gt":
            data["cluster_preds"][model_info] = np.load(
                os.path.join(root, f"rep_cluster_pred/{dataset_name}_{model_info}_cluster_preds.npy")
            )
        else:
            pass

    imgs      = np.load(os.path.join(root, f"img/{dataset_name}_imgs.npy"))
    gt_masks  = np.load(os.path.join(root, f"label/{dataset_name}_labels.npy"))

    return data, imgs, gt_masks


# ── colormap preview ───────────────────────────────────────────────────────────

def plot_colormap(class_cmap, n_classes, dataset_name, save_path):
    """Save a horizontal colormap strip for quick visual reference."""
    fig, ax = plt.subplots(figsize=(12, 1.4))
    ax.imshow([range(n_classes)], cmap=class_cmap, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.join(save_path, "assets"), exist_ok=True)
    fig.savefig(os.path.join(save_path, f"assets/{dataset_name}_cmap.png"),
                bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────────

def run_qualitative_plots(dataset_meta, model_info_list, all_good_images, save_path):
    """
    Full pipeline to produce qualitative segmentation result plots for one dataset.

    Loads precomputed predictions and avg maps, builds color LUTs, saves a
    colormap preview, then plots representative samples alongside average maps.

    Args:
        dataset_meta    : dict with keys:
                            - name        : str, used for file naming
                            - class_names : list of str
                            - class_colors      : list of [R, G, B] values (0-255)
        model_info_list : list of model run keys matching saved checkpoint names
        all_good_images : list of representative image indices
        save_path       : root directory containing avg_maps/, rep_cluster_pred/, etc.
    """
    n_classes = len(dataset_meta["class_names"])
    dataset_name = dataset_meta["name"]
    cmap_rgb255 = dataset_meta["class_colors"]


    data, imgs, gt_masks = load_qualitative_data(save_path, dataset_name, model_info_list)

    color_lut_rgb, cmap  = build_color_luts(cmap_rgb255, n_classes)

    plot_colormap(cmap, n_classes, dataset_name, save_path)

    plot_segmentation_results(
        num_rep_samples     = len(all_good_images),
        model_info_list     = model_info_list,
        data                = data,
        imgs                = imgs,
        gt_masks            = gt_masks,
        class_color_lut_rgb = color_lut_rgb,
        n_classes           = n_classes,
        selected_classes    = None,
        save_path           = save_path
    )
