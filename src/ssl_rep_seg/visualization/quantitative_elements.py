# quantitative_elements.py
import os
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

from os.path import join
from .utils.eval_utils import PositionalBiasEvaluator
from .utils.plot_utils import _style_bar_axes, _add_group_separators


# ── data loading / computation ─────────────────────────────────────────────────

def load_or_compute_bar_data(save_path, dataset_name, model_info_list, n_classes, n_bins):
    """
    Load precomputed bar figure data if available, otherwise compute from
    positional class counts and cache to disk.

    Returns a list of dicts — GT first, then one per model — each with keys:
        model_info, global_mi, global_nmi, classwise_mi
    """
    cache_path = join(save_path, "bar_figure_data.npz")

    if os.path.isfile(cache_path):
        bar_data = np.load(cache_path, allow_pickle=True)["arr_0"].tolist()
        assert bar_data[0]["model_info"] == "GT", (
            f"First entry should be GT, got {bar_data[0]['model_info']}"
        )
        print(colored(
            f"Loaded cache — GT global_mi={bar_data[0]['global_mi']:.4f}, "
            f"global_nmi={bar_data[0]['global_nmi']:.4f}", "green"
        ))
        # for entry in bar_data:
        #     print(f"  {entry['model_info']:20s}  MI={entry['global_mi']:.4f}  NMI={entry['global_nmi']:.4f}")
        return bar_data

    return _compute_bar_data(save_path, dataset_name, model_info_list, n_classes, n_bins, cache_path)


def _compute_bar_data(save_path, dataset_name, model_info_list, n_classes, n_bins, cache_path):
    """Compute MI metrics from scratch and cache to disk."""
    mi_evaluator = PositionalBiasEvaluator(n_classes, n_bins=n_bins)

    gt_counts = np.load(join(save_path, "pos_class_counts", f"{dataset_name}_gt_counts.npy"))
    gt_dict   = mi_evaluator.fit_ground_truth(gt_masks=None, gt_counts=gt_counts)
    print(colored(f"GT global_mi={gt_dict['GT_MI']:.4f}, global_nmi={gt_dict['GT_NMI']:.4f}", "green"))

    bar_data = [{
        "model_info":   "GT",
        "global_mi":    gt_dict["GT_MI"],
        "global_nmi":   gt_dict["GT_NMI"],
        "classwise_mi": gt_dict["GT_classwise_MI"],
    }]

    for model_info in model_info_list:
        pred_counts = np.load(
            join(save_path, "pos_class_counts", f"{dataset_name}_{model_info}_counts.npy")
        )
        pred_dict = mi_evaluator.evaluate_predictions(pred_masks=None, pred_counts=pred_counts)
        print(f"  {model_info:20s}  MI={pred_dict['global_MI_pred']:.4f}  NMI={pred_dict['global_NMI_pred']:.4f}")

        bar_data.append({
            "model_info":   model_info,
            "global_mi":    pred_dict["global_MI_pred"],
            "global_nmi":   pred_dict["global_NMI_pred"],
            "classwise_mi": pred_dict["classwise_MI_pred"],
        })

    np.savez(cache_path, bar_data)
    return bar_data




# ── plot: total NMI bar ────────────────────────────────────────────────────────

def plot_total_nmi_bars(bar_data,  save_path):
    """One horizontal bar per model showing global NMI. GT in black, models in Blues."""
    n_models  = len(bar_data)
    h         = 0.2
    colors    = plt.cm.Blues(np.linspace(0.3, 0.9, n_models))
    colors[0] = [0, 0, 0, 1]
    gap_ratio =0.5 # hardcoded to visually separate GT from models; can be tuned if needed

    group_height = h * 1.2
    extra_top    = gap_ratio * group_height

    fig, ax = plt.subplots(figsize=(2.1, (n_models * group_height + extra_top)))

    for g, entry in enumerate(bar_data):
        group_rank = n_models - 1 - g
        y_pos      = group_rank * group_height + extra_top * 2 + h * 0.1 / 2
        if g >= 1:
            y_pos -= extra_top
        ax.barh(y_pos, entry["global_nmi"], height=h,
                color=colors[g], alpha=1.0, label=entry["model_info"])

    _style_bar_axes(ax, x_on_top=False)

    os.makedirs(os.path.join(save_path, "assets"), exist_ok=True)
    fig.savefig(os.path.join(save_path, f"assets/total_nmi_bars.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── plot: total seg performance bar ────────────────────────────────────────────────────────
def plot_segmentation_miou_bars(model_info_list, miou_list, save_path):
    """
    One horizontal bar per model showing mIoU performance.
    Models are colored on a Greens ramp, evenly spaced top to bottom.

    Args:
        model_info_list : list of model run keys (used as bar labels)
        miou_list       : list of mIoU values, one per model
        save_path       : root directory; figure saved under assets/
    """
    assert len(model_info_list) == len(miou_list), \
        "model_info_list and miou_list must have the same length"

    n_models     = len(model_info_list)
    h            = 0.2
    colors       = plt.cm.Greens(np.linspace(0.3, 0.9, n_models + 1))[1:]
    group_height = h * 1.2

    fig, ax = plt.subplots(figsize=(3 * 0.7, n_models * group_height))

    for g, (model_info, miou) in enumerate(zip(model_info_list, miou_list)):
        y_pos = (n_models - 1 - g) * group_height
        ax.barh(y_pos, miou, height=h,
                color=colors[g], alpha=1.0, label=model_info)

    _style_bar_axes(ax, x_on_top=False)

    os.makedirs(os.path.join(save_path, "assets"), exist_ok=True)
    fig.savefig(os.path.join(save_path, "assets/segmentation_miou_bars.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

# ── plot: classwise MI bars ────────────────────────────────────────────────────

def plot_classwise_mi_bars(bar_data, save_path, top_k=6, group_gap=0.03,
                           gap_ratio=0.39, class_color_lut=None):
    """
    For each model (GT first), show the top-k classes by MI as horizontal bars,
    colored by class. Groups are stacked vertically, one per model, with one
    extra gap between GT (first row) and the first model (second row).
    """
    n_models     = len(bar_data)
    h            = 0.008
    group_height = (8 * h + group_gap) * 1.06
    extra_top    = gap_ratio * group_height

    fig, ax = plt.subplots(figsize=(3, (n_models * group_height + extra_top) * 10))

    group_centres = []
    for g, entry in enumerate(bar_data):
        mi      = np.asarray(entry["classwise_mi"], dtype=float)
        top_idx = np.argsort(mi)[::-1][:top_k][::-1]
        colors  = [class_color_lut[cid]/255 for cid in top_idx]

        group_rank  = n_models - 1 - g
        group_base  = group_rank * group_height
        y_positions = group_base + np.arange(8 - top_k, 8) * h + h / 2 + group_gap

        # one extra gap pushes GT (g==0, top of chart) away from the first model
        if g == 0:
            group_base  += extra_top
            y_positions += extra_top

        ax.barh(y_positions, mi[top_idx], height=h,
                color=colors, alpha=1.0, label=entry["model_info"])
        group_centres.append(group_base + (top_k * h) / 2)

    sep_lines_pos = [l-(top_k * h) / 2 for l in group_centres]
    sep_lines_pos.insert(1,sep_lines_pos[0] -extra_top)
    _add_group_separators(ax, sep_lines_pos) # add separator lines between groups (optional)
    _style_bar_axes(ax, x_on_top=True)

    os.makedirs(os.path.join(save_path, "assets"), exist_ok=True)
    fig.savefig(os.path.join(save_path, f"assets/classwise_mi_bar_gr{gap_ratio}.png"),
                dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────────

def run_quantitative_plots(dataset_meta, model_info_list, miou_list, n_bins, top_k, gap_ratio, save_path):
    """
    Full pipeline to produce quantitative MI bar plots for one dataset.

    Produces:
        assets/total_nmi_bars.png
        assets/classwise_mi_bar_gr{gap_ratio}.png
        assets/segmentation_miou_bars.png

    Args:
        dataset_meta    : dict with keys:
                            - name        : str, used for file naming
                            - class_names : list of str
                            - class_colors      : list of [R, G, B] values (0-255)
        model_info_list : model run keys matching saved count files
        miou_list       : list of mIoU values for each model in model_info_list
        n_bins          : number of spatial bins used during count accumulation
        top_k           : number of top classes to show per model (max 8)
        gap_ratio       : vertical gap ratio between model groups
        save_path       : root directory containing pos_class_counts/
    """
    assert top_k <= 8, "top_k must be <= 8"

    dataset_name     = dataset_meta["name"]
    class_names      = dataset_meta["class_names"]
    n_classes        = len(class_names)
    class_color_lut  = dataset_meta["class_colors"]

    bar_data = load_or_compute_bar_data(save_path, dataset_name, model_info_list, n_classes, n_bins)

    plot_total_nmi_bars(bar_data, save_path)
    plot_classwise_mi_bars(bar_data, save_path,
                           top_k=min(top_k, n_classes),
                           group_gap=0.02,
                           gap_ratio=gap_ratio,
                           class_color_lut=class_color_lut)
    plot_segmentation_miou_bars(model_info_list, miou_list, save_path)