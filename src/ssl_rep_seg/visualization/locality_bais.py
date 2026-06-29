
import torch
import numpy as np
from torch.utils.data import DataLoader
from ..extractor import ViTExtractor
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from .utils.plot_utils import normalize_image, binarize_attn
from .utils.eval_utils import AttentionMIAnalyzer
import os
from dataclasses import dataclass
import pandas as pd


# ────── helper ──────────────────────────────────────────────────────
@dataclass
class ExtractorConfig:
    model_type: str
    model_size: str
    layer: int
    patch_size: int
    facet: str = "attn"
    stride: int = None          # defaults to patch_size if None
    norm_patchtoken: bool = False
    resize: list = (448,448)

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.patch_size


_SIZE_MAP = {"b": "base", "l": "large", "s": "small"}
_MODEL_RE = re.compile(r'^([a-z]+)_(.+?)_attn_l(\d+)$')

def parse_model_info(model_info: str, resize: int) -> ExtractorConfig:
    """
    'dino_b_attn_l11'  →  ExtractorConfig(model_type='dino', model_size='base',
                                           layer=11, patch_size=16, ...)
    """
    m = _MODEL_RE.match(model_info)
    if not m:
        raise ValueError(f"Cannot parse model_info: {model_info!r}")

    model_type = m.group(2)
    model_size = _SIZE_MAP[m.group(1)]
    layer      = int(m.group(3))
    patch_size = 14 if (
        model_type in ("dinov2", "dinov2_reg")
        or (model_type == "clip" and model_size == "large")
    ) else 16

    return ExtractorConfig(model_type=model_type, model_size=model_size,
                           layer=layer, patch_size=patch_size, resize = resize)



def write_attn_dict_to_csv(result_dict, model_info, layer_idx, save_path):

    head_avg_mi = result_dict["head_avg_mi"]
    overall_avg_mi = result_dict["overall_avg_mi"]
    num_heads = len(head_avg_mi)

    # safer than strip
    model_info_ex_layer = model_info.replace(f"_l{layer_idx}", "")

    csv_dir = os.path.join(save_path, "tabs")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f"{model_info_ex_layer}_attn_mi_all.csv")

    # Columns
    columns = ['model_info_ex_layer', 'layer', 'avghead'] + [f'head{i}' for i in range(num_heads)]

    # Row data
    row_data = [model_info_ex_layer, layer_idx, overall_avg_mi] + list(head_avg_mi)

    df = pd.DataFrame([row_data], columns=columns)

    # Write / append with controlled float precision
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False, float_format='%.4f')
    else:
        df.to_csv(csv_path, index=False, float_format='%.4f')
    
    print(f"Saved attention MI results to: {csv_path}")
    return 



# ────── attn_matrix ────────────────────────────────────────────────────
def collect_attn_matrices(
    loader: DataLoader,
    ext_cfg: ExtractorConfig,
    target_image_ids: list[int],
    batch_size: int,
) -> tuple[dict[int, np.ndarray], dict[str, torch.Tensor]]:
    """
    Returns:
        attn_maps:  {global_image_id: ndarray(num_heads, N, N)}
        saved_imgs: {"img": Tensor(K, C, H, W)}
    """
    extractor = ViTExtractor(**vars(ext_cfg))          # accepts dataclass directly

    target_ids    = torch.tensor(target_image_ids)
    batch_nums    = target_ids // batch_size
    batch_offsets = target_ids %  batch_size

    attn_maps = []
    raw_imgs = []
    collected, needed = 0, len(target_image_ids)

    for batch_idx, batch in enumerate(tqdm(loader)):
        if collected >= needed:
            break
        with torch.no_grad():
            imgs  = batch["img"].cuda()
            attn  = extractor.extract_descriptors(imgs, layer=ext_cfg.layer, facet="attn", include_cls= False).detach()
            # attn.shape  # (B, num_heads, N, N)

        for offset in batch_offsets[batch_nums == batch_idx].tolist():
            attn_maps.append(attn.cpu().numpy()[offset]) # num_heads, N, N
            raw_imgs.append(imgs.cpu().numpy()[offset]) # C, H, W
            collected += 1

        del imgs, attn
        torch.cuda.empty_cache()


    attn_maps = np.stack(attn_maps,  axis=0)
    raw_imgs = np.stack(raw_imgs, axis=0)
    return attn_maps, raw_imgs




def collect_all_attn_matrices(
    loader: DataLoader,
    model_info_list: list[str],
    target_image_ids: list[int],
    batch_size: int,
    resize: list,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Iterate over every model string, collect attention maps
    Returns:
        {model_info: {image_id: ndarray(num_heads, N, N)}}
    """
    data: dict[str, dict[int, np.ndarray]] = {}
    images_saved = False

    for model_info in model_info_list:
        print(f"\n── {model_info} ──")
        ext_cfg = parse_model_info(model_info, resize=resize)

        attn_matrix, saved_imgs = collect_attn_matrices(
            loader=loader,
            ext_cfg=ext_cfg,
            target_image_ids=target_image_ids,
            batch_size=batch_size,
        )
        data[model_info] = attn_matrix

        if not images_saved:
            data["imgs"]= saved_imgs 
            images_saved = True

    return data

# ────── attn locality bias ────────────────────────────────────────────────

def calculate_loc_bias_mi(
    loader: DataLoader,
    ext_cfg: ExtractorConfig,
    model_info: str,
    save_path: str
) -> tuple[dict[int, np.ndarray], dict[str, torch.Tensor]]:
    """
    Returns:
        attn_maps:  {global_image_id: ndarray(num_heads, N, N)}
        saved_imgs: {"img": Tensor(K, C, H, W)}
    """
    extractor = ViTExtractor(**vars(ext_cfg))          # accepts dataclass directly

    analyzer = AttentionMIAnalyzer(eps=1e-12)

    for batch_idx, batch in enumerate(tqdm(loader)):
        with torch.no_grad():
            imgs  = batch["img"].cuda()
            attn  = extractor.extract_descriptors(imgs, layer=ext_cfg.layer, facet="attn", include_cls= False).detach()
            # attn.shape  # (B, num_heads, N, N)
            analyzer.update(attn.cpu().numpy())

        del imgs, attn
        torch.cuda.empty_cache()

    results_dict = analyzer.finalize()
    write_attn_dict_to_csv(results_dict, model_info,  ext_cfg.layer, save_path)

    del analyzer
    return 
    


def save_all_locality_bias_mi(
    loader: DataLoader,
    model_info_list: list[str],
    resize: list,
    save_path: str
    ):

    for model_info in model_info_list:
        print(f"\n── {model_info} ──")
        ext_cfg = parse_model_info(model_info, resize=resize)
        calculate_loc_bias_mi(loader, ext_cfg, model_info, save_path)

    return 
   


# ─────── plot ───────────────────────────────────────────

def plot_attn_grid_samples(
    data: dict[str, dict[int, np.ndarray]],
    model_keys: list[str],
    save_path: str,
) -> None:
    
    raw_imgs = data["imgs"]

    rows, cols = len(raw_imgs), len(model_keys) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows), squeeze=False,facecolor="black")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    # Make every subplot black
    for ax in axes.flat:
        ax.set_facecolor("black")

    
    for r in range(rows):
        axes[r, 0].imshow(normalize_image(raw_imgs[r]))
        axes[r, 0].axis("off")
        for c, key in enumerate(model_keys, start=1):
            axes[r, c].imshow(binarize_attn(data[key][r]), cmap="gray_r")
            axes[r, c].axis("off")


    # Column headers(optional)
    axes[0, 0].set_title("Image", fontsize=12, pad=10, color="white")
    for c, key in enumerate(model_keys, start=1):
        axes[0, c].set_title(key.replace("_attn", ""), fontsize=12, pad=10, color="white")


    os.makedirs(os.path.join(save_path, "assets"), exist_ok=True)
    out_path = os.path.join(save_path, f"assets/attn_grid_samples.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")  # increase to 0.2, 0.3, etc.)
    plt.close(fig)
    print(f"Saved: {out_path}")



