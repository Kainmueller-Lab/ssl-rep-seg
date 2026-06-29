import yaml
import os
import argparse
from ssl_rep_seg.visualization.avgmap_and_counts_generation import run_all_models
from ssl_rep_seg.visualization.qualitative_elements import run_qualitative_plots
from ssl_rep_seg.visualization.quantitative_elements import run_quantitative_plots
from ssl_rep_seg.visualization.utils.checkpoint_utils import resolve_model_checkpoints
from datasets.agumentation import get_class_labels, get_dataset_colormap
from datasets.dataloader import build_testdataloader
from training.train import LitUnsupervisedSegmenter


def build_dataset_meta(cfg: dict) -> dict:
    dataset_name = cfg["data"]["name"]
    class_names  = get_class_labels(dataset_name)
    n_classes    = len(class_names)
    return {
        "name":         dataset_name,
        "class_names":  class_names,
        "class_colors": get_dataset_colormap(dataset_name, n_classes),
    }


def build_model_info_list(cfg: dict) -> list:
    return [ "b_dino_k_l10", "b_dinov2_k_l11", "b_dinov2_reg_k_l11", "b_dinov3_k_l8", "b_ibot_k_l7", "b_mae_k_l10" ]

  
def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_meta    = build_dataset_meta(cfg)
    model_info_list = build_model_info_list(cfg)


    # wraps the save path with the current file's directory to ensure consistent relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, cfg["output"]["save_path"])

    test_dataloader  = build_testdataloader(cfg["data"]["root"], cfg["data"]["name"], cfg["data"]["resize"], 
                                        cfg["batch_size"], cfg["num_workers"])
    
    
    # if return_value is True, resolve_model_checkpoints will return both miou_list and model_ckpt_paths, otherwise it will return model_ckpt_paths, None
    model_ckpt_paths, miou_list = resolve_model_checkpoints(cfg["model"]["checkpoint_folder"], cfg["data"]["name"], model_info_list,
                                                 cfg["model"]["training"]["batch_size"], cfg["model"]["training"]["lr"],
                                                 cfg["model"]["training"]["num_batch_pcainit"], cfg["model"]["training"]["sample_interval"],
                                                 return_value=True)


    # Step 1: Generate avg maps, position counts, and cluster predictions per layer
    run_all_models(
        test_dataloader  = test_dataloader,
        img_size         = cfg["data"]["resize"], # tuple
        segmenter        = LitUnsupervisedSegmenter,
        model_ckpt_paths = model_ckpt_paths,
        model_info_list  = model_info_list,
        dataset_name     = dataset_meta["name"],
        n_bins           = cfg["evaluation"]["n_bins"],
        all_good_images  = cfg["evaluation"]["all_good_images"],
        save_path        = save_path,
    )

    # Step 2: Qualitative visualization — segmentation maps for representative images
    run_qualitative_plots(dataset_meta, model_info_list,  cfg["evaluation"]["all_good_images"], save_path)

    # Step 3: Quantitative visualization — NMI bars and mIoU across layers
    run_quantitative_plots(dataset_meta, model_info_list, miou_list,  cfg["evaluation"]["n_bins"], cfg["evaluation"]["top_k"], cfg["evaluation"]["gap_ratio"], save_path)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/protocol_2_compare_ssl/config.yaml")
    args = parser.parse_args()
    main(args.config)