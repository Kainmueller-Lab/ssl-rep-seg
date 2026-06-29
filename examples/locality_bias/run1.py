import yaml
import os
import argparse
from ssl_rep_seg.visualization.locality_bais import collect_all_attn_matrices, plot_attn_grid_samples
from datasets.dataloader import build_testdataloader



def build_model_info_list(cfg: dict) -> list:
    return [ "b_dino_attn_l11", "b_mugs_attn_l11", "b_mocov3_attn_l11", 
            "b_mae_attn_l11","b_ibot_attn_l11", "b_dinov2_reg_attn_l11", "b_dinov3_attn_l11" ]

  
def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    model_info_list = build_model_info_list(cfg)


    # wraps the save path with the current file's directory to ensure consistent relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, cfg["output"]["save_path"])

    test_dataloader  = build_testdataloader(cfg["data"]["root"], cfg["data"]["name"], cfg["data"]["resize"], 
                                        cfg["batch_size"], cfg["num_workers"])
    
    
    
    all_data = collect_all_attn_matrices(
        loader           = test_dataloader,
        model_info_list  = model_info_list,
        target_image_ids = cfg["evaluation"]["all_good_images"],
        batch_size       = cfg["batch_size"],
        resize           = cfg["data"]["resize"]
    )

    plot_attn_grid_samples(
        data        = all_data,
        model_keys  = model_info_list,
        save_path   = save_path,
    )

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/locality_bias/config1.yaml")
    args = parser.parse_args()
    main(args.config)