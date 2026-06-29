import yaml
import os
import argparse
from ssl_rep_seg.visualization.locality_bais import save_all_locality_bias_mi
from datasets.dataloader import build_testdataloader



def build_model_info_list(cfg: dict) -> list:
    '''
    for layerwise case
    '''
    model_type = cfg["model"]["type"]
    model_size = cfg["model"]["size"]
    num_layers = 12 if model_size == "b" else 24  # please adjust it if you don't want to go through all layers, e.g. only 6 layers for b and 12 layers for l
    return [f"{model_size}_{model_type}_attn_l{i}" for i in range(num_layers)]

  
def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    model_info_list = build_model_info_list(cfg)


    # wraps the save path with the current file's directory to ensure consistent relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, cfg["output"]["save_path"])

    test_dataloader  = build_testdataloader(cfg["data"]["root"], cfg["data"]["name"], cfg["data"]["resize"], 
                                        cfg["batch_size"], cfg["num_workers"])
    
    
    
    save_all_locality_bias_mi(test_dataloader, model_info_list, cfg["data"]["resize"], save_path)
  
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/locality_bias/config2.yaml")
    args = parser.parse_args()
    main(args.config)
   