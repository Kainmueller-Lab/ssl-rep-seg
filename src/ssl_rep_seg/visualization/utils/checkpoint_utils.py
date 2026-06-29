import re
import os
from termcolor import colored
import torch
import numpy as np
#################################
def find_max_miou_checkpoint(file_name_list, prefer_norm_ckpt=False, return_value=False):
    """
    Select checkpoint with highest mIoU.
    If prefer_norm_ckpt=True, only consider files ending with 'norm.ckpt'.
    """

    pattern = r'clustermiou-(\d+(?:\.\d+)?)'

    candidates = file_name_list
    if prefer_norm_ckpt:
        candidates = [f for f in candidates if f.endswith("norm.ckpt")]

    max_miou = -1
    best_file = None

    for filename in candidates:
        match = re.search(pattern, filename)
        if match:
            miou = float(match.group(1))
            if miou > max_miou:
                max_miou = miou
                best_file = filename

    if return_value:
        return best_file, max_miou
    else:
        return best_file, None


def parse_model_info(model_info: str):
    model_info_pattern = re.compile(
    r"^(?P<model_size>[bl])_"
    r"(?P<model_name>[a-zA-Z0-9_]+)_"
    r"(?P<facet>[kqvt])_"
    r"l(?P<layer>\d+)$"
)
    match =  model_info_pattern.match(model_info)
    if not match:
        raise ValueError(f"Invalid model_info format: {model_info}")

    return (
        match.group("model_size"),
        match.group("model_name"),
        match.group("facet"),
        int(match.group("layer")),
    )

def resolve_model_checkpoints(
    checkpoint_folder,
    dataset_name,
    model_info_list,
    batch_size,
    lr,
    num_batch_pcainit,
    sample_interval,
    return_value=False
):
    model_ckpt_paths = []
    miou_list=[] if return_value else None

    for model_info in model_info_list:
        folder = os.path.join(
            checkpoint_folder,
            f"{dataset_name}_{model_info}_lr{lr}_bs{batch_size}_pca{num_batch_pcainit}-{sample_interval}",
        )

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Checkpoint folder not found: {folder}")

        model_size, _, facet, layer = parse_model_info(model_info)
        if (layer == 11 and facet== "t" and model_size == "b") or (layer == 23 and facet== "t" and model_size == "l"):
            prefer_norm_ckpt = True
            print(colored(f"Looking for norm.ckpt for {model_info}", "green"))
        else:
            prefer_norm_ckpt = False
      
        ckpts = [f for f in os.listdir(folder) if f.endswith(".ckpt")]

        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in: {folder}")

        best_ckpt, miou_value = find_max_miou_checkpoint( ckpts, prefer_norm_ckpt=prefer_norm_ckpt,return_value=return_value)
        model_ckpt_paths.append(os.path.join(folder, best_ckpt))
        if return_value:
            miou_list.append(miou_value)

    return model_ckpt_paths, miou_list


######################################

def load_model(segmenter, model_path):
    print(f"Loading model from checkpoint: {model_path}")
    model = segmenter.load_from_checkpoint(model_path, weights_only=False)
    model.eval().cuda()

    assignments_path = model_path.replace(
        "checkpoints", "assignments"
    ).replace(".ckpt", "_assignments.pt")

    assignments = torch.load(assignments_path)

    return model, assignments


################################### assignment
def map_unsup_clusters_prediction(clusters, assignments, extra_clusters=0, n_classes=27, merge_extra_clusters=False, stats=None):
    assert isinstance(assignments, tuple) and len(assignments) == 2, \
                    "assignments should be a tuple of (matched_clusters, matched_classes)"
    if extra_clusters == 0:
        cluster_to_class = assignments[1]
        return  cluster_to_class[clusters]
    else:
        if not merge_extra_clusters:
            missing = sorted(list(set(range(n_classes + extra_clusters)) - set(assignments[0])))
            cluster_to_class = assignments[1]
    
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
           
            return cluster_to_class[clusters]
        
        else:
            assert stats is not None, "stats must be provided when merge_extra_clusters is True"

            cluster_to_class = -1 * torch.ones(n_classes + extra_clusters, dtype=torch.long)  # Initialize
            # First Hungarian matches
            for cluster_idx, class_idx in zip(assignments[0], assignments[1]):
                cluster_to_class[cluster_idx] = class_idx

            # For unmatched clusters
            missing = sorted(list(set(range(n_classes + extra_clusters)) - set(assignments[0])))
            for i in missing:
                cluster_to_class[i] = stats[i].argmax()
            return cluster_to_class[clusters]