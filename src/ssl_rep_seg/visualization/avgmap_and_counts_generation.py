import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from os.path import join
from .utils.checkpoint_utils import load_model, map_unsup_clusters_prediction
from .utils.eval_utils import PositionStatsAccumulator

def init_folders(save_path):
    os.makedirs(save_path, exist_ok=True)
    for sub in ["pos_class_counts", "avg_maps", "rep_cluster_pred", "img", "label"]:
        os.makedirs(os.path.join(save_path, sub), exist_ok=True)
    return



def run_single_model(
    segmenter,
    model_ckpt_path,
    test_dataloader,
    model_class_spatial_stats,
    gt_class_spatial_stats,
    rep_img_idxs,
    save_path
):
    model, assignments = load_model(segmenter, model_ckpt_path)
    model.cluster_metrics.reset()

    feature_extractor = model.net
    model_cfg         = model.cfg
    batch_size        = test_dataloader.batch_size

    # Precompute rep-image lookup: {batch_idx -> [local_offsets]}
    rep_lookup = {}
    if rep_img_idxs is not None:
        for img_idx in rep_img_idxs:
            rep_lookup.setdefault(img_idx // batch_size, []).append(img_idx % batch_size)

    rep_preds  = []
    rep_labels = []
    rep_imgs   = []
    total_N    = 0

    for i, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            img   = batch["img"].cuda()
            label = batch["label"].cuda()

            raw_feats = (
                feature_extractor
                .extract_descriptors(img, model_cfg.layer, model_cfg.facet) #include_cls=False
                .detach()
                .squeeze(1)
                .permute(0, 2, 1)
                .reshape(img.shape[0], -1, *feature_extractor.num_patches)
                .contiguous()
            )
            feats = F.interpolate(raw_feats, label.shape[-2:], mode="bilinear", align_corners=False)

            cluster_preds        = model.cluster_probe(feats, 2, log_probs=True).argmax(1).detach()
            cluster_preds_mapped = map_unsup_clusters_prediction(cluster_preds.cpu(), assignments).numpy()

            model_class_spatial_stats.update(cluster_preds_mapped)
            if gt_class_spatial_stats is not None:
                gt_class_spatial_stats.update(label.detach().cpu().numpy())

            # Grab all rep-image slices for this batch in one shot
            if i in rep_lookup:
                offsets = rep_lookup[i]
                rep_preds.append(cluster_preds_mapped[offsets])
                if gt_class_spatial_stats is not None:
                    rep_labels.append(label.cpu().numpy()[offsets])
                    rep_imgs.append(img.cpu().numpy()[offsets])

            total_N += label.shape[0]

    model_rep_cluster_pred = np.concatenate(rep_preds,  axis=0) if rep_preds  else None
    rep_labels             = np.concatenate(rep_labels, axis=0) if rep_labels else None
    rep_imgs               = np.concatenate(rep_imgs,   axis=0) if rep_imgs   else None

    if gt_class_spatial_stats is not None:
        np.save(join(save_path, "pos_class_counts", f"{model_cfg.dataset_name}_gt_counts.npy"), gt_class_spatial_stats.get_counts())
        np.save(join(save_path, "avg_maps", f"{model_cfg.dataset_name}_gt_avg_map.npy"), gt_class_spatial_stats.get_avg_var(total_N)[0])
        np.save(join(save_path, "img", f"{model_cfg.dataset_name}_imgs.npy"), rep_imgs)
        np.save(join(save_path, "label", f"{model_cfg.dataset_name}_labels.npy"), rep_labels)

    return model_class_spatial_stats, model_rep_cluster_pred, gt_class_spatial_stats


def run_all_models(
    test_dataloader,
    img_size,
    segmenter,
    model_ckpt_paths,
    model_info_list,
    dataset_name,
    n_bins,
    all_good_images,
    save_path
):
    """
    Core protocol loop: run segmentation inference for each model and save
    position counts, avg maps, and representative cluster predictions.

    Args:
        test_dataloader  : PyTorch DataLoader, built and owned by the caller
        model_ckpt_paths : list of checkpoint paths, resolved by the caller
        model_info_list  : list of model identifier strings, e.g. "b_mae_k_l0"
        dataset_name     : used for output file naming only
        n_bins           : number of spatial bins for PositionStatsAccumulator
        all_good_images  : indices of representative images for qualitative eval
        save_path        : root output directory, already initialized by caller
    """
    init_folders(save_path)

    n_classes = test_dataloader.dataset.n_classes
    H, W = img_size

    for idx, (model_ckpt_path, model_info) in enumerate(zip(model_ckpt_paths, model_info_list)):

        model_class_spatial_stats = PositionStatsAccumulator(n_bins, n_classes, H, W)
        gt_class_spatial_stats    = PositionStatsAccumulator(n_bins, n_classes, H, W) if idx == 0 else None

        model_class_spatial_stats, model_rep_cluster_pred, gt_class_spatial_stats = run_single_model(
            segmenter,
            model_ckpt_path,
            test_dataloader,
            model_class_spatial_stats,
            gt_class_spatial_stats,
            rep_img_idxs=all_good_images,
            save_path=save_path
        )

        np.save(
            join(save_path, "pos_class_counts", f"{dataset_name}_{model_info}_counts.npy"),
            model_class_spatial_stats.get_counts(),
        )
        np.save(
            join(save_path, "avg_maps", f"{dataset_name}_{model_info}_avg_map.npy"),
            model_class_spatial_stats.get_avg_var(len(test_dataloader.dataset))[0],
        )
        np.save(
            join(save_path, "rep_cluster_pred", f"{dataset_name}_{model_info}_cluster_preds.npy"),
            model_rep_cluster_pred,
        )

        del model_class_spatial_stats, gt_class_spatial_stats
