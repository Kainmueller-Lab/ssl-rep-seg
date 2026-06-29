
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import glob
from scipy.ndimage import distance_transform_edt

def chunkwise_inner_product(x, normed_clusters, chunk_size=1024):
    """
    Compute inner products between x and normed_clusters in chunks.
    This avoids storing all normalized x in memory.
    
    Args:
        x: (N, C) tensor
        normed_clusters: (n_classes, C) tensor (already normalized)
        chunk_size: int, how many rows of x to process at a time

    Returns:
        inner_products: (N, n_classes)
    """
    results = []
    for i in range(0, x.size(0), chunk_size):
        chunk = x[i:i+chunk_size]                 # (chunk_size, C)
        chunk = F.normalize(chunk, dim=1)         # normalize this chunk
        out = torch.matmul(chunk, normed_clusters.T)  # (chunk_size, n_classes)
        results.append(out)
    return torch.cat(results, dim=0)

class ClusterLookup(nn.Module):
    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha=None, log_probs=False):
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (N, C) chunkwise_inner_product
        Returns:
            cluster_loss: scalar tensor
            cluster_probs: assignment probabilities (soft or hard one-hot)
        """
        # normalize cluster centroids
        normed_clusters = F.normalize(self.clusters, dim=1)

        # normalize input features and compute inner products
        if x.dim() == 4:  # (B, C, H, W)
            normed_features = F.normalize(x, dim=1)
            inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        elif x.dim() == 2:  # (N, C)
            inner_products = chunkwise_inner_product(x, normed_clusters, chunk_size=1024)
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        # compute assignments
        if alpha is not None:
            cluster_probs = F.softmax(inner_products * alpha, dim=1)
        else:
            # hard one-hot assignment
            if x.dim() == 4:
                cluster_probs = F.one_hot(
                    inner_products.argmax(dim=1),
                    num_classes=self.n_classes
                ).permute(0, 3, 1, 2).float()
            else:  # (N, C)
                cluster_probs = F.one_hot(
                    inner_products.argmax(dim=1),
                    num_classes=self.n_classes
                ).float()

        # cluster loss: encourage feature-centroid alignment
        cluster_probs = cluster_probs.detach()
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()

        if log_probs:
            return F.log_softmax(inner_products * (alpha if alpha else 1.0), dim=1)
        else:
            return cluster_loss, cluster_probs

def build_cluster_probes( dim, target_n_classes):
    """Build the train/eval cluster-lookup probes
    """
  
    train_probe =  ClusterLookup(dim, target_n_classes)
    eval_probe =  ClusterLookup(dim, target_n_classes)
   
    return train_probe, eval_probe

def build_unsupervised_metrics(cfg, class_labels, metrics_cls):
    """Construct (cluster, linear, train_cluster, train_linear) metric trackers
    """
    extra_kwargs = {}
    cluster_metrics = metric_cls("test/cluster/", class_labels, cfg.extra_clusters, True, **extra_kwargs)
    linear_metrics = metric_cls("test/linear/", class_labels, 0, False, **extra_kwargs)
    train_cluster_metrics = metric_cls("train/cluster/", class_labels, cfg.extra_clusters, True, **extra_kwargs)
    train_linear_metrics = metric_cls("train/linear/", class_labels, 0, False, **extra_kwargs)
    return cluster_metrics, linear_metrics, train_cluster_metrics, train_linear_metrics



 
# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
 
def fill_boundary_with_nearest(label):
    """Replace boundary pixels (value -1) with the nearest non-boundary label,
    per-image in the batch."""
    label_filled = label.clone()
    for b in range(label.shape[0]):
        lbl_np = label[b].cpu().numpy()
        boundary = lbl_np == -1
        if boundary.any():
            _, indices = distance_transform_edt(boundary, return_indices=True)
            lbl_np[boundary] = lbl_np[tuple(indices[:, boundary])]
        label_filled[b] = torch.from_numpy(lbl_np)
    return label_filled
 
 
def generate_patch_res_labels(
    label,
    n_classes,
    patch_size,
    bg_weight=0.3,
    fg_weight=0.7,
):
    """Downsample a pixel-resolution label map to patch resolution via a
    weighted majority vote (background down-weighted relative to foreground
    classes), filling boundary pixels.
    """
    
    label = fill_boundary_with_nearest(label)
 
    class_weights = torch.full((n_classes,), fg_weight, device=label.device)
    class_weights[0] = bg_weight
 
    one_hot = F.one_hot(label, num_classes=n_classes).float()  # B x H x W x C
    pooled = F.avg_pool2d(
        one_hot.permute(0, 3, 1, 2),
        kernel_size=(patch_size, patch_size),
        stride=patch_size,
    )
    weighted = class_weights[None, :, None, None] * pooled
    return torch.argmax(weighted, dim=1)  # B x h' x w'
 


# ---------------------------------------------------------------------------
# Dense feature extraction
# ---------------------------------------------------------------------------
 
def extract_chunk_features(net, img_chunk, cfg):
    """Extract dense ViT features for a single chunk of images.
    Returns shape (chunk_B, d, h', w')."""
    descriptors = net.extract_descriptors(img_chunk, cfg.layer, cfg.facet, include_cls=False)
    h_patches, w_patches = net.num_patches
    return descriptors.squeeze(1).permute(0, 2, 1).reshape(descriptors.size(0), -1, h_patches, w_patches)
 
 
def extract_dense_features(net, img, cfg):
    """Extract dense ViT features for a full batch, processed in chunks of
    cfg.sub_batchsize to bound peak memory. Returns shape (B, d, h', w')."""
    chunks = [
        extract_chunk_features(net, img[start:start + cfg.sub_batchsize], cfg)
        for start in range(0, img.shape[0], cfg.sub_batchsize)
    ]
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Foreground masking
# ---------------------------------------------------------------------------
 
def compute_fg_mask(hres_fg_mask, patch_size, threshold=0.3, return_hres=False):
    """Resolve a foreground mask at either full image resolution or pooled
    patch resolution.
 
    hres_fg_mask: foreground mask at full image resolution, shape (B, H, W).
    return_hres: if True, return the mask as-is (cast to bool); otherwise
        average-pool to patch resolution and threshold.
    """
    if return_hres:
        return hres_fg_mask.bool()
    pooled = F.avg_pool2d(
        hres_fg_mask.float().unsqueeze(1),
        kernel_size=(patch_size, patch_size),
        stride=patch_size
    )
    return (pooled >= threshold).squeeze(1) # a boolean tensor


# ---------------------------------------------------------------------------
# Cluster predictions
# ---------------------------------------------------------------------------
 
def resolve_cluster_assignment(cluster_logits, include_bg):
    """Convert cluster-probe logits to hard cluster ids, shifting by one when
    a background cluster occupies index 0."""
    preds = cluster_logits.argmax(1)
    return preds + 1 if not include_bg else preds
 
 
def scatter_cluster_predictions(select_cluster_preds, fg_mask, template):
    """Scatter foreground-only cluster predictions back into a full tensor
    shaped like `template`; background positions stay 0."""
    cluster_preds = torch.zeros_like(template)
    cluster_preds[fg_mask] = select_cluster_preds.to(cluster_preds.device)
    return cluster_preds
 


# ---------------------------------------------------------------------------
# PCA cluster initialization
# ---------------------------------------------------------------------------

def collect_pca_features(net, dataloader, cfg, include_bg):
    """Collect a pool of dense features (optionally foreground-masked and
    subsampled) from up to cfg.num_batch_pcainit dataloader batches, for
    PCA-based cluster/linear probe initialization.
    """
    collected = []
    for idx, batch in enumerate(dataloader):
        if idx >= cfg.num_batch_pcainit:       
            break
        img = batch["img"].to(net.device) # get the image batch (still on put before calling training_step) and move to the same device as the network
        for start in range(0, img.shape[0], cfg.sub_batchsize):
            chunk_feats = extract_chunk_features(net, img[start:start + cfg.sub_batchsize], cfg)
            flat = chunk_feats.permute(0, 2, 3, 1).reshape(-1, chunk_feats.size(1))
 
            if not include_bg:
                mask_chunk = batch["mask"][start:start + cfg.sub_batchsize] if cfg.use_bg else None
                fg_mask = compute_fg_mask(mask_chunk, cfg.patch_size).detach()
                flat = flat[fg_mask.reshape(-1)]

            collected.append(flat.detach().cpu()[::cfg.sample_interval, :])
 
    return torch.cat(collected, dim=0)


def compute_pca_cluster_init(features, target_n_classes, n_classes, niter=200):
    """Run low-rank PCA over collected features and derive initial cluster-probe
    and linear-probe weights from the leading principal components.
 
    Returns (cluster_init_weights, linear_init_weights):
      cluster_init_weights: (target_n_classes, d)
      linear_init_weights:  (n_classes, d, 1, 1) — ready to copy into a Conv2d weight.
    """
    _, _, v = torch.pca_lowrank(features, q=2 * target_n_classes, niter=niter)
    cluster_weights = v[:, :target_n_classes].permute(1, 0)
    linear_weights = v[:, :n_classes].permute(1, 0).unsqueeze(-1).unsqueeze(-1)
    return cluster_weights, linear_weights



# ---------------------------------------------------------------------------
# Validation bookkeeping
# ---------------------------------------------------------------------------
 
def update_best_metrics(tb_metrics, best_metrics, std_logger, epoch, global_step):
    """Log every validation metric and update the running best-metrics dict in place."""
    for k, v in tb_metrics.items():
        std_logger.info(f"Validation metrics at epoch {epoch}-{global_step}: {k}: {v:.2f}")
        if k not in best_metrics or v > best_metrics[k]:
            best_metrics[k] = v
 
 
def build_checkpoint_name(cfg, cluster_miou, step=None):
    """Build a checkpoint filename. The initial (step=None) checkpoint is
    tagged "pca_init" and additionally flags norm_patchtoken; later checkpoints
    are tagged by global step.
    """
    if step is None:
        suffix = "_norm" if cfg.norm_patchtoken else ""
        return f"pca_init-ex_class{cfg.extra_clusters}-clustermiou-{cluster_miou:.2f}{suffix}.ckpt"
    return f"step{step}-ex_class{cfg.extra_clusters}-clustermiou-{cluster_miou:.2f}.ckpt"
 
 
def build_checkpoint_and_assignments_fullpath(cfg, ckpt_name, assignments):
    """Save a Lightning checkpoint and its cluster-to-class assignment tensor
    side by side in the run's checkpoint directory."""
    ckpt_dir = os.path.join(cfg.output_root, "checkpoints", cfg.run_name)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    # trainer.save_checkpoint(ckpt_path)
    assign_dir = os.path.join(cfg.output_root, "assignments", cfg.run_name)
    assignment_path = os.path.join(assign_dir, ckpt_name.replace(".ckpt", "_assignments.pt"))
    # torch.save(assignment, assignment_path)
    return ckpt_path, assignment_path
 
 
def prune_old_assignments(cfg, extra_clusters):
    """Delete previously saved step-checkpoint assignment files for this
    extra-cluster count (the pca_init checkpoint doesn't match the "step*"
    prefix, so it's left untouched)."""
    assign_dir = os.path.join(cfg.output_root, "assignments", cfg.run_name)
    pattern = os.path.join(assign_dir, f"step*ex_class{extra_clusters}*_assignments.pt")
    for path in glob.glob(pattern):
        os.remove(path)
 
 

def configure_token_norm_patch(args):
    '''
    Set args.norm_patchtoken to True for the final ViT layer of token-only probes
    '''
    args.norm_patchtoken = (
        (args.model_size == "base" and args.facet == "token" and args.layer == 11)
        or
        (args.model_size == "large" and args.facet == "token" and args.layer == 23)
    )
    return args



# ---------------------------------------------------------------------------
# Derived parameters
# ---------------------------------------------------------------------------
    
def derive_runtime_params(args, scalar_log_freq=10, train_metrics_log_interval=100,
                        max_sub_batchsize=8, wandb_dotenv_path="wandb.env"):
    """Compute secondary parameters that follow deterministically from raw CLI args.

    Fills in: dataset-specific loader/image-set conventions, logging cadences,
    sub-batch size, the PCA-init batch count adjusted for the sampling interval,
    and the run name. Mutates and returns args so the call site reads as:

        args = derive_runtime_params(parser.parse_args())
    """
    # Dataset-specific conventions
    args.train_image_set = "train_aug" if args.dataset_name == "VOC2012" else "train"

    # Logging cadences (kept as named params so callers can override if needed)
    args.scalar_log_freq = scalar_log_freq
    args.train_metrics_log_interval = train_metrics_log_interval

    # Clamp n_images to actual batch size (can't log more than we have)
    args.n_images = min(args.n_images, args.batch_size)
    args.sub_batchsize = min(max_sub_batchsize, args.batch_size)

    # Scale PCA batch count to compensate for patch sub-sampling
    if args.sample_interval > 1:
        args.num_batch_pcainit *= args.sample_interval
        print(
            f"PCA init batch count scaled to {args.num_batch_pcainit} "
            f"(sample_interval={args.sample_interval})"
        )

    args.wandb_dotenv_path = wandb_dotenv_path
    args.run_name =   f"{args.dataset_name}_{args.model_size[0]}_{args.model_type}_{args.facet[0]}_l{args.layer}"  + \
                                        f"_lr{args.cluster_prob_lr}_bs{args.batch_size}" + \
                                        f"_pca{args.num_batch_pcainit}-{args.sample_interval}"
    return args