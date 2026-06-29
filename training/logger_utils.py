import logging
import os,sys
from pytorch_lightning.loggers import WandbLogger
from dotenv import load_dotenv 
import wandb
from matplotlib import pyplot as plt
import torch.nn.functional as F
import torch
import pandas as pd
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
 
def setup_std_logger(output_root, log_level=logging.INFO):
    """Configure the root logger to write to both stdout and a rotating log
    file under output_root/run.log. Returns the module-level logger."""
    os.makedirs(output_root, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=log_level,
        handlers=[
            logging.FileHandler(os.path.join(output_root, "run.log"), mode='a'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
 
def setup_output_dirs(cfg):
    """Create the full output-directory tree for a run and return the paths
    as a plain dict so callers don't have to reconstruct them."""
    dirs = {
        "data":       os.path.join(cfg.output_root, "data"),
        "logs":       os.path.join(cfg.output_root, "logs"),
        "checkpoints":os.path.join(cfg.output_root, "checkpoints"),
        "assignments":os.path.join(cfg.output_root, "assignments"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
 
    # Per-run subdirectories (keyed by run_name to avoid collisions)
    os.makedirs(os.path.join(dirs["checkpoints"], cfg.run_name), exist_ok=True)
    os.makedirs(os.path.join(dirs["assignments"],  cfg.run_name), exist_ok=True)
 
    # Stable log prefix used for the run's full name
    cfg.full_name = "{}/{}_{}".format(dirs["logs"], cfg.dataset_name, "run")
 
    return dirs

# ---------------------------------------------------------------------------
# Weights & Biases
# ---------------------------------------------------------------------------



def setup_wandb(cfg) -> None:
    """
    Sets up wandb logging and saves the config to wandb.
    Make sure that the keys "wandb_dotenv_path" and "run_name" are contained in the config.

    Args:
        cfg (DictConfig): Hydra config.
    """
    load_dotenv(cfg.wandb_dotenv_path)
    wandb.init(config=vars(cfg), name=cfg.run_name, settings=wandb.Settings(code_dir="."))


def setup_pl_logger(cfg, model):
    """Return a WandbLogger (with model-weight watching) when cfg.wandb is set,
    or False to disable Lightning's built-in logger entirely."""
    if not cfg.wandb:
        return False
    setup_wandb(cfg)
    pl_logger = WandbLogger(log_model=True)
    pl_logger.watch(model, log="all", log_freq=20)
    return pl_logger


# ---------------- log -------------------------------------------

def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2

unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def log_prediction_grid_to_wandb(
    output,
    label_cmap,
    cluster_map_fn,
    step=None,
    epoch=None,
    ):
    required_keys = [
        "img",
        "label",
        "linear_preds",
        "cluster_preds",
        "fg_mask",
    ]

    missing_keys = [k for k in required_keys if k not in output]
    assert not missing_keys, f"Missing keys in output: {missing_keys}"

    n = len(output["img"]) 

    fig, ax = plt.subplots(5, n, figsize=(n * 3, 5 * 3))

    cmap = label_cmap

    for i in range(n):
        ax[0, i].imshow(prep_for_plot(output["img"][i]))
        ax[1, i].imshow(cmap[output["label"][i]])
        ax[2, i].imshow(cmap[output["linear_preds"][i]])
        ax[3, i].imshow(
            cmap[cluster_map_fn(output["cluster_preds"][i])]
        )
        ax[4, i].imshow(cmap[output["fg_mask"][i]])

    ax[0, 0].set_ylabel("Image", fontsize=16)
    ax[1, 0].set_ylabel("Label", fontsize=16)
    ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
    ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
    ax[4, 0].set_ylabel("Fg Mask", fontsize=16)

    remove_axes(ax)
    plt.tight_layout()

    log_dict = {
        "predictions": wandb.Image(fig),
    }

    if epoch is not None:
        log_dict["epoch"] = epoch
    if step is not None:
        log_dict["global_step"] = step

    wandb.log(log_dict)

    plt.close(fig)

 
# ---------------------------------------------------------------------------
# Post-training reporting
# ---------------------------------------------------------------------------
 
def log_best_metrics(best_metrics, std_logger):
    """Write each best metric to the standard logger after training completes."""
    for key, value in best_metrics.items():
        std_logger.info(f"Best {key}: {value:.2f}")
 

def save_results_to_csv(metrics, facet, layer, num_batch_pcainit, sample_interval, output_root, csv_filename):
    # Save results in a csv file.
    cluster_miou, linear_miou = metrics['test/cluster/mIoU'], metrics['test/linear/mIoU']
    new_row = [{
        'facet': facet,
        'layer': layer,
        'num_batch_pcainit':num_batch_pcainit,
        'sample_interval': sample_interval,
        'cluster_miou': f"{cluster_miou :.2f}",
        'linear_miou': f"{linear_miou :.2f}"
    }]
    df = pd.DataFrame(new_row, columns = new_row[0].keys())
    csv_path = os.path.join(output_root, "data", f'{csv_filename}.csv')
    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)
    return 



def save_run_results(best_metrics, cfg):
    """Persist best metrics to the shared CSV results file for the dataset."""
    save_results_to_csv(
        best_metrics,
        cfg.facet,
        cfg.layer,
        cfg.num_batch_pcainit,
        cfg.sample_interval,
        cfg.output_root,
        csv_filename=f"{cfg.dataset_name}_{cfg.model_size[0]}_{cfg.model_type}_{cfg.facet[0]}",
    )
 