
#TODO ask CLAUDE TO refactor the code
import argparse
import torch
import os
import torch.nn as nn

from src.ssl_rep_seg.extractor import ViTExtractor
from training.train_utils import (build_cluster_probes,build_unsupervised_metrics, 
                            collect_pca_features , compute_pca_cluster_init, 
                            extract_dense_features, generate_patch_res_labels, compute_fg_mask,
                            scatter_cluster_predictions, resolve_cluster_assignment, update_best_metrics,
                            build_checkpoint_name, build_checkpoint_and_assignments_fullpath, prune_old_assignments,
                            derive_runtime_params, configure_token_norm_patch)
from training.logger_utils import (setup_std_logger, setup_output_dirs, setup_pl_logger,log_prediction_grid_to_wandb,
                                    log_best_metrics, save_run_results)
from training.metrics import  UnsupervisedMetrics
from datasets.agumentation import get_class_labels, get_dataset_colormap
from datasets.dataloader import build_train_val_dataloaders
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
import torch.multiprocessing
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg, std_logger):
        super().__init__()

 
        class_labels = get_class_labels(cfg.dataset_name)
        n_classes = len(class_labels)
 
        self.cfg = cfg
        self.n_classes = n_classes
        self.target_n_classes = n_classes + cfg.extra_clusters if self.cfg.include_bg else n_classes + cfg.extra_clusters -1
        self.label_cmap = get_dataset_colormap(cfg.dataset_name, n_classes)
        self.logger_ = std_logger
        self.validation_outputs = []
        self.best_metrics = {
            "test/linear/mIoU": 0, "test/linear/Accuracy": 0,
            "test/cluster/mIoU": 0, "test/cluster/Accuracy": 0,
        }
 
        self.net = ViTExtractor(**vars(cfg))
        dim = self.net.model.feature_dim
 
        
        self.train_cluster_probe, self.cluster_probe = build_cluster_probes(dim, self.target_n_classes)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))
 
        (
            self.cluster_metrics,
            self.linear_metrics,
            self.train_cluster_metrics,
            self.train_linear_metrics,
        ) = build_unsupervised_metrics(cfg, class_labels, UnsupervisedMetrics)
 
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()
        self.automatic_optimization = False
        self.val_steps = 0
        self.save_hyperparameters()
 

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------
 
    def on_fit_start(self):
        if self.cfg.pca_initialization:
            self._pca_cluster_initialization()
            self.logger_.info('---- PCA initialization done ----')
        else:
            self.logger_.info('---- Skip PCA initialization ----')
 
        self.trainer.fit_loop.epoch_loop.val_loop.run()
        self.train()
 
    def _pca_cluster_initialization(self):
        self.logger_.info('-- PCA cluster initialization')
        self.eval()
 
        with torch.no_grad():
            dataloader = self.trainer.fit_loop._data_source.dataloader()
            features = collect_pca_features(self.net, dataloader, self.cfg, self.cfg.include_bg)
            features = features.to(self.device)
            self.logger_.info(f"collected features for PCA init with shape {features.shape}")
 
            # cluster_weights shape (target_n_classes, d)
            # linear_weights shape (n_classes, d, 1, 1)
            cluster_weights, linear_weights = compute_pca_cluster_init(
                features, self.target_n_classes, self.n_classes
            )
            self.train_cluster_probe.clusters.copy_(cluster_weights)
            self.cluster_probe.clusters.copy_(cluster_weights)
            self.linear_probe.weight.copy_(linear_weights)
 
        del features
        torch.cuda.empty_cache()
 
    def on_train_start(self):
        if self.cfg.wandb:
            self.logger.log_hyperparams(self.cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
 
    def training_step(self, batch, batch_idx):
        linear_probe_optim, cluster_probe_optim = self.optimizers()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()
 
        with torch.no_grad():
            img = batch["img"]
            label = batch["label"]
 
            feats = extract_dense_features(self.net, img, self.cfg)
            patch_res_label = generate_patch_res_labels(
                label, self.n_classes, self.cfg.patch_size)
 
            if not self.cfg.include_bg:
                fg_mask = compute_fg_mask(batch["mask"], self.cfg.patch_size).detach()
            else:
                fg_mask = torch.ones_like(patch_res_label, dtype=torch.bool)
 
        log_args = dict(sync_dist=False, rank_zero_only=True)
        loss = 0
        valid_mask = (patch_res_label >= 0) & (patch_res_label < self.n_classes)
 
        detached_code = torch.clone(feats.detach())
        linear_logits = self.linear_probe(detached_code).permute(0, 2, 3, 1)
 
        linear_loss = self.linear_probe_loss_fn(
            linear_logits[valid_mask], patch_res_label[valid_mask]).mean()
        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)
 
        fg_feats = detached_code.permute(0, 2, 3, 1)[fg_mask].detach()
        cluster_loss, cluster_logits = self.cluster_probe(fg_feats, None)
        select_cluster_preds = resolve_cluster_assignment(cluster_logits, self.cfg.include_bg)
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)
 
        self.train_linear_metrics.update(linear_logits[valid_mask].argmax(1), patch_res_label[valid_mask])
 
        cluster_preds = scatter_cluster_predictions(select_cluster_preds, fg_mask, patch_res_label)
        if not self.cfg.include_bg:
            self.train_cluster_metrics.update(select_cluster_preds, patch_res_label[fg_mask].reshape(-1))
        else:
            self.train_cluster_metrics.update(cluster_preds, patch_res_label)
 
        self.manual_backward(loss)
        cluster_probe_optim.step()
        linear_probe_optim.step()
 
        if self.global_step % self.cfg.train_metrics_log_interval == 0:
            self.log_dict({**self.train_linear_metrics.compute(), **self.train_cluster_metrics.compute()})
            self.train_linear_metrics.reset()
            self.train_cluster_metrics.reset()
 
        torch.cuda.empty_cache()
        return loss
    

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
 
    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
 
        with torch.no_grad():
            lowres_feats = extract_dense_features(self.net, img, self.cfg).contiguous() # B x C x h' x w'
            
            hres_feats = F.interpolate(
                lowres_feats, size=label.shape[-2:], mode="bilinear", align_corners=False
            )
          
            linear_preds = self.linear_probe(hres_feats).argmax(1)
            self.linear_metrics.update(linear_preds, label)
 
            if not self.cfg.include_bg:
                fg_mask = compute_fg_mask(batch["mask"], self.cfg.patch_size, return_hres=True).detach()
            else:
                fg_mask = torch.ones_like(label, dtype=torch.bool)
 
            fg_feats = hres_feats.permute(0, 2, 3, 1)[fg_mask]
            
            del hres_feats, lowres_feats
            torch.cuda.empty_cache()
 
            cluster_loss, cluster_logits = self.cluster_probe(fg_feats, None)
            select_cluster_preds = resolve_cluster_assignment(cluster_logits, self.cfg.include_bg)
            cluster_preds = scatter_cluster_predictions(select_cluster_preds, fg_mask, label)

            if not self.cfg.include_bg:
                self.cluster_metrics.update(select_cluster_preds, label.reshape(-1)[fg_mask.reshape(-1)])
            else:
                self.cluster_metrics.update(cluster_preds, label)
 
            if batch_idx == 0:
                self.validation_outputs = [{
                    'img': img[:self.cfg.n_images].detach().cpu(),
                    'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                    'cluster_preds': cluster_preds[:self.cfg.n_images].detach().cpu(),
                    'label': label[:self.cfg.n_images].detach().cpu(),
                    'fg_mask': fg_mask[:self.cfg.n_images].long().detach().cpu(),
                }]
          
 
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
 
        if not self.validation_outputs:
            return
 
        with torch.no_grad():
            tb_metrics = {**self.linear_metrics.compute(), **self.cluster_metrics.compute()}
 
            if self.trainer.is_global_zero and self.current_epoch % 10 == 0 and self.cfg.wandb:
                
                outputs = self.validation_outputs
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}
                log_prediction_grid_to_wandb(output,
                                        self.label_cmap,
                                        self.cluster_metrics.map_clusters,
                                        step=self.global_step)
 
            self.log_dict(tb_metrics)
            self.log("test_cluster_mIoU", tb_metrics["test/cluster/mIoU"], prog_bar=True)
 
            update_assignment = tb_metrics["test/cluster/mIoU"]>self.best_metrics["test/cluster/mIoU"] # a flag to decide if we need to update the assignment
            update_best_metrics(tb_metrics, self.best_metrics, self.logger_, self.current_epoch, self.global_step)
            self._save_checkpoint_if_needed(tb_metrics, update_assignment)
 
            self.linear_metrics.reset()
            self.cluster_metrics.reset()
 
            
    def _save_checkpoint_if_needed(self, tb_metrics, update_assignment):
        cluster_miou = tb_metrics["test/cluster/mIoU"]
 
        if self.trainer.global_step == 0:
            ckpt_name = build_checkpoint_name(self.cfg, cluster_miou, step=None)
            ckpt_path, assignment_path= build_checkpoint_and_assignments_fullpath(self.cfg, ckpt_name, self.cluster_metrics.assignments)
            # save both checkpoint and assignments for the first step
            self.trainer.save_checkpoint(ckpt_path)
            torch.save( self.cluster_metrics.assignments, assignment_path)

        elif update_assignment:
            prune_old_assignments(self.cfg, self.cfg.extra_clusters)
            ckpt_name = build_checkpoint_name(self.cfg, cluster_miou, step=self.trainer.global_step)
            ckpt_path, assignment_path= build_checkpoint_and_assignments_fullpath(self.cfg, ckpt_name, self.cluster_metrics.assignments)
            # save the assignment
            torch.save( self.cluster_metrics.assignments, assignment_path)
    
    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
 
    def configure_optimizers(self):
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=self.cfg.cluster_prob_lr)

        return linear_probe_optim, cluster_probe_optim


def build_checkpoint_filename(cfg):
    """Construct the ModelCheckpoint filename template.
    The _norm suffix is appended when patch tokens are L2-normalised."""
    suffix = "_norm" if cfg.norm_patchtoken else ""
    return (
        f"step{{step}}-ex_class{cfg.extra_clusters}"
        f"-clustermiou-{{test_cluster_mIoU:.2f}}{suffix}"
    )
 

def build_trainer(cfg, checkpoint_dir, pl_logger):
    """Construct and return the Lightning Trainer with a single best-model
    checkpoint callback."""
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, cfg.run_name),
        filename=build_checkpoint_filename(cfg),
        monitor="test/cluster/mIoU",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    return Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=pl_logger,
        max_steps=cfg.max_steps,
        val_check_interval=cfg.val_freq,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
    )

def my_app(cfg):

    # environment setup (logging, output dirs, random seed)
    std_logger = setup_std_logger(cfg.output_root) #setup standard logger 
    std_logger.info("%s", cfg)
    dirs = setup_output_dirs(cfg)
    seed_everything(seed=44)
    sys.stdout.flush()
 
    # prepare for training
    train_loader, val_loader = build_train_val_dataloaders(
        cfg.dataset_root, cfg.dataset_name, cfg.resize,
        cfg.batch_size, cfg.num_workers, cfg.scale_ratio,
        cfg.train_loader_crop_type, cfg.val_loader_crop_type,
        cfg.train_image_set, cfg.apply_geometric_aug, cfg.apply_photometric_aug,
    )
    model = LitUnsupervisedSegmenter(train_loader.dataset.n_classes, cfg, std_logger)
    pl_logger = setup_pl_logger(cfg, model)

    # train
    trainer = build_trainer(cfg, dirs["checkpoints"], pl_logger)
    trainer.fit(model, train_loader, val_loader)
    std_logger.info(f"{cfg.run_name} — training done.")
 

    # post-training reporting
    log_best_metrics(model.best_metrics, std_logger)
    save_run_results(model.best_metrics, cfg)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unsupervised Segmentation Training')

    # data
    parser.add_argument('--dataset_root', type=str, default='/home/xiaoyan/Documents/Data/',
                        help='path to dataset root')
    parser.add_argument('--dataset_name', type=str, default='VOC2012',
                        help='dataset name: [VOC2012 | cityscapes | cocostuff27 | pascalpart | cub_200]')
    parser.add_argument('--resize', default=(448, 448), nargs='+', type=int,
                        help='resize input images to this (H W) size')
    parser.add_argument('--val_loader_crop_type', type=str, default= None)
    parser.add_argument('--train_loader_crop_type', type=str, default= None)

    # model
    parser.add_argument('--model_type', type=str, default='dino', help='model type name')
    parser.add_argument('--model_size', default='base', type=str,
                        help='transformer model size: small | base | large')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size of the transformer')
    parser.add_argument('--stride', default=16, type=int,
                        help='stride of first conv layer; smaller -> higher resolution')
    parser.add_argument('--facet', default='key', type=str,
                        help="facet for descriptors: 'key' | 'query' | 'value' | 'token'")
    parser.add_argument('--layer', default=11, type=int, help='layer to create descriptors from')
    parser.add_argument('--norm_patchtoken', action='store_true', help='apply LayerNorm to patch tokens')
    
    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--include_bg', action='store_true',
                        help='include background features during clustering')
    parser.add_argument('--cluster_prob_lr', type=float, default=5e-3,
                        help='learning rate for the cluster probe')
    parser.add_argument('--max_steps', type=int, default=5000, help='maximum training steps')
    parser.add_argument('--val_freq', type=int, default=100,
                        help='validation frequency in training steps')
    parser.add_argument('--extra_clusters', type=int, default=0,
                        help='extra cluster slots on top of n_classes (cluster probe only)')
    parser.add_argument('--apply_geometric_aug', action='store_true')
    parser.add_argument('--apply_photometric_aug', action='store_true')
    parser.add_argument('--scale_ratio', type=float, default=0.8)

    # pca
    parser.add_argument('--pca_initialization', action='store_true',
                        help='initialise cluster centres via PCA')
    parser.add_argument('--num_batch_pcainit', type=int, default=96,
                        help='batches to sample for PCA initialisation')
    parser.add_argument('--sample_interval', type=int, default=10,
                        help='patch-sampling interval during PCA feature collection')

    # output
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--wandb', action='store_true', help='log run to Weights & Biases')
    parser.add_argument('--n_images', type=int, default=6,
                        help='images logged to wandb per validation pass')

    args = parser.parse_args()
    # automatically turn norm_patchtoken on for token facet in the last layer
    # optional, if remove this, please also modify the code in ssl_rep_seg/visualization/utils/checkpoint_utils.py 
    # accordingly to ensure the checkpoint loading is consistent with the training config
    args = configure_token_norm_patch(args) 

    cfg = derive_runtime_params(args)
    my_app(cfg)
