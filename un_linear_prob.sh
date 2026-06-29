#!/bin/bash

# -----------------cocostuff27-------------------------------------

# for layer in {11..23}
# do
#     sbatch train_slurm.sh -m training.train \
#     --num_workers 8 \
#     --dataset_root /fast/AG_Kainmueller/data/ \
#     --include_bg \
#     --dataset_name cocostuff27 \
#     --cluster_prob_lr 5e-3 \
#     --batch_size 24 \
#     --pca_initialization \
#     --layer $layer \
#     --model_type dinov3 \
#     --model_size large \
#     --patch_size 16\
#     --stride 16 \
#     --facet token \
#     --num_batch_pcainit 96 \
#     --sample_interval 1 \
#     --max_steps 5000 \
#     --val_freq 100 \
#     --train_loader_crop_type  center \
#     --val_loader_crop_type center \
#     --output_root xiaoyan_cc_s48

# done


# -----------------pascalpart-------------------------------------

sbatch train_slurm.sh -m training.train \
    --num_workers 8 \
    --dataset_root /fast/AG_Kainmueller/data/ \
    --include_bg \
    --dataset_name cityscapes \
    --cluster_prob_lr 2e-4 \
    --batch_size 24 \
    --pca_initialization \
    --layer 10 \
    --model_type mae \
    --model_size base \
    --patch_size 16\
    --stride 16 \
    --facet key \
    --num_batch_pcainit 96 \
    --sample_interval 1 \
    --max_steps 10000 \
    --val_freq 40 \
    --train_loader_crop_type  center \
    --val_loader_crop_type center \
    --output_root cityscapes_cc_s44_re

   




 

# for layer in {0..11}
# do
#     sbatch train_slurm.sh -m src.train \
#         --num_workers 8 \
#         --dataset_root /fast/AG_Kainmueller/data/ \
#         --include_bg \
#         --dataset_name pascalpart\
#         --cluster_prob_lr 2e-4 \
#         --batch_size 24 \
#         --pca_initialization \
#         --layer $layer  \
#         --model_type dinov3  \
#         --model_size base \
#         --patch_size 16\
#         --stride 16 \
#         --facet token\
#         --num_batch_pcainit 96 \
#         --sample_interval 4 \
#         --val_freq 40 \
#         --max_steps  5000 \
#         --output_root pascalpart_output_pca96-2 
# done


############# test different batch size influence on cocostuff27
# for layer in {12..23}
# do
#     sbatch train_slurm.sh -m src.train \
#         --num_workers 8 \
#         --dataset_root /fast/AG_Kainmueller/data/ \
#         --include_bg \
#         --dataset_name cocostuff27 \
#         --cluster_prob_lr 5e-3 \
#         --batch_size 24 \
#         --pca_initialization \
#         --layer $layer  \
#         --model_type dinov3  \
#         --model_size large \
#         --patch_size 16\
#         --stride 16 \
#         --facet token \
#         --num_batch_pcainit 96 \
#         --sample_interval 1 \
#         --val_freq 100 \
#         --max_steps 10000 \
#         --output_root train_out_cc_bs24_rebuttal 
# done

############# learning rate for different dataset
# for lr in 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 1e-2
# do
#     sbatch train_slurm.sh -m src.train \
#             --num_workers 8 \
#             --dataset_root /fast/AG_Kainmueller/data/ \
#             --include_bg \
#             --dataset_name cityscapes \
#             --cluster_prob_lr $lr \
#             --batch_size 24 \
#             --pca_initialization \
#             --layer 22 \
#             --model_type dinov3 \
#             --model_size large \
#             --patch_size 16\
#             --stride 16 \
#             --facet token\
#             --num_batch_pcainit 96 \
#             --sample_interval 1 \
#             --val_freq 40 \
#             --max_steps 10000 \
#             --apply_geometric_aug \
#             --output_root cityscapes_lr_test \
#             --wandb 
# done



# for layer in {0..23}
# do
#     sbatch train_slurm.sh -m src.train \
#             --num_workers 8 \
#             --dataset_root /fast/AG_Kainmueller/data/ \
#             --include_bg \
#             --dataset_name cityscapes \
#             --cluster_prob_lr 2e-4 \
#             --batch_size 24 \
#             --pca_initialization \
#             --layer  $layer  \
#             --model_type dinov3 \
#             --model_size large \
#             --patch_size 16\
#             --stride 16 \
#             --facet token \
#             --num_batch_pcainit 96 \
#             --sample_interval 1 \
#             --val_freq 40 \
#             --max_steps 10000 \
#             --scale_ratio 0.99 \
#             --output_root cityscapes_output_sr1
# done
