# !/bin/bash

# Loop through all values of args.layer from 6 to 11
# for layer in $(seq 6 8)
# do
#     echo "Running train.py with args.layer=$layer"
#     python -m training.train \
#         --num_workers 2 \
#         --include_bg \
#         --dataset_name cocostuff27 \
#         --cluster_prob_lr 5e-3 \
#         --batch_size 24 \
#         --pca_initialization \
#         --layer $layer  \
#         --model_type dinov3 \
#         --model_size base \
#         --patch_size 16 \
#         --stride 16 \
#         --facet key \
#         --num_batch_pcainit 96 \
#         --sample_interval 1 \
#         --output_root cocostuff27_results \
#         --train_loader_crop_type  center \
#         --val_loader_crop_type center \

# done


   
#####################  experiments cocostuff27 ######################

python -m training.train \
    --num_workers 16 \
    --include_bg \
    --dataset_root /home/xiaoyan/Documents/Data/ \
    --dataset_name cocostuff27 \
    --cluster_prob_lr 5e-3 \
    --batch_size 24 \
    --layer 11  \
    --pca_initialization \
    --model_type dinov3 \
    --model_size base \
    --patch_size 16 \
    --stride 16 \
    --facet key \
    --num_batch_pcainit 96 \
    --sample_interval 1\
    --max_steps 5000 \
    --val_freq 100  \
    --output_root cocostuff27_results \
    --train_loader_crop_type  center \
    --val_loader_crop_type center \



#####################  experiments cityscapes and pascalpart lr 2e-4 ######################

# python -m src.train \
#     --num_workers 16 \
#     --include_bg \
#     --dataset_root /home/xiaoyan/Documents/Data/ \
#     --dataset_name pascalpart \
#     --cluster_prob_lr 2e-4 \
#     --batch_size 24 \
#     --pca_initialization \
#     --layer 11  \
#     --model_type dino \
#     --model_size base \
#     --patch_size 16 \
#     --stride 16 \
#     --facet key \
#     --num_batch_pcainit 96 \
#     --sample_interval 1\
#     --max_steps 10000 \
#     --val_freq 40  \
#     --output_root pascalpart_results \
#     --train_loader_crop_type  center \
#     --val_loader_crop_type center \

    



# python -m src.train \
#     --num_workers 16 \
#     --include_bg \
#     --dataset_root /home/xiaoyan/Documents/Data/ \
#     --dataset_name cityscapes \
#     --cluster_prob_lr 2e-4 \
#     --batch_size 24 \
#     --pca_initialization \
#     --layer 11  \
#     --model_type dino \
#     --model_size base \
#     --patch_size 16 \
#     --stride 16 \
#     --facet key \
#     --num_batch_pcainit 96 \
#     --sample_interval 1\
#     --max_steps 10000 \
#     --val_freq 40  \
#     --output_root cityscapes_results \
#     --train_loader_crop_type  center \
#     --val_loader_crop_type center \
    
