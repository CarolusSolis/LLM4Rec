#!/bin/sh

# Configuration
lambda_V=1
dataset=beauty_100_users_500_items
batch_size=4
grad_accum_steps=4
wandb_project="LLM4Rec"

# Training
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision fp16 \
    src/training.py \
    --dataset $dataset \
    --lambda_V $lambda_V \
    --batch_size $batch_size \
    --gradient_accumulation_steps $grad_accum_steps \
    --wandb_project $wandb_project \
    --wandb_name "${dataset}_train"

# Finetuning
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision fp16 \
    src/finetuning.py \
    --dataset $dataset \
    --lambda_V $lambda_V \
    --batch_size $batch_size \
    --gradient_accumulation_steps $grad_accum_steps \
    --wandb_project $wandb_project \
    --wandb_name "${dataset}_finetune"

# Prediction
python src/predict.py --dataset $dataset --lambda_V $lambda_V