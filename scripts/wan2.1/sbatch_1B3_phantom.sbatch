#!/bin/bash
#SBATCH --job-name=1B3-384p         
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal
#SBATCH --account=liuyuan

export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets"
export DATASET_META_NAME="$DATASET_NAME/train_mix.json"

export NCCL_DEBUG=WARN
export PYTHONWARNINGS="ignore::FutureWarning"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_TIMEOUT=3600
# export NCCL_IB_TIMEOUT=300
# export NCCL_IB_RETRY_CNT=10

LEARNING_RATE=2e-05
BATCH_SIZE=3
MAX_TRAIN_STEPS=6000
CHECKPOINTING_STEPS=500
RESUME_FROM_CHECKPOINT="latest"

OUTPUT_DIR="ckpts/1006_phantom1B3_mix_lr${LEARNING_RATE}"

VALIDATION_STEPS=200
VALIDATION_PROMPTS="White pickup truck parked on a grassy area. The truck is a modern model with a large grille and black wheels. In the background, there is a red pickup truck parked next to the white truck. The scene appears to be set in a rural or semi-rural area, with a building and trees visible in the distance. The sky is partly cloudy, suggesting it might be a cool or overcast day."
VALIDATION_REF_PATH="datasets/synworld12/fg_video/H7z_-9IjXBA_85_23to151_fg.mp4"
VALIDATION_SIZE="384 672 121"  # height width frames

source /project/liuyuan/zekai/miniconda3/etc/profile.d/conda.sh
conda activate ref

cd /project/liuyuan/zekai/code/multiref2

## fsdp stage3
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.1/train_phantom.py \

## deepspeed zero2
# accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/wan2.1/train_phantom.py \

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/wan2.1/train_phantom.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --random_hw_adapt \
  --video_sample_stride=1 \
  --video_sample_n_frames=121 \
  --train_batch_size=$BATCH_SIZE \
  --video_repeat=0 \
  --dataloader_num_workers=8 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=3 \
  --validation_steps=$VALIDATION_STEPS \
  --validation_prompts "$VALIDATION_PROMPTS" \
  --validation_ref_path $VALIDATION_REF_PATH \
  --validation_size $VALIDATION_SIZE \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="normal" \
  --trainable_modules 'self_attn' \
  --report_model_info \
  --report_to="wandb" \
  --tracker_project_name="phantom_1B3-384p" \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --gradient_checkpointing \
  # --low_vram \
  # --gradient_accumulation_steps=4 \
  # --enable_profiler \
  # --random_hw_adapt \
  # --video_token_length=49 \
  # --training_with_video_token_length \