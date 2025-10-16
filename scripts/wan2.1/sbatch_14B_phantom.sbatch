#!/bin/bash
#SBATCH --job-name=1B3-256p         
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=preempt
#SBATCH --account=liuyuan

export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-14B"
export DATASET_NAME="datasets"
export DATASET_META_NAME="$DATASET_NAME/train_vid.json"

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
BATCH_SIZE=2
MAX_TRAIN_STEPS=5000
CHECKPOINTING_STEPS=500
RESUME_FROM_CHECKPOINT="models/Personalized_Model/phantom-14B"

OUTPUT_DIR="ckpts/1015_phantom14B_vid15k_lr${LEARNING_RATE}_t0_gap1_selfattn_fromscratch"

VALIDATION_STEPS=200
VALIDATION_PROMPTS="this object is on the table in the kitchen"
VALIDATION_REF_PATH="asset/ref_1.png"
VALIDATION_SIZE="256 448 49"  # height width frames

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
  --video_sample_size=256 \
  --random_hw_adapt \
  --video_sample_stride=1 \
  --video_sample_n_frames=49 \
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
  --tracker_project_name="phantom_14B-256p" \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --gradient_checkpointing \
  # --low_vram \
  # --gradient_accumulation_steps=4 \
  # --enable_profiler \
  # --random_hw_adapt \
  # --video_token_length=49 \
  # --training_with_video_token_length \