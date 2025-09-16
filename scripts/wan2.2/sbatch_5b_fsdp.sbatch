#!/bin/bash
#SBATCH --job-name=train5b         
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal
#SBATCH --account=liuyuan

export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/synworld10/"
export DATASET_META_NAME="datasets/synworld10/train.json"

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_TIMEOUT=3600
# export NCCL_IB_TIMEOUT=300
# export NCCL_IB_RETRY_CNT=10

LEARNING_RATE=1e-04
BATCH_SIZE=1
EPOCHS=2000
CHECKPOINTING_STEPS=200
RESUME_FROM_CHECKPOINT="latest"

MODEL_SUFFIX=$(basename "$MODEL_NAME" | sed 's/.*-//')
DATASET_SUFFIX=$(echo "$DATASET_META_NAME" | sed 's|.*/\([^/]*\)/[^/]*$|\1|')
OUTPUT_DIR="ckpts/$(date +%m%d)_${MODEL_SUFFIX}_${DATASET_SUFFIX}_${EPOCHS}ep_ref-t0_afterconcat"

VALIDATION_STEPS=200
VALIDATION_EPOCHS=99999
VALIDATION_PROMPTS="White pickup truck parked on a grassy area. The truck is a modern model with a large grille and black wheels. In the background, there is a red pickup truck parked next to the white truck. The scene appears to be set in a rural or semi-rural area, with a building and trees visible in the distance. The sky is partly cloudy, suggesting it might be a cool or overcast day."
VALIDATION_REF_PATH="datasets/synworld10/fg_video/H7z_-9IjXBA_85_23to151_fg.mp4"
VALIDATION_SIZE="480 832 121"  # height width frames

module load slurm "nvhpc-hpcx-cuda12/23.11"

source /project/liuyuan/zekai/miniconda3/etc/profile.d/conda.sh
conda activate sam

cd /project/liuyuan/zekai/code/multiref2

## normal
# accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_ref.py \

## fsdp stage2
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "SHARD_GRAD_OP" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_ref.py \

## fsdp stage3
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_ref.py \

accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_ref.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=121 \
  --train_batch_size=$BATCH_SIZE \
  --video_repeat=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=$EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=5 \
  --validation_steps=$VALIDATION_STEPS \
  --validation_epochs=$VALIDATION_EPOCHS \
  --validation_prompts "$VALIDATION_PROMPTS" \
  --validation_ref_path $VALIDATION_REF_PATH \
  --validation_size $VALIDATION_SIZE \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type="full" \
  --train_mode="ti2v" \
  --trainable_modules "." \
  --report_model_info \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  # --gradient_checkpointing \
  # --low_vram \
  # --gradient_accumulation_steps=4 \
  # --enable_profiler


# sbatch --wait -o train_5b.out scripts/wan2.2/sbatch_5b_ref.sbatch