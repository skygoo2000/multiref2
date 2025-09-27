#!/bin/bash
#SBATCH --job-name=img5b-4gpu
#SBATCH --nodes=1                  
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --account=liuyuan

export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/synworldimg53k/"
export DATASET_META_NAME="$DATASET_NAME/train.json"

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_TIMEOUT=3600
# export NCCL_IB_TIMEOUT=300
# export NCCL_IB_RETRY_CNT=10

LEARNING_RATE=2e-05
BATCH_SIZE=3
MAX_TRAIN_STEPS=10000
CHECKPOINTING_STEPS=200
RESUME_FROM_CHECKPOINT="latest"

MODEL_SUFFIX=$(basename "$MODEL_NAME" | sed 's/.*-//')
OUTPUT_DIR="ckpts/0925_${MODEL_SUFFIX}_img53k_lr${LEARNING_RATE}_ref-t0_beforeconcat_selfattn_neg-rope"

VALIDATION_STEPS=200
VALIDATION_PROMPTS="On a sunlit porch, the Pine-Sol Multi-Surface Cleaner sits atop an outdoor table surrounded by lush greenery. The camera angle is a close-up, focusing on the detailed textures and bright label of the bottle. The morning light is clean and crisp, highlighting the dew on nearby leaves. In the background, hints of a garden with colorful flowers can be seen, complemented by the soft chirping of birds, suggesting a tranquil, nature-infused environment."
VALIDATION_REF_PATH="$DATASET_NAME/ref/00000000.mp4"
VALIDATION_SIZE="512 512 1"  # height width frames


source /project/liuyuan/zekai/miniconda3/etc/profile.d/conda.sh
conda activate ref

cd /project/liuyuan/zekai/code/multiref2

## fsdp stage3
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_ref.py \

accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/wan2.2/train_ref.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=360 \
  --video_sample_stride=1 \
  --video_sample_n_frames=49 \
  --train_batch_size=$BATCH_SIZE \
  --video_repeat=1 \
  --dataloader_num_workers=8 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=5 \
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
  --train_mode="ti2v" \
  --trainable_modules "self_attn" \
  --report_model_info \
  --report_to="wandb" \
  --tracker_project_name="multiref-img" \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --gradient_checkpointing \
  # --low_vram \
  # --gradient_accumulation_steps=2 \
  # --enable_profiler


# sbatch --wait -o train_5b_img_8gpu.out scripts/wan2.2/sbatch_5b_img.sbatch