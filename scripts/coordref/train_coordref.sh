export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
export DATASET_NAME="datasets/Google"
export DATASET_META_NAME="$DATASET_NAME/train.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_VISIBLE_DEVICES=6,7

LEARNING_RATE=1e-04
BATCH_SIZE=4
MAX_TRAIN_STEPS=4000
CHECKPOINTING_STEPS=1000
RESUME_FROM_CHECKPOINT="latest"

OUTPUT_DIR="ckpts/1230_croodref1B3_Google14k_lr${LEARNING_RATE}_fullattn"

VALIDATION_STEPS=200
VALIDATION_PROMPTS="a small, chibi-style figurine with light purple hair in twin ponytails, elf ears, and a blue scarf."
VALIDATION_REF_PATH="samples/sam3/furiren/ref.mp4"
VALIDATION_REF_COORDMAP_PATH="samples/sam3/furiren/ref_coordmap.mp4"
VALIDATION_FG_COORDMAP_PATH="samples/sam3/furiren/fg_coordmap.mp4"
VALIDATION_GT_PATH="samples/sam3/furiren/furiren_gt.mp4"
VALIDATION_SIZE="192 336 49"  # height width frames

## fsdp stage3
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False sscripts/coordref/train_coordref.py \

## deepspeed zero2
# accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/coordref/train_coordref.py \

accelerate launch --mixed_precision="bf16" scripts/coordref/train_coordref.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=256 \
  --video_sample_stride=1 \
  --video_sample_n_frames=49 \
  --train_batch_size=$BATCH_SIZE \
  --video_repeat=0 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=24 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=3 \
  --validation_steps=$VALIDATION_STEPS \
  --validation_prompts "$VALIDATION_PROMPTS" \
  --validation_ref_path $VALIDATION_REF_PATH \
  --validation_ref_coordmap_path $VALIDATION_REF_COORDMAP_PATH \
  --validation_fg_coordmap_path $VALIDATION_FG_COORDMAP_PATH \
  --validation_gt_path $VALIDATION_GT_PATH \
  --validation_size $VALIDATION_SIZE \
  --learning_rate=$LEARNING_RATE \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=200 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --train_mode="control_ref" \
  --trainable_modules "." \
  --report_model_info \
  --report_to="wandb" \
  --tracker_project_name="fun_1B3-256p" \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --gradient_checkpointing \
  # --low_vram \
  # --gradient_accumulation_steps=4 \
  # --random_hw_adapt \