export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/pose1k5"
export DATASET_META_NAME="$DATASET_NAME/train.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONWARNINGS="ignore::FutureWarning"
export CUDA_VISIBLE_DEVICES=1,2,3

LEARNING_RATE=1e-04
BATCH_SIZE=10
MAX_TRAIN_STEPS=5000
CHECKPOINTING_STEPS=1000
RESUME_FROM_CHECKPOINT="latest"

OUTPUT_DIR="ckpts/1210_phantom1B3_pose1k5_lr${LEARNING_RATE}_selfattn_refloss"

VALIDATION_STEPS=200
VALIDATION_PROMPTS="A vintage luxury car with an elongated body and classic design."
VALIDATION_REF_PATH="datasets/pose_vali/cropped_ref/1043f7cdbe1146df828a047dcbf42cc2.mp4"
VALIDATION_MASK_PATH="datasets/pose50/mask/1043f7cdbe1146df828a047dcbf42cc2.mp4"
VALIDATION_FG_PATH="datasets/pose50/video/1043f7cdbe1146df828a047dcbf42cc2.mp4"
VALIDATION_POSE_PATH="datasets/pose50/pose/1043f7cdbe1146df828a047dcbf42cc2.json"
VALIDATION_SIZE="192 336 49"  # height width frames

## fsdp stage3
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/poseref/train_poseref.py \

## deepspeed zero2
# accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/poseref/train_poseref.py \

accelerate launch scripts/poseref/train_poseref.py \
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
  --dataloader_num_workers=24 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=3 \
  --validation_steps=$VALIDATION_STEPS \
  --validation_prompts "$VALIDATION_PROMPTS" \
  --validation_ref_path $VALIDATION_REF_PATH \
  --validation_mask_path $VALIDATION_MASK_PATH \
  --validation_fg_path $VALIDATION_FG_PATH \
  --validation_pose_path $VALIDATION_POSE_PATH \
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
  --boundary_type="full" \
  --train_mode="normal" \
  --trainable_modules 'self_attn' 'pos_emb' 'projector' \
  --report_model_info \
  --report_to="wandb" \
  --tracker_project_name="poseref_1B3-256p" \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --gradient_checkpointing \
  # --low_vram \
  # --gradient_accumulation_steps=4 \
  # --enable_profiler \
  # --random_hw_adapt \
  # --video_token_length=49 \
  # --training_with_video_token_length \