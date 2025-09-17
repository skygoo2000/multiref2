export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/synworld12/"
export DATASET_META_NAME="$DATASET_NAME/train.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

LEARNING_RATE=2e-05
BATCH_SIZE=1
MAX_TRAIN_STEPS=1200
CHECKPOINTING_STEPS=200
RESUME_FROM_CHECKPOINT="latest"

MODEL_SUFFIX=$(basename "$MODEL_NAME" | sed 's/.*-//')
OUTPUT_DIR="ckpts/$(date +%m%d)_${MODEL_SUFFIX}_overfit_${MAX_TRAIN_STEPS}steps_lr${LEARNING_RATE}_ref-t0_afterconcat"

VALIDATION_STEPS=200
VALIDATION_PROMPTS="White pickup truck parked on a grassy area. The truck is a modern model with a large grille and black wheels. In the background, there is a red pickup truck parked next to the white truck. The scene appears to be set in a rural or semi-rural area, with a building and trees visible in the distance. The sky is partly cloudy, suggesting it might be a cool or overcast day."
VALIDATION_REF_PATH="$DATASET_NAME/fg_video/H7z_-9IjXBA_85_23to151_fg.mp4"
VALIDATION_SIZE="480 832 121"  # height width frames

## normal
# accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_ref.py \

## fsdp stage2
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "SHARD_GRAD_OP" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_ref.py \

## fsdp stage3
# accelerate launch --mixed_precision="bf16" --use_fsdp --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock --fsdp_sharding_strategy "FULL_SHARD" --fsdp_state_dict_type=SHARDED_STATE_DICT --fsdp_backward_prefetch "BACKWARD_PRE" --fsdp_cpu_ram_efficient_loading False scripts/wan2.2/train_ref.py \

## deepspeed zero2
# accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json scripts/wan2.2/train_ref.py \

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train_ref.py \
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
  --dataloader_num_workers=8 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --checkpoints_total_limit=10 \
  --validation_steps=$VALIDATION_STEPS \
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
  --report_to="wandb" \
  --tracker_project_name="multiref-ti2v-5b" \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT \
  --gradient_checkpointing \
  --low_vram \
  # --gradient_accumulation_steps=2 \
  # --enable_profiler