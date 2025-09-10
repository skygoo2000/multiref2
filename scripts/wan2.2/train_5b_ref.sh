export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/synworld_11K/"
export DATASET_META_NAME="datasets/synworld_11K/train.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

LEARNING_RATE=2e-05
BATCH_SIZE=4
EPOCHS=1
CHECKPOINTING_STEPS=500
VALIDATION_STEPS=200
OUTPUT_DIR="ckpts/0911_5b_synworld11k_1ep"
RESUME_FROM_CHECKPOINT="latest"

accelerate launch --mixed_precision="bf16" scripts/wan2.2/train.py \
  --config_path="config/wan2.2/wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=256 \
  --token_sample_size=512 \
  --video_sample_stride=2 \
  --video_sample_n_frames=81 \
  --train_batch_size=$BATCH_SIZE \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=$EPOCHS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --validation_steps=$VALIDATION_STEPS \
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
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --gradient_checkpointing \
  --boundary_type="full" \
  --train_mode="ti2v" \
  --trainable_modules "." \
  --resume_from_checkpoint=$RESUME_FROM_CHECKPOINT