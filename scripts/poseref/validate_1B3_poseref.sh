export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/pose_vali"
export DATASET_META_NAME="$DATASET_NAME/oldcar_crop.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=7

CKPT_DIR="ckpts/1210_phantom1B3_pose1k5_lr1e-04_selfattn_refloss/checkpoint-5000"

python scripts/poseref/poseref_validation.py \
    --custom_transformer_path $CKPT_DIR/transformer \
    --validation_json $DATASET_META_NAME \
    --validation_samples 20 \
    --height 192 \
    --width 336 \
    --num_frames 49 \
    --num_ref_frames 8 \
    --fps 16 \
    --guidance_scale 6.0 \
    --num_inference_steps 50 \
    --sampler_name Flow_Unipc \
    --shift 5 \
    --save_comparison \
    --output_dir samples/multiref_pose_1B3/$CKPT_DIR/oldcar_crop \
    --gpu_memory_mode model_full_load \
    --weight_dtype bfloat16 \
    --seed 42
