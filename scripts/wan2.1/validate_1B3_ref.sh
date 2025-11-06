export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-T2V-1.3B"
export DATASET_NAME="datasets/TexVerse_vali"
export DATASET_META_NAME="$DATASET_NAME/train.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="ckpts/1031_phantom1B3_vace10k_lr1e-04_fullattn/checkpoint-10000"

python scripts/wan2.1/phantom_ref_validation.py \
    --custom_transformer_path $CKPT_DIR/transformer \
    --validation_json $DATASET_META_NAME \
    --validation_samples 20 \
    --height 256 \
    --width 448 \
    --num_frames 49 \
    --fps 16 \
    --guidance_scale 6.0 \
    --num_inference_steps 50 \
    --sampler_name Flow_Unipc \
    --shift 5 \
    --save_comparison \
    --output_dir samples/phantom1.3b/vace10k_lr1e-04_fullattn_2 \
    --gpu_memory_mode model_full_load \
    --weight_dtype bfloat16 \
    --seed 42

