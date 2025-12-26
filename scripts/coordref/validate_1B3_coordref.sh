export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
export DATASET_NAME="samples/sam3/furiren"
export DATASET_META_NAME="$DATASET_NAME/coordref.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0

CKPT_DIR="ckpts/1223_croodref1B3_Google7k_lr1e-04_fullattn/checkpoint-10000"

python scripts/coordref/coordref_validation.py \
    --model_name $MODEL_NAME \
    --custom_transformer_path $CKPT_DIR/transformer \
    --validation_json $DATASET_META_NAME \
    --validation_samples 10 \
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
    --output_dir samples/coordref_1B3/$CKPT_DIR/furiren \
    --gpu_memory_mode model_full_load \
    --weight_dtype bfloat16 \
    --seed 42

