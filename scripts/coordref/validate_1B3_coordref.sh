export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"
export DATASET_NAME="datasets/coordref_vali"
export DATASET_META_NAME="$DATASET_NAME/coordref_w.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

CKPT_DIR="ckpts/0115_croodref1B3_mix94k_lr2e-05_fullattn_2fullref_negrope_480p"

python scripts/coordref/coordref_validation.py \
    --model_name $MODEL_NAME \
    --custom_transformer_path $CKPT_DIR \
    --validation_json $DATASET_META_NAME \
    --validation_samples 30 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --num_ref_frames 8 \
    --fps 16 \
    --num_inference_steps 30 \
    --sampler_name Flow_Unipc \
    --shift 5 \
    --save_comparison \
    --output_dir samples/coordref_1B3/no-appearance-cfg/$CKPT_DIR/mix94k \
    --gpu_memory_mode model_cpu_offload \
    --weight_dtype bfloat16 \
    --seed 42 \
    --validation_mode