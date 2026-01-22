export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-Fun-5B-Control"
export DATASET_NAME="datasets/coordref_vali"
export DATASET_META_NAME="$DATASET_NAME/coordref_w.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=1

CKPT_DIR="ckpts/0122_croodref5B_vid41k_lr2e-05_fullattn_2fullref_negrope_480p_sft/checkpoint-1000"

python scripts/coordref5b/coordref_5b_validation.py \
    --config_path config/wan2.2/wan_civitai_5b.yaml \
    --model_name $MODEL_NAME \
    --custom_transformer_path $CKPT_DIR/transformer \
    --validation_json $DATASET_META_NAME \
    --validation_samples 30 \
    --height 480 \
    --width 832 \
    --num_frames 121 \
    --num_ref_frames 8 \
    --fps 24 \
    --num_inference_steps 50 \
    --sampler_name Flow_Unipc \
    --shift 5 \
    --save_comparison \
    --output_dir samples/coordref_5B/no-appearance-cfg/$CKPT_DIR/mix94k \
    --gpu_memory_mode model_full_load \
    --weight_dtype bfloat16 \
    --seed 42 \
    # --validation_mode
