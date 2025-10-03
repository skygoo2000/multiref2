export MODEL_NAME="models/Diffusion_Transformer/Wan2.2-TI2V-5B"
export DATASET_NAME="datasets/synworld12val"
export DATASET_META_NAME="$DATASET_NAME/train.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

CKPT_DIR="ckpts/0928_5B_overfit_lr2e-05_ref-noisy_beforeconcat_selfattn_neg-rope_256p_nodrop_cfg/checkpoint-10000/"

python scripts/wan2.2/ref_validation.py \
    --custom_transformer_path $CKPT_DIR/transformer \
    --validation_json $DATASET_META_NAME \
    --validation_samples 20 \
    --height 256 \
    --width 448 \
    --num_frames 49 \
    --guide_scale_text 5.0 \
    --guide_scale_ref 5.0 \
    --num_inference_steps 50 \
    --save_comparison \
    --output_dir samples/noisy_val