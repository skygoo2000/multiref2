import argparse
import glob
import json
import os
import sys
import time
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from decord import VideoReader
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (
    AutoencoderKLWan,
    WanT5EncoderModel,
    AutoTokenizer,
    CLIPModel
)
from videox_fun.models.multiref_transformer3d import CroodRefTransformer3DModel
from videox_fun.pipeline import WanFunCroodRefPipeline
from videox_fun.utils.utils import filter_kwargs, save_videos_grid, get_image_latent, padding_image
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for CroodRef model")
    
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Path to config file")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control", help="Base model path")
    parser.add_argument("--custom_transformer_path", type=str, default=None, help="Custom transformer checkpoint path")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE checkpoint path")
    
    parser.add_argument("--validation_json", type=str, required=True, help="Path to validation json")
    parser.add_argument("--validation_samples", type=int, default=10, help="Number of samples to validate")
    
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=448, help="Video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--num_ref_frames", type=int, default=8, help="Number of reference frames to sample from video")
    parser.add_argument("--fps", type=int, default=24, help="FPS for output video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Text guidance scale")
    parser.add_argument("--shift", type=int, default=5, help="Shift parameter for scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--output_dir", type=str, default="outputs/croodref_1B3", help="Output directory")
    parser.add_argument("--save_comparison", action="store_true", help="Save comparison videos (ref + generated)")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Weight dtype")
    
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    
    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load", 
                        choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"],
                        help="GPU memory mode")
    
    parser.add_argument("--validation_mode", action="store_true", 
                        help="Enable validation mode: keep models in GPU and auto-detect new checkpoint-validation checkpoints")
    parser.add_argument("--check_interval", type=int, default=60,
                        help="Interval in seconds to check for new validation checkpoints")
    
    return parser.parse_args()


def load_reference_video_full(ref_path, val_height, val_width):
    """
    Load full reference video/images without sampling
    Returns: torch.Tensor [1, C, F, H, W] in range [0, 1] or None
    """
    if os.path.isdir(ref_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(ref_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(ref_path, f'*{ext.upper()}')))
        
        if len(image_files) == 0:
            print(f"Warning: No image files found in directory: {ref_path}")
            return None
        
        print(f"Found {len(image_files)} images in directory")
        ref_list = []
        sample_size = [val_height, val_width]
        for img_path in sorted(image_files):
            frame_latent = get_image_latent(ref_image=img_path, sample_size=sample_size, padding=False)
            ref_list.append(frame_latent)
        
        validation_ref = torch.cat(ref_list, dim=2)
        print(f"Loaded reference from directory with all {len(image_files)} frames, shape: {validation_ref.shape}, range: [0, 1]")
        return validation_ref
    
    ref_ext = ref_path.lower().split('.')[-1]
    if ref_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
        sample_size = [val_height, val_width]
        validation_ref = get_image_latent(ref_image=ref_path, sample_size=sample_size, padding=False)
        print(f"Loaded reference image with shape: {validation_ref.shape}, range: [0, 1]")
        return validation_ref
    
    try:
        vr = VideoReader(ref_path)
        total_frames = len(vr)
        
        ref_list = []
        sample_size = [val_height, val_width]
        for idx in range(total_frames):
            frame = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame)
            
            frame_pil = padding_image(frame_pil, sample_size[1], sample_size[0])
            frame_pil = frame_pil.resize((sample_size[1], sample_size[0]))
            
            frame_tensor = torch.from_numpy(np.array(frame_pil))
            frame_tensor = frame_tensor.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255.0  # [0, 1]
            frame_tensor = frame_tensor.squeeze(2)
            ref_list.append(frame_tensor)
        
        validation_ref = torch.cat([f.unsqueeze(2) for f in ref_list], dim=2)
        print(f"Loaded reference video with all {total_frames} frames, shape: {validation_ref.shape}, range: [0, 1]")
        return validation_ref
        
    except Exception as e:
        print(f"Failed to load reference video: {e}")
        return None


def sample_reference_frames(ref_full, num_ref_frames):
    """
    Sample specified number of frames from full reference video
    Using same logic as train_croodref.py sample_ref_frames
    Args:
        ref_full: [1, C, F, H, W] full reference video
        num_ref_frames: number of frames to sample
    Returns:
        sampled_ref: torch.Tensor [1, C, num_ref_frames, H, W]
        ref_batch_index: List[int] sampled frame indices
    """
    total_frames = ref_full.shape[2]
    
    if total_frames <= num_ref_frames:
        ref_batch_index = list(range(total_frames))
    else:
        ref_batch_index = [0]
        
        if num_ref_frames > 1:
            segment_boundaries = np.linspace(0, total_frames - 1, num_ref_frames, dtype=int)
            for i in range(1, num_ref_frames):
                segment_start = segment_boundaries[i - 1]
                segment_end = segment_boundaries[i]
                if segment_start >= segment_end:
                    segment_end = min(segment_start + 1, total_frames - 1)
                random_idx = np.random.randint(max(segment_start + 1, 1), segment_end + 1)
                ref_batch_index.append(int(random_idx))
    
    sampled_ref = ref_full[:, :, ref_batch_index, :, :]
    print(f"Sampled {len(ref_batch_index)} frames from {total_frames} total frames, indices: {ref_batch_index}")
    return sampled_ref, ref_batch_index


def load_validation_video_full(video_path, val_height, val_width):
    """
    Load full video without sampling
    Returns: torch.Tensor [1, F, C, H, W] in range [0, 1] or None
    """
    try:
        vr = VideoReader(video_path)
        total_frames = len(vr)
        
        frames = []
        for idx in range(total_frames):
            frame = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((val_width, val_height))
            frames.append(np.array(frame_pil))
        
        validation_video = np.stack(frames, axis=0)
        validation_video = torch.from_numpy(validation_video).permute(0, 3, 1, 2).float() / 255.0  # [0, 1]
        validation_video = validation_video.unsqueeze(0)
        print(f"Loaded full validation video with shape: {validation_video.shape}, range: [0, 1]")
        return validation_video
    except Exception as e:
        print(f"Failed to load validation video: {e}")
        return None


def load_first_frame_image(image_path, val_height, val_width):
    """
    Load a single image as first frame
    Returns: torch.Tensor [1, C, 1, H, W] in range [0, 1] or None
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((val_width, val_height))
        img_array = np.array(img)
        
        # Convert to tensor [C, H, W] in range [0, 1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        # Add batch and frame dimensions: [1, C, 1, H, W]
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(2)
        print(f"Loaded first frame image with shape: {img_tensor.shape}, range: [0, 1]")
        return img_tensor
    except Exception as e:
        print(f"Failed to load first frame image: {e}")
        return None


def find_validation_checkpoint(training_output_dir):
    """
    Find checkpoint-validation-{steps} folder in training output directory
    Returns: (checkpoint_path, step_number) or (None, None)
    """
    if not os.path.exists(training_output_dir):
        return None, None
    
    checkpoints = [d for d in os.listdir(training_output_dir) if d.startswith("checkpoint-validation-")]
    if not checkpoints:
        return None, None
    
    # Should only be one, but take the newest if multiple exist
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    checkpoint_name = checkpoints[0]
    step_number = int(checkpoint_name.split("-")[-1])
    checkpoint_path = os.path.join(training_output_dir, checkpoint_name)
    
    return checkpoint_path, step_number


def load_transformer_from_checkpoint(checkpoint_path, base_transformer, config, device, weight_dtype):
    """
    Load transformer weights from a checkpoint directory (handles DeepSpeed Zero3 and regular formats)
    """
    print(f"Loading transformer from checkpoint: {checkpoint_path}")
    
    from safetensors.torch import load_file
    import subprocess
    
    # Check if this is a DeepSpeed Zero3 checkpoint (has pytorch_model dir with zero_pp_rank files)
    pytorch_model_dir = os.path.join(checkpoint_path, "pytorch_model")
    
    if os.path.exists(pytorch_model_dir):
        # Check for Zero3 checkpoint format
        zero_files = [f for f in os.listdir(pytorch_model_dir) if f.startswith("zero_pp_rank_") and f.endswith("_model_states.pt")]
        
        if zero_files:
            print(f"Detected DeepSpeed Zero3 checkpoint format with {len(zero_files)} shards")
            
            # Check if already converted to transformer folder
            transformer_output_dir = os.path.join(checkpoint_path, "transformer")
            converted_model_path = os.path.join(transformer_output_dir, "pytorch_model.bin")
            converted_index_path = os.path.join(transformer_output_dir, "pytorch_model.bin.index.json")
            
            # Check if conversion already exists
            if os.path.isfile(converted_model_path) or os.path.isfile(converted_index_path):
                print(f"Found existing converted model in transformer folder, loading directly...")
                
                if os.path.isfile(converted_model_path):
                    # Single file
                    print(f"Loading from: {converted_model_path}")
                    state_dict = torch.load(converted_model_path, map_location="cpu")
                else:
                    # Sharded files
                    print(f"Loading sharded checkpoint from: {converted_index_path}")
                    with open(converted_index_path, 'r') as f:
                        index = json.load(f)
                    
                    state_dict = {}
                    weight_map = index["weight_map"]
                    shard_files = set(weight_map.values())
                    
                    for shard_file in tqdm(shard_files, desc="Loading shards"):
                        shard_path = os.path.join(transformer_output_dir, shard_file)
                        shard_state_dict = torch.load(shard_path, map_location="cpu")
                        state_dict.update(shard_state_dict)
            else:
                # Need to convert
                print(f"No existing conversion found, converting Zero3 checkpoint...")
                
                # Check if zero_to_fp32.py exists
                zero_to_fp32_script = os.path.join(checkpoint_path, "zero_to_fp32.py")
                if not os.path.exists(zero_to_fp32_script):
                    raise FileNotFoundError(f"zero_to_fp32.py not found in checkpoint: {checkpoint_path}")
                
                # Create transformer directory for output
                os.makedirs(transformer_output_dir, exist_ok=True)
                
                try:
                    print(f"Converting Zero3 checkpoint using zero_to_fp32.py...")
                    print(f"This may take a few minutes...")
                    print(f"Output will be saved to: {transformer_output_dir}")
                    
                    # Run zero_to_fp32.py script to merge sharded weights
                    # The script signature: zero_to_fp32.py checkpoint_dir output_dir [--tag TAG]
                    cmd = [
                        "python",
                        zero_to_fp32_script,
                        checkpoint_path,
                        transformer_output_dir,
                        "--tag", "pytorch_model"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    
                    if result.returncode != 0:
                        print(f"Error running zero_to_fp32.py:")
                        print(result.stderr)
                        raise RuntimeError(f"Failed to convert Zero3 checkpoint: {result.stderr}")
                    
                    print(f"Successfully converted Zero3 checkpoint to: {transformer_output_dir}")
                    
                    # Load the converted model
                    if os.path.isfile(converted_model_path):
                        # Single file
                        print(f"Loading converted model from: {converted_model_path}")
                        state_dict = torch.load(converted_model_path, map_location="cpu")
                    elif os.path.isfile(converted_index_path):
                        # Sharded files
                        print(f"Loading sharded converted model...")
                        with open(converted_index_path, 'r') as f:
                            index = json.load(f)
                        
                        state_dict = {}
                        weight_map = index["weight_map"]
                        shard_files = set(weight_map.values())
                        
                        for shard_file in tqdm(shard_files, desc="Loading shards"):
                            shard_path = os.path.join(transformer_output_dir, shard_file)
                            shard_state_dict = torch.load(shard_path, map_location="cpu")
                            state_dict.update(shard_state_dict)
                    else:
                        raise FileNotFoundError(f"Conversion completed but no model file found in {transformer_output_dir}")
                    
                except Exception as e:
                    # Clean up transformer folder on error to allow retry
                    if os.path.exists(transformer_output_dir):
                        print(f"Cleaning up failed conversion in {transformer_output_dir}")
                        shutil.rmtree(transformer_output_dir, ignore_errors=True)
                    raise RuntimeError(f"Failed to load Zero3 checkpoint: {e}")
        else:
            # Regular checkpoint in pytorch_model dir
            print(f"Regular checkpoint format in pytorch_model directory")
            pt_file = os.path.join(pytorch_model_dir, "pytorch_model.bin")
            if os.path.exists(pt_file):
                state_dict = torch.load(pt_file, map_location="cpu")
            else:
                raise FileNotFoundError(f"No model file found in {pytorch_model_dir}")
    else:
        # Try loading from transformer subdirectory (non-Zero3 format)
        transformer_dir = os.path.join(checkpoint_path, "transformer")
        
        if not os.path.exists(transformer_dir):
            raise FileNotFoundError(f"Neither pytorch_model nor transformer directory found in checkpoint: {checkpoint_path}")
        
        # Check for sharded safetensors (FSDP format)
        index_files = [f for f in os.listdir(transformer_dir) if f.endswith('.safetensors.index.json')]
        
        if index_files:
            # Load from sharded format
            index_file = os.path.join(transformer_dir, index_files[0])
            print(f"Loading sharded checkpoint from index: {index_file}")
            
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            state_dict = {}
            weight_map = index["weight_map"]
            shard_files = set(weight_map.values())
            
            for shard_file in tqdm(shard_files, desc="Loading shards"):
                shard_path = os.path.join(transformer_dir, shard_file)
                if os.path.exists(shard_path):
                    shard_state_dict = load_file(shard_path)
                    state_dict.update(shard_state_dict)
        else:
            # Try to load from regular safetensors or pytorch files
            safetensors_files = [f for f in os.listdir(transformer_dir) if f.endswith('.safetensors') and not f.endswith('.index.json')]
            
            if safetensors_files:
                print(f"Loading from safetensors files: {safetensors_files}")
                state_dict = {}
                for safetensors_file in safetensors_files:
                    safetensors_path = os.path.join(transformer_dir, safetensors_file)
                    shard_state_dict = load_file(safetensors_path)
                    state_dict.update(shard_state_dict)
            else:
                # Try pytorch_model.bin
                pt_file = os.path.join(transformer_dir, "pytorch_model.bin")
                if os.path.exists(pt_file):
                    print(f"Loading from pytorch file: {pt_file}")
                    state_dict = torch.load(pt_file, map_location="cpu")
                else:
                    raise FileNotFoundError(f"No model files found in {transformer_dir}")
    
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    
    # Load state dict
    m, u = base_transformer.load_state_dict(state_dict, strict=False)
    print(f"Loaded transformer - missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    return base_transformer


def run_validation_on_checkpoint(checkpoint_path, step_number, args, pipeline, device, weight_dtype, config, training_output_dir=None):
    """
    Run validation on a specific checkpoint
    Args:
        training_output_dir: If provided (validation_mode), output to training_output_dir/validation/step-{step_number}
                            If None (normal mode), output to args.output_dir
    """
    print(f"\n{'='*80}")
    print(f"Running validation for checkpoint at step {step_number}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Load transformer from checkpoint
    pipeline.transformer = load_transformer_from_checkpoint(
        checkpoint_path, 
        pipeline.transformer, 
        config, 
        device, 
        weight_dtype
    )
    pipeline.transformer = pipeline.transformer.to(device, dtype=weight_dtype).eval()
    
    # Create output directory for this step
    if training_output_dir is not None:
        # Validation mode: output to training_output_dir/validation/step-{step_number}
        step_output_dir = os.path.join(training_output_dir, "validation", f"step-{step_number}")
    else:
        # Normal mode: output to args.output_dir
        step_output_dir = os.path.join(args.output_dir, f"sample_{step_number:04d}")
    os.makedirs(step_output_dir, exist_ok=True)
    
    print(f"Loading validation data from: {args.validation_json}")
    with open(args.validation_json, 'r') as f:
        validation_data = json.load(f)
    
    validation_data = validation_data[:args.validation_samples]
    json_base_dir = os.path.dirname(args.validation_json)
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    print(f"Running validation on {len(validation_data)} samples...")
    
    for idx, sample in enumerate(tqdm(validation_data, desc=f"Validating step-{step_number}")):
        try:
            # Required fields
            prompt = sample.get('text')
            if not prompt:
                print(f"Warning: 'text' field is required but missing in sample {idx}, skipping...")
                continue
            
            ref_relative_path = sample.get('ref')
            if not ref_relative_path:
                print(f"Warning: 'ref' field is required but missing in sample {idx}, skipping...")
                continue
            
            ref_coordmap_relative_path = sample.get('ref_coordmap')
            if not ref_coordmap_relative_path:
                print(f"Warning: 'ref_coordmap' field is required but missing in sample {idx}, skipping...")
                continue
            
            fg_coordmap_relative_path = sample.get('fg_coordmap')
            if not fg_coordmap_relative_path:
                print(f"Warning: 'fg_coordmap' field is required but missing in sample {idx}, skipping...")
                continue
            
            # Optional fields
            bgvideo_relative_path = sample.get('bgvideo', sample.get('bg', None))
            gt_relative_path = sample.get('gt', sample.get('video_path', None))
            firstframe_relative_path = sample.get('firstframe', None)
            
            # Resolve paths
            def resolve_path(relative_path):
                if relative_path is None:
                    return None
                if os.path.isabs(relative_path):
                    return relative_path
                return os.path.join(json_base_dir, relative_path)
            
            ref_path = resolve_path(ref_relative_path)
            ref_coordmap_path = resolve_path(ref_coordmap_relative_path)
            fg_coordmap_path = resolve_path(fg_coordmap_relative_path)
            bgvideo_path = resolve_path(bgvideo_relative_path)
            gt_path = resolve_path(gt_relative_path)
            firstframe_path = resolve_path(firstframe_relative_path)
            
            # Check required files exist
            if not os.path.exists(ref_path):
                print(f"Warning: Required ref file not found: {ref_path}, skipping...")
                continue
            
            if not os.path.exists(ref_coordmap_path):
                print(f"Warning: Required ref_coordmap file not found: {ref_coordmap_path}, skipping...")
                continue
            
            if not os.path.exists(fg_coordmap_path):
                print(f"Warning: Required fg_coordmap file not found: {fg_coordmap_path}, skipping...")
                continue
            
            # Load required data
            ref_full = load_reference_video_full(ref_path, args.height, args.width)
            if ref_full is None:
                print(f"Warning: Failed to load required ref, skipping...")
                continue
            
            ref_coordmap_full = load_validation_video_full(ref_coordmap_path, args.height, args.width)
            if ref_coordmap_full is None:
                print(f"Warning: Failed to load required ref_coordmap, skipping...")
                continue
            
            fg_coordmap = load_validation_video_full(fg_coordmap_path, args.height, args.width)
            if fg_coordmap is None:
                print(f"Warning: Failed to load required fg_coordmap, skipping...")
                continue
            
            # Load optional data
            bg_video = None
            if bgvideo_path and os.path.exists(bgvideo_path):
                bg_video = load_validation_video_full(bgvideo_path, args.height, args.width)
            
            gt_video = None
            if gt_path and os.path.exists(gt_path):
                gt_video = load_validation_video_full(gt_path, args.height, args.width)
            
            # Sample ref and ref_coordmap
            validation_ref, ref_batch_index = sample_reference_frames(ref_full, args.num_ref_frames)
            
            ref_coordmap_full_reformat = ref_coordmap_full.permute(0, 2, 1, 3, 4)
            validation_ref_coordmap = ref_coordmap_full_reformat[:, :, ref_batch_index, :, :]
            validation_ref_coordmap = validation_ref_coordmap.permute(0, 2, 1, 3, 4)
            
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
            
            video_length = int((args.num_frames - 1) // pipeline.vae.config.temporal_compression_ratio * pipeline.vae.config.temporal_compression_ratio) + 1 if args.num_frames != 1 else 1
            
            # Prepare inputs
            ref_image_input = validation_ref.to(device=device, dtype=weight_dtype)
            ref_coordmap_input = validation_ref_coordmap.to(device=device, dtype=weight_dtype)
            fg_coordmap_input = fg_coordmap.to(device=device, dtype=weight_dtype)
            bg_video_input = bg_video.to(device=device, dtype=weight_dtype) if bg_video is not None else None
            
            # Prepare start_image
            start_image_input = None
            if firstframe_path and os.path.exists(firstframe_path):
                firstframe_image = load_first_frame_image(firstframe_path, args.height, args.width)
                if firstframe_image is not None:
                    start_image_input = firstframe_image.to(device=device, dtype=weight_dtype)
            elif gt_video is not None:
                gt_input = gt_video.to(device=device, dtype=weight_dtype)
                gt_input = gt_input.permute(0, 2, 1, 3, 4)
                start_image_input = gt_input[:, :, 0:1, :, :]
            
            with torch.no_grad():
                sample_video = pipeline(
                    prompt,
                    num_frames=video_length,
                    negative_prompt=negative_prompt,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    ref_image=ref_image_input,
                    ref_coordmap=ref_coordmap_input,
                    fg_coordmap=fg_coordmap_input,
                    bg_video=bg_video_input,
                    start_image=start_image_input,
                    shift=args.shift,
                ).videos
            
            # Save outputs (same format as training script)
            if sample_video.shape[2] == 1:
                save_videos_grid(sample_video, os.path.join(step_output_dir, f"{idx}.gif"), fps=args.fps)
            else:
                save_videos_grid(sample_video, os.path.join(step_output_dir, f"{idx}.mp4"), fps=args.fps)
            
            # Save comparison (same as training script)
            ref_frames = validation_ref.shape[2]
            sample_frames = sample_video.shape[2]
            
            if ref_frames != sample_frames:
                B, C, F_ref, H, W = validation_ref.shape
                validation_ref_reshaped = validation_ref.view(B * C, 1, F_ref, H * W)
                validation_ref_interpolated = torch.nn.functional.interpolate(
                    validation_ref_reshaped,
                    size=(sample_frames, H * W),
                    mode='nearest'
                )
                validation_ref = validation_ref_interpolated.view(B, C, sample_frames, H, W)
            
            ref_h, ref_w = validation_ref.shape[3], validation_ref.shape[4]
            sample_h, sample_w = sample_video.shape[3], sample_video.shape[4]
            
            if ref_h != sample_h or ref_w != sample_w:
                scale = max(ref_h / sample_h, ref_w / sample_w)
                new_h, new_w = int(sample_h * scale), int(sample_w * scale)
                
                sample_resized = F.interpolate(
                    sample_video.view(-1, sample_video.shape[1], sample_h, sample_w),
                    size=(new_h, new_w), mode='nearest'
                )
                sample_resized = sample_resized.view(sample_video.shape[0], sample_video.shape[1], sample_video.shape[2], new_h, new_w)
                
                start_h = max(0, (new_h - ref_h) // 2)
                start_w = max(0, (new_w - ref_w) // 2)
                end_h = start_h + ref_h
                end_w = start_w + ref_w
                sample_video = sample_resized[:, :, :, start_h:end_h, start_w:end_w]
            
            if fg_coordmap_input is not None:
                viz_fg_coordmap = fg_coordmap_input.permute(0, 2, 1, 3, 4).cpu()
                if viz_fg_coordmap.shape[2] != sample_video.shape[2]:
                    viz_fg_coordmap = torch.nn.functional.interpolate(
                        viz_fg_coordmap,
                        size=(sample_video.shape[2], viz_fg_coordmap.shape[3], viz_fg_coordmap.shape[4]),
                        mode='nearest'
                    )
            else:
                viz_fg_coordmap = torch.zeros((1, 3, sample_video.shape[2], sample_video.shape[3], sample_video.shape[4]))
            
            comparison_list = [validation_ref.cpu(), sample_video.cpu(), viz_fg_coordmap]
            
            if gt_video is not None:
                gt_video_reformat = gt_video.permute(0, 2, 1, 3, 4).cpu()
                if gt_video_reformat.shape[2] != sample_video.shape[2]:
                    gt_video_reformat = torch.nn.functional.interpolate(
                        gt_video_reformat,
                        size=(sample_video.shape[2], gt_video_reformat.shape[3], gt_video_reformat.shape[4]),
                        mode='trilinear'
                    )
                comparison_list.append(gt_video_reformat)
            
            comparison_video = torch.cat(comparison_list, dim=0)
            if comparison_video.shape[2] == 1:
                comparison_frame = comparison_video[0, :, 0, :, :].permute(1, 2, 0).clamp(0, 1) * 255
                comparison_frame = comparison_frame.cpu().numpy().astype('uint8')
                Image.fromarray(comparison_frame).save(os.path.join(step_output_dir, f"{idx}_comparison.png"))
            else:
                save_videos_grid(comparison_video, os.path.join(step_output_dir, f"{idx}_comparison.mp4"), fps=args.fps)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nValidation complete for step {step_number}! Results saved to: {step_output_dir}")


def main():
    args = parse_args()
    
    # Validate arguments for validation_mode
    if args.validation_mode:
        if args.custom_transformer_path is None:
            raise ValueError("--custom_transformer_path is required when --validation_mode is enabled")
        
        # Use custom_transformer_path directly as training_output_dir
        training_output_dir = os.path.abspath(args.custom_transformer_path)
        
        # Check if logs directory exists (validation that this is a training output dir)
        logs_dir = os.path.join(training_output_dir, "logs")
        if not os.path.exists(logs_dir):
            raise ValueError(f"logs directory not found in {training_output_dir}. Make sure --custom_transformer_path points to the training output directory (e.g., ckpts/your_experiment).")
        
        print(f"Validation mode enabled. Monitoring: {training_output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }[args.weight_dtype]
    
    config = OmegaConf.load(args.config_path)
    
    print("Loading models...")
    
    transformer = CroodRefTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype,
    )
    
    # In validation_mode
    if args.custom_transformer_path is not None and not args.validation_mode:
        print(f"Loading custom transformer from: {args.custom_transformer_path}")
        from safetensors.torch import load_file
        
        if os.path.isdir(args.custom_transformer_path):
            index_files = [f for f in os.listdir(args.custom_transformer_path) if f.endswith('.safetensors.index.json')]
            
            if index_files:
                index_file = os.path.join(args.custom_transformer_path, index_files[0])
                with open(index_file, 'r') as f:
                    index = json.load(f)
                state_dict = {}
                weight_map = index["weight_map"]
                shard_files = set(weight_map.values())
                for shard_file in shard_files:
                    shard_path = os.path.join(args.custom_transformer_path, shard_file)
                    shard_state_dict = load_file(shard_path)
                    state_dict.update(shard_state_dict)
            else:
                safetensors_files = [f for f in os.listdir(args.custom_transformer_path) if f.endswith('.safetensors')]
                if not safetensors_files:
                    raise FileNotFoundError(f"No .safetensors files found in directory: {args.custom_transformer_path}")
                
                state_dict = {}
                for safetensors_file in safetensors_files:
                    safetensors_path = os.path.join(args.custom_transformer_path, safetensors_file)
                    shard_state_dict = load_file(safetensors_path)
                    state_dict.update(shard_state_dict)
        elif args.custom_transformer_path.endswith(".safetensors"):
            state_dict = load_file(args.custom_transformer_path)
        else:
            state_dict = torch.load(args.custom_transformer_path, map_location="cpu")
        
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"Loaded transformer - missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    if args.vae_path is not None:
        print(f"Loading VAE from checkpoint: {args.vae_path}")
        from safetensors.torch import load_file
        if args.vae_path.endswith(".safetensors"):
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"Loaded VAE - missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    
    # Load clip_image_encoder
    clip_image_encoder_path = os.path.join(args.model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder'))
    if os.path.exists(clip_image_encoder_path):
        clip_image_encoder = CLIPModel.from_pretrained(clip_image_encoder_path).to(device, dtype=weight_dtype).eval()
    else:
        print(f"Warning: CLIPModel not found at {clip_image_encoder_path}, using None")
        clip_image_encoder = None
    
    Chosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[args.sampler_name]
    
    if args.sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1
    
    scheduler = Chosen_Scheduler(
        **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    pipeline = WanFunCroodRefPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
    
    if args.gpu_memory_mode == "sequential_cpu_offload":
        from videox_fun.utils.fp8_optimization import replace_parameters_by_name
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline = pipeline.to(device)
    
    # If validation_mode is enabled, enter monitoring loop
    if args.validation_mode:
        print("\n" + "="*80)
        print("VALIDATION MODE ACTIVATED")
        print("="*80)
        print(f"Models loaded and kept in GPU memory")
        print(f"Monitoring directory: {training_output_dir}")
        print(f"Check interval: {args.check_interval} seconds")
        print(f"Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        processed_steps = set()
        
        try:
            while True:
                # Check for new validation checkpoint
                checkpoint_path, step_number = find_validation_checkpoint(training_output_dir)
                
                if checkpoint_path and step_number not in processed_steps:
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] New validation checkpoint detected!")
                    
                    # Run validation with training_output_dir
                    run_validation_on_checkpoint(
                        checkpoint_path, 
                        step_number, 
                        args, 
                        pipeline, 
                        device, 
                        weight_dtype,
                        config,
                        training_output_dir=training_output_dir
                    )
                    
                    processed_steps.add(step_number)
                    
                    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Validation complete. Resuming monitoring...")
                else:
                    # No new checkpoint, wait
                    if checkpoint_path:
                        print(f"\r[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring... (Last processed: step-{step_number})", end="", flush=True)
                    else:
                        print(f"\r[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring... (No checkpoints found yet)", end="", flush=True)
                
                time.sleep(args.check_interval)
                
        except KeyboardInterrupt:
            print("\n\nValidation mode stopped by user.")
            print(f"Processed steps: {sorted(processed_steps)}")
            return
    
    # Normal mode: run validation once
    print(f"Loading validation data from: {args.validation_json}")
    with open(args.validation_json, 'r') as f:
        validation_data = json.load(f)
    
    validation_data = validation_data[:args.validation_samples]
    
    json_base_dir = os.path.dirname(args.validation_json)
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    print(f"Running validation on {len(validation_data)} samples...")
    
    for idx, sample in enumerate(tqdm(validation_data)):
        try:
            # Required fields
            prompt = sample.get('text')
            if not prompt:
                print(f"Warning: 'text' field is required but missing in sample {idx}, skipping...")
                continue
            
            ref_relative_path = sample.get('ref')
            if not ref_relative_path:
                print(f"Warning: 'ref' field is required but missing in sample {idx}, skipping...")
                continue
            
            ref_coordmap_relative_path = sample.get('ref_coordmap')
            if not ref_coordmap_relative_path:
                print(f"Warning: 'ref_coordmap' field is required but missing in sample {idx}, skipping...")
                continue
            
            fg_coordmap_relative_path = sample.get('fg_coordmap')
            if not fg_coordmap_relative_path:
                print(f"Warning: 'fg_coordmap' field is required but missing in sample {idx}, skipping...")
                continue
            
            # Optional fields
            bgvideo_relative_path = sample.get('bgvideo', sample.get('bg', None))
            gt_relative_path = sample.get('gt', sample.get('video_path', None))
            firstframe_relative_path = sample.get('firstframe', None)
            
            # Resolve paths (support both relative and absolute paths)
            def resolve_path(relative_path):
                if relative_path is None:
                    return None
                if os.path.isabs(relative_path):
                    return relative_path
                return os.path.join(json_base_dir, relative_path)
            
            ref_path = resolve_path(ref_relative_path)
            ref_coordmap_path = resolve_path(ref_coordmap_relative_path)
            fg_coordmap_path = resolve_path(fg_coordmap_relative_path)
            bgvideo_path = resolve_path(bgvideo_relative_path)
            gt_path = resolve_path(gt_relative_path)
            firstframe_path = resolve_path(firstframe_relative_path)
            
            # Check required files exist
            if not os.path.exists(ref_path):
                print(f"Warning: Required ref file not found: {ref_path}, skipping...")
                continue
            
            if not os.path.exists(ref_coordmap_path):
                print(f"Warning: Required ref_coordmap file not found: {ref_coordmap_path}, skipping...")
                continue
            
            if not os.path.exists(fg_coordmap_path):
                print(f"Warning: Required fg_coordmap file not found: {fg_coordmap_path}, skipping...")
                continue
            
            print(f"\n[{idx+1}/{len(validation_data)}] Processing: {os.path.basename(ref_path)}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Load required data
            ref_full = load_reference_video_full(ref_path, args.height, args.width)
            if ref_full is None:
                print(f"Warning: Failed to load required ref, skipping...")
                continue
            
            ref_coordmap_full = load_validation_video_full(ref_coordmap_path, args.height, args.width)
            if ref_coordmap_full is None:
                print(f"Warning: Failed to load required ref_coordmap, skipping...")
                continue
            
            fg_coordmap = load_validation_video_full(fg_coordmap_path, args.height, args.width)
            if fg_coordmap is None:
                print(f"Warning: Failed to load required fg_coordmap, skipping...")
                continue
            
            # Load optional data
            bg_video = None
            if bgvideo_path and os.path.exists(bgvideo_path):
                bg_video = load_validation_video_full(bgvideo_path, args.height, args.width)
            
            gt_video = None
            if gt_path and os.path.exists(gt_path):
                gt_video = load_validation_video_full(gt_path, args.height, args.width)
            
            # Step 2: Sample ref and ref_coordmap using unified sampling logic
            validation_ref, ref_batch_index = sample_reference_frames(ref_full, args.num_ref_frames)
            
            # Sample ref_coordmap using the SAME indices as ref
            ref_coordmap_full_reformat = ref_coordmap_full.permute(0, 2, 1, 3, 4)  # [1, F, C, H, W] -> [1, C, F, H, W]
            validation_ref_coordmap = ref_coordmap_full_reformat[:, :, ref_batch_index, :, :]
            # Convert back to [1, F, C, H, W]
            validation_ref_coordmap = validation_ref_coordmap.permute(0, 2, 1, 3, 4)
            print(f"Sampled ref_coordmap with shape: {validation_ref_coordmap.shape}")
            
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
            
            video_length = int((args.num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.num_frames != 1 else 1
            
            # Prepare required inputs
            ref_image_input = validation_ref.to(device=device, dtype=weight_dtype)
            ref_coordmap_input = validation_ref_coordmap.to(device=device, dtype=weight_dtype)
            fg_coordmap_input = fg_coordmap.to(device=device, dtype=weight_dtype)
            
            # Prepare optional inputs
            bg_video_input = bg_video.to(device=device, dtype=weight_dtype) if bg_video is not None else None
            
            # Prepare start_image from first frame of GT video if provided
            start_image_input = None
            
            # Priority Use firstframe if specified in JSON
            if firstframe_path and os.path.exists(firstframe_path):
                firstframe_image = load_first_frame_image(firstframe_path, args.height, args.width)
                if firstframe_image is not None:
                    start_image_input = firstframe_image.to(device=device, dtype=weight_dtype)
                    print(f"Start image input from firstframe: {start_image_input.shape}")
            
            # Extract from GT video if no firstframe and GT is available
            elif gt_video is not None:
                gt_input = gt_video.to(device=device, dtype=weight_dtype)
                # Rearrange from [B, F, C, H, W] to [B, C, F, H, W]
                gt_input = gt_input.permute(0, 2, 1, 3, 4)
                # Extract first frame: [1, C, 1, H, W]
                start_image_input = gt_input[:, :, 0:1, :, :]
                print(f"Start image input from GT video: {start_image_input.shape}")
            
            with torch.no_grad():
                sample_video = pipeline(
                    prompt,
                    num_frames=video_length,
                    negative_prompt=negative_prompt,
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    ref_image=ref_image_input,
                    ref_coordmap=ref_coordmap_input,
                    fg_coordmap=fg_coordmap_input,
                    bg_video=bg_video_input,
                    start_image=start_image_input,
                    shift=args.shift,
                ).videos
            
            sample_dir = os.path.join(args.output_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            if sample_video.shape[2] == 1:
                save_videos_grid(sample_video, os.path.join(sample_dir, "generated.gif"), fps=args.fps)
            else:
                save_videos_grid(sample_video, os.path.join(sample_dir, "generated.mp4"), fps=args.fps)
            
            if validation_ref is not None:
                if validation_ref.shape[2] == 1:
                    save_videos_grid(validation_ref, os.path.join(sample_dir, "reference.gif"), fps=args.fps)
                else:
                    save_videos_grid(validation_ref, os.path.join(sample_dir, "reference.mp4"), fps=args.fps)
            
            if args.save_comparison:
                ref_frames = validation_ref.shape[2]
                sample_frames = sample_video.shape[2]
                
                if ref_frames != sample_frames:
                    B, C, F_ref, H, W = validation_ref.shape
                    validation_ref_reshaped = validation_ref.view(B * C, 1, F_ref, H * W)
                    validation_ref_interpolated = torch.nn.functional.interpolate(
                        validation_ref_reshaped,
                        size=(sample_frames, H * W),
                        mode='nearest'
                    )
                    validation_ref = validation_ref_interpolated.view(B, C, sample_frames, H, W)
                
                ref_h, ref_w = validation_ref.shape[3], validation_ref.shape[4]
                sample_h, sample_w = sample_video.shape[3], sample_video.shape[4]
                
                if ref_h != sample_h or ref_w != sample_w:
                    scale = max(ref_h / sample_h, ref_w / sample_w)
                    new_h, new_w = int(sample_h * scale), int(sample_w * scale)
                    
                    sample_resized = F.interpolate(
                        sample_video.view(-1, sample_video.shape[1], sample_h, sample_w),
                        size=(new_h, new_w), mode='nearest'
                    )
                    sample_resized = sample_resized.view(sample_video.shape[0], sample_video.shape[1], sample_video.shape[2], new_h, new_w)
                    
                    start_h = max(0, (new_h - ref_h) // 2)
                    start_w = max(0, (new_w - ref_w) // 2)
                    end_h = start_h + ref_h
                    end_w = start_w + ref_w
                    sample_video = sample_resized[:, :, :, start_h:end_h, start_w:end_w]
                
                # Use fg_coordmap for visualization if available, otherwise use zeros
                if fg_coordmap_input is not None:
                    viz_fg_coordmap = fg_coordmap_input.permute(0, 2, 1, 3, 4).cpu()
                    # Resample fg_coordmap to match sample_video frames if needed
                    if viz_fg_coordmap.shape[2] != sample_video.shape[2]:
                        print(f"Resampling fg_coordmap from {viz_fg_coordmap.shape[2]} to {sample_video.shape[2]} frames for comparison")
                        viz_fg_coordmap = torch.nn.functional.interpolate(
                            viz_fg_coordmap,
                            size=(sample_video.shape[2], viz_fg_coordmap.shape[3], viz_fg_coordmap.shape[4]),
                            mode='nearest'
                        )
                else:
                    viz_fg_coordmap = torch.zeros((1, 3, sample_video.shape[2], sample_video.shape[3], sample_video.shape[4]))
                
                comparison_list = [validation_ref.cpu(), sample_video.cpu(), viz_fg_coordmap]
                
                # Add gt_video if available
                if gt_video is not None:
                    gt_video_reformat = gt_video.permute(0, 2, 1, 3, 4).cpu()
                    # Resample gt_video to match sample_video frames if needed
                    if gt_video_reformat.shape[2] != sample_video.shape[2]:
                        print(f"Resampling gt_video from {gt_video_reformat.shape[2]} to {sample_video.shape[2]} frames for comparison")
                        gt_video_reformat = torch.nn.functional.interpolate(
                            gt_video_reformat,
                            size=(sample_video.shape[2], gt_video_reformat.shape[3], gt_video_reformat.shape[4]),
                            mode='trilinear'
                        )
                    comparison_list.append(gt_video_reformat)
                
                comparison_video = torch.cat(comparison_list, dim=0)
                if comparison_video.shape[2] == 1:
                    comparison_frame = comparison_video[0, :, 0, :, :].permute(1, 2, 0).clamp(0, 1) * 255
                    comparison_frame = comparison_frame.cpu().numpy().astype('uint8')
                    Image.fromarray(comparison_frame).save(os.path.join(sample_dir, "comparison.png"))
                else:
                    save_videos_grid(comparison_video, os.path.join(sample_dir, "comparison.mp4"), fps=args.fps)
            
            with open(os.path.join(sample_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Reference: {ref_path}\n")
                f.write(f"Ref Coordmap: {ref_coordmap_path}\n")
                f.write(f"FG Coordmap: {fg_coordmap_path}\n")
                if bgvideo_path:
                    f.write(f"BG Video: {bgvideo_path}\n")
                if firstframe_path:
                    f.write(f"First Frame: {firstframe_path}\n")
                if gt_path:
                    f.write(f"GT Video: {gt_path}\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nValidation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

