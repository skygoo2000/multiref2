import argparse
import glob
import json
import os
import sys
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
    MultiRefTransformer3DModel
)
from videox_fun.pipeline import WanFunPoseRefPipeline
from videox_fun.utils.utils import filter_kwargs, save_videos_grid, get_image_latent
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.data.dataset_image_video import padding_image, pose_dict_to_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for Wan2.1 Phantom model")
    
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Path to config file")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.1-T2V-1.3B", help="Base model path")
    parser.add_argument("--custom_transformer_path", type=str, default=None, help="Custom transformer checkpoint path")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE checkpoint path")
    
    parser.add_argument("--validation_json", type=str, required=True, help="Path to validation json")
    parser.add_argument("--validation_samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--validation_ref_dir", type=str, default=None, help="Directory containing reference videos/images")
    
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=448, help="Video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--num_ref_frames", type=int, default=4, help="Number of reference frames to sample from video")
    parser.add_argument("--fps", type=int, default=16, help="FPS for output video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Text guidance scale")
    parser.add_argument("--shift", type=int, default=5, help="Shift parameter for scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--output_dir", type=str, default="outputs/phantom_ref_validation", help="Output directory")
    parser.add_argument("--save_comparison", action="store_true", help="Save comparison videos (ref + generated)")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Weight dtype")
    
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    
    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load", 
                        choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"],
                        help="GPU memory mode")
    
    return parser.parse_args()


def load_reference_video_full(ref_path, val_height, val_width):
    """
    Load full reference video/images without sampling
    Returns: torch.Tensor [1, C, F, H, W] or None
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
            frame_latent = get_image_latent(ref_image=img_path, sample_size=sample_size, padding=True)
            ref_list.append(frame_latent)
        
        validation_ref = torch.cat(ref_list, dim=2)
        print(f"Loaded reference from directory with all {len(image_files)} frames, shape: {validation_ref.shape}")
        return validation_ref
    
    ref_ext = ref_path.lower().split('.')[-1]
    if ref_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
        sample_size = [val_height, val_width]
        validation_ref = get_image_latent(ref_image=ref_path, sample_size=sample_size, padding=True)
        print(f"Loaded reference image with shape: {validation_ref.shape}")
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
            frame_tensor = frame_tensor.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255
            frame_tensor = frame_tensor.squeeze(2)
            ref_list.append(frame_tensor)
        
        validation_ref = torch.cat([f.unsqueeze(2) for f in ref_list], dim=2)
        print(f"Loaded reference video with all {total_frames} frames, shape: {validation_ref.shape}")
        return validation_ref
        
    except Exception as e:
        print(f"Failed to load reference video: {e}")
        return None


def sample_reference_frames(ref_full, num_ref_frames):
    """
    Sample specified number of frames from full reference video
    Using same logic as dataset_image_video.py (1110-1124)
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
                ref_batch_index.append(random_idx)
    
    sampled_ref = ref_full[:, :, ref_batch_index, :, :]
    print(f"Sampled {len(ref_batch_index)} frames from {total_frames} total frames, indices: {ref_batch_index}")
    return sampled_ref, ref_batch_index


def load_validation_mask_full(mask_path, val_height, val_width):
    """
    Load full mask video without sampling
    Returns: torch.Tensor [1, F, C, H, W] or None
    """
    try:
        vr = VideoReader(mask_path)
        total_frames = len(vr)
        
        mask_frames = []
        for idx in range(total_frames):
            frame = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((val_width, val_height))
            mask_frames.append(np.array(frame_pil))
        
        validation_mask = np.stack(mask_frames, axis=0)
        validation_mask = torch.from_numpy(validation_mask).permute(0, 3, 1, 2).float() / 255.0
        validation_mask = validation_mask * 2.0 - 1.0
        validation_mask = validation_mask.unsqueeze(0)
        print(f"Loaded full validation mask with shape: {validation_mask.shape}")
        return validation_mask
    except Exception as e:
        print(f"Failed to load validation mask: {e}")
        return None


def load_validation_fg_full(fg_path, val_height, val_width):
    """
    Load full fg video without sampling
    Returns: torch.Tensor [1, F, C, H, W] or None
    """
    try:
        vr = VideoReader(fg_path)
        total_frames = len(vr)
        
        fg_frames = []
        for idx in range(total_frames):
            frame = vr[idx].asnumpy()
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((val_width, val_height))
            fg_frames.append(np.array(frame_pil))
        
        validation_fg = np.stack(fg_frames, axis=0)
        validation_fg = torch.from_numpy(validation_fg).permute(0, 3, 1, 2).float() / 255.0
        validation_fg = validation_fg * 2.0 - 1.0
        validation_fg = validation_fg.unsqueeze(0)
        print(f"Loaded full validation fg with shape: {validation_fg.shape}")
        return validation_fg
    except Exception as e:
        print(f"Failed to load validation fg: {e}")
        return None


def load_validation_pose_full(pose_path):
    """
    Load full pose data without sampling
    Returns: (ref_poses, video_poses) tuple, each is numpy array or None
    """
    try:
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        video_poses = None
        ref_poses = None
        
        if 'video' in pose_data:
            video_pose_list = []
            for pose_dict in pose_data['video']:
                pose_matrix = pose_dict_to_matrix(pose_dict)
                video_pose_list.append(pose_matrix)
            
            if video_pose_list:
                video_poses = np.stack(video_pose_list, axis=0)
                print(f"Loaded full video_pose with {len(video_pose_list)} frames")
        
        if 'ref' in pose_data:
            ref_pose_list = []
            for pose_dict in pose_data['ref']:
                pose_matrix = pose_dict_to_matrix(pose_dict)
                ref_pose_list.append(pose_matrix)
            
            if ref_pose_list:
                ref_poses = np.stack(ref_pose_list, axis=0)
                print(f"Loaded full ref_pose with {len(ref_pose_list)} frames")
        
        return ref_poses, video_poses
    
    except Exception as e:
        print(f"Failed to load validation pose: {e}")
        return None, None


def sample_pose_frames(poses, indices):
    """
    Sample pose frames based on indices
    Args:
        poses: numpy array [F, 4, 4]
        indices: List[int] frame indices to sample
    Returns:
        torch.Tensor [1, len(indices), 4, 4]
    """
    if poses is None:
        return None
    
    sampled_poses = poses[indices]
    sampled_poses = torch.from_numpy(sampled_poses).unsqueeze(0)
    print(f"Sampled {len(indices)} poses from {len(poses)} total poses")
    return sampled_poses


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }[args.weight_dtype]
    
    config = OmegaConf.load(args.config_path)
    
    print("Loading models...")
    
    transformer = MultiRefTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    if args.custom_transformer_path is not None:
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
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()
    
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
    
    pipeline = WanFunPoseRefPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
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
    
    print(f"Loading validation data from: {args.validation_json}")
    with open(args.validation_json, 'r') as f:
        validation_data = json.load(f)
    
    validation_data = validation_data[:args.validation_samples]
    
    json_base_dir = os.path.dirname(args.validation_json)
    
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    print(f"Running validation on {len(validation_data)} samples...")
    
    for idx, sample in enumerate(tqdm(validation_data)):
        try:
            prompt = sample.get('text', sample.get('cap', sample.get('caption', '')))
            
            ref_relative_path = sample.get('ref', sample.get('ref_path', sample.get('video_path', '')))
            mask_relative_path = sample.get('mask', sample.get('mask_path', ''))
            fg_relative_path = sample.get('fg', sample.get('fg_path', ''))
            pose_relative_path = sample.get('pose', sample.get('pose_path', ''))
            
            if args.validation_ref_dir is not None:
                ref_path = os.path.join(args.validation_ref_dir, os.path.basename(ref_relative_path)) if ref_relative_path else None
                mask_path = os.path.join(args.validation_ref_dir, os.path.basename(mask_relative_path)) if mask_relative_path else None
                fg_path = os.path.join(args.validation_ref_dir, os.path.basename(fg_relative_path)) if fg_relative_path else None
                pose_path = os.path.join(args.validation_ref_dir, os.path.basename(pose_relative_path)) if pose_relative_path else None
            else:
                ref_path = os.path.join(json_base_dir, ref_relative_path) if ref_relative_path else None
                mask_path = os.path.join(json_base_dir, mask_relative_path) if mask_relative_path else None
                fg_path = os.path.join(json_base_dir, fg_relative_path) if fg_relative_path else None
                pose_path = os.path.join(json_base_dir, pose_relative_path) if pose_relative_path else None
            
            if ref_path and not os.path.exists(ref_path):
                print(f"Warning: Reference not found: {ref_path}, skipping...")
                continue
            
            print(f"\n[{idx+1}/{len(validation_data)}] Processing: {os.path.basename(ref_path) if ref_path else 'No ref'}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Step 1: Load all data without sampling
            ref_full = None
            if ref_path and os.path.exists(ref_path):
                ref_full = load_reference_video_full(ref_path, args.height, args.width)
            
            validation_mask = None
            if mask_path and os.path.exists(mask_path):
                validation_mask = load_validation_mask_full(mask_path, args.height, args.width)
            
            validation_fg = None
            if fg_path and os.path.exists(fg_path):
                validation_fg = load_validation_fg_full(fg_path, args.height, args.width)
            
            ref_poses_full = None
            video_poses_full = None
            if pose_path and os.path.exists(pose_path):
                ref_poses_full, video_poses_full = load_validation_pose_full(pose_path)
            
            # Step 2: Sample ref and ref_pose using unified sampling logic
            validation_ref = None
            validation_ref_pose = None
            ref_batch_index = None
            
            if ref_full is not None:
                validation_ref, ref_batch_index = sample_reference_frames(ref_full, args.num_ref_frames)
                
                if ref_poses_full is not None:
                    validation_ref_pose = sample_pose_frames(ref_poses_full, ref_batch_index)
            
            # video_pose keeps full
            validation_video_pose = None
            if video_poses_full is not None:
                validation_video_pose = torch.from_numpy(video_poses_full).unsqueeze(0)
            
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
            
            video_length = int((args.num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.num_frames != 1 else 1
            
            ref_video_input = validation_ref.to(device=device, dtype=weight_dtype) if validation_ref is not None else None
            mask_input = validation_mask.to(device=device, dtype=weight_dtype) if validation_mask is not None else None
            fg_input = validation_fg.to(device=device, dtype=weight_dtype) if validation_fg is not None else None
            ref_pose_input = validation_ref_pose.to(device=device, dtype=weight_dtype) if validation_ref_pose is not None else None
            video_pose_input = validation_video_pose.to(device=device, dtype=weight_dtype) if validation_video_pose is not None else None
            
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
                    subject_ref_images=ref_video_input,
                    bg_mask=mask_input,
                    fg=fg_input,
                    ref_pose=ref_pose_input,
                    video_pose=video_pose_input,
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
            
            if args.save_comparison and validation_ref is not None:
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
                        size=(new_h, new_w), mode='bilinear', align_corners=False
                    )
                    sample_resized = sample_resized.view(sample_video.shape[0], sample_video.shape[1], sample_video.shape[2], new_h, new_w)
                    
                    start_h = max(0, (new_h - ref_h) // 2)
                    start_w = max(0, (new_w - ref_w) // 2)
                    end_h = start_h + ref_h
                    end_w = start_w + ref_w
                    sample_video = sample_resized[:, :, :, start_h:end_h, start_w:end_w]
                
                comparison_video = torch.cat([validation_ref, sample_video], dim=0)
                if comparison_video.shape[2] == 1:
                    comparison_frame = comparison_video[0, :, 0, :, :].permute(1, 2, 0).clamp(0, 1) * 255
                    comparison_frame = comparison_frame.cpu().numpy().astype('uint8')
                    Image.fromarray(comparison_frame).save(os.path.join(sample_dir, "comparison.png"))
                else:
                    save_videos_grid(comparison_video, os.path.join(sample_dir, "comparison.mp4"), fps=args.fps)
            
            with open(os.path.join(sample_dir, "prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {prompt}\n")
                if ref_path:
                    f.write(f"Reference: {ref_path}\n")
                if mask_path:
                    f.write(f"Mask: {mask_path}\n")
                if fg_path:
                    f.write(f"FG: {fg_path}\n")
                if pose_path:
                    f.write(f"Pose: {pose_path}\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nValidation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
