import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
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
    WanTransformer3DModel
)
from videox_fun.pipeline import WanFunPhantomPipeline
from videox_fun.utils.utils import filter_kwargs, save_videos_grid, get_image_latent
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for Wan2.1 Phantom model")
    
    # Model paths
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Path to config file")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.1-T2V-1.3B", help="Base model path")
    parser.add_argument("--custom_transformer_path", type=str, default=None, help="Custom transformer checkpoint path")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE checkpoint path")
    
    # Validation dataset
    parser.add_argument("--validation_json", type=str, required=True, help="Path to validation json")
    parser.add_argument("--validation_samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--validation_ref_dir", type=str, default=None, help="Directory containing reference videos/images")
    
    # Generation parameters
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=448, help="Video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--fps", type=int, default=16, help="FPS for output video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Text guidance scale")
    parser.add_argument("--shift", type=int, default=5, help="Shift parameter for scheduler")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/phantom_ref_validation", help="Output directory")
    parser.add_argument("--save_comparison", action="store_true", help="Save comparison videos (ref + generated)")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Weight dtype")
    
    # Scheduler
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    
    # Memory optimization
    parser.add_argument("--gpu_memory_mode", type=str, default="model_full_load", 
                        choices=["model_full_load", "model_cpu_offload", "sequential_cpu_offload"],
                        help="GPU memory mode")
    
    return parser.parse_args()


def get_video_frames_as_latent(video_path, sample_size, padding=False, max_frames=None):
    """
    Extract frames from a video file and convert to latent format.
    Randomly and uniformly samples 4 frames from the video.
    
    Args:
        video_path: Path to video file
        sample_size: [height, width] target size
        padding: Whether to pad the frames
        max_frames: Maximum number of frames to extract (deprecated, always samples 4 frames)
    
    Returns:
        torch.Tensor: Video frames in format [1, C, F, H, W]
    """
    vr = VideoReader(video_path)
    total_frames = len(vr)
    
    # Randomly and uniformly sample 6 frames
    num_sample_frames = 6
    if total_frames <= num_sample_frames:
        # If video has 6 or fewer frames, use all frames
        frame_indices = list(range(total_frames))
    else:
        # Uniformly sample 6 frames
        frame_indices = np.linspace(0, total_frames - 1, num_sample_frames, dtype=int).tolist()
    
    # Process each sampled frame using get_image_latent
    ref_list = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        # Convert to PIL Image
        frame_pil = Image.fromarray(frame)
        
        # Use get_image_latent to process (returns [1, C, 1, H, W])
        frame_latent = get_image_latent(ref_image=frame_pil, sample_size=sample_size, padding=padding)
        # Squeeze the frame dimension: [1, C, 1, H, W] -> [1, C, H, W]
        frame_latent = frame_latent.squeeze(2)
        ref_list.append(frame_latent)
    
    # Concatenate all frames: list of [1, C, H, W] -> [1, C, F, H, W]
    frames_tensor = torch.cat([f.unsqueeze(2) for f in ref_list], dim=2)
    
    return frames_tensor


def match_video_size(ref_video, sample_video):
    """Match reference and sample video sizes for comparison"""
    ref_frames = ref_video.shape[2]
    sample_frames = sample_video.shape[2]
    
    # Match frame count using nearest interpolation
    if ref_frames != sample_frames:
        print(f"Matching video size: ref_frames={ref_frames}, sample_frames={sample_frames}")
        B, C, F_ref, H, W = ref_video.shape
        ref_video_reshaped = ref_video.view(B * C, 1, F_ref, H * W)
        ref_video_interpolated = torch.nn.functional.interpolate(
            ref_video_reshaped,
            size=(sample_frames, H * W),
            mode='nearest'
        )
        ref_video = ref_video_interpolated.view(B, C, sample_frames, H, W)
    
    # Match spatial size
    ref_h, ref_w = ref_video.shape[3], ref_video.shape[4]
    sample_h, sample_w = sample_video.shape[3], sample_video.shape[4]
    
    if ref_h != sample_h or ref_w != sample_w:
        import torch.nn.functional as F
        scale = max(ref_h / sample_h, ref_w / sample_w)
        new_h, new_w = int(sample_h * scale), int(sample_w * scale)
        
        # Resize
        sample_resized = F.interpolate(
            sample_video.view(-1, sample_video.shape[1], sample_h, sample_w),
            size=(new_h, new_w), mode='bilinear', align_corners=False
        )
        sample_resized = sample_resized.view(
            sample_video.shape[0], sample_video.shape[1], sample_video.shape[2], new_h, new_w
        )
        
        # Center crop
        start_h = max(0, (new_h - ref_h) // 2)
        start_w = max(0, (new_w - ref_w) // 2)
        end_h = start_h + ref_h
        end_w = start_w + ref_w
        sample_video = sample_resized[:, :, :, start_h:end_h, start_w:end_w]
    
    return ref_video, sample_video


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    weight_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }[args.weight_dtype]
    
    # Load config
    config = OmegaConf.load(args.config_path)
    
    # Load models
    print("Loading models...")
    
    # Load transformer
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    # Load custom transformer checkpoint if provided
    if args.custom_transformer_path is not None:
        print(f"Loading custom transformer from: {args.custom_transformer_path}")
        from safetensors.torch import load_file
        
        if os.path.isdir(args.custom_transformer_path):
            # Load from directory with multiple shards
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
                # Load all safetensors files
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
    
    # Load VAE
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
    
    # Load tokenizer and text encoder
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
    
    # Load scheduler
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
    
    # Create pipeline
    pipeline = WanFunPhantomPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    
    # Apply memory optimization
    if args.gpu_memory_mode == "sequential_cpu_offload":
        from videox_fun.utils.fp8_optimization import replace_parameters_by_name
        replace_parameters_by_name(transformer, ["modulation",], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline = pipeline.to(device)
    
    # Load validation data
    print(f"Loading validation data from: {args.validation_json}")
    with open(args.validation_json, 'r') as f:
        validation_data = json.load(f)
    
    # Limit samples
    validation_data = validation_data[:args.validation_samples]
    
    # Get base directory for reference paths
    json_base_dir = os.path.dirname(args.validation_json)
    
    # Negative prompt
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    
    # Run validation
    print(f"Running validation on {len(validation_data)} samples...")
    
    for idx, sample in enumerate(tqdm(validation_data)):
        try:
            # Get prompt
            prompt = sample.get('text', sample.get('cap', sample.get('caption', '')))
            
            # Get reference paths (support multiple references)
            ref_info = sample.get('ref', sample.get('ref_path', sample.get('video_path', '')))
            
            # Handle both single ref and multiple refs
            if isinstance(ref_info, list):
                ref_relative_paths = ref_info
            else:
                ref_relative_paths = [ref_info]
            
            # Construct full reference paths
            ref_paths = []
            for ref_relative_path in ref_relative_paths:
                if args.validation_ref_dir is not None:
                    ref_path = os.path.join(args.validation_ref_dir, os.path.basename(ref_relative_path))
                else:
                    ref_path = os.path.join(json_base_dir, ref_relative_path)
                
                if not os.path.exists(ref_path):
                    print(f"Warning: Reference not found: {ref_path}, skipping...")
                    continue
                ref_paths.append(ref_path)
            
            if not ref_paths:
                print(f"No valid references found for sample {idx}, skipping...")
                continue
            
            print(f"\n[{idx+1}/{len(validation_data)}] Processing: {[os.path.basename(p) for p in ref_paths]}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Load reference images/videos (same as predict_s2v.py)
            subject_ref_images = None
            if ref_paths is not None:
                # If subject_ref_images is a string (single video/image path), convert to list
                if isinstance(ref_paths, str):
                    ref_paths = [ref_paths]
                
                processed_refs = []
                for _subject_ref_image in ref_paths:
                    # Check if it's a directory (folder of images)
                    if os.path.isdir(_subject_ref_image):
                        # Load all images from the directory
                        image_files = sorted([
                            os.path.join(_subject_ref_image, f) 
                            for f in os.listdir(_subject_ref_image) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'))
                        ])
                        for img_file in image_files:
                            img = get_image_latent(img_file, sample_size=[args.height, args.width], padding=True)
                            processed_refs.append(img)
                    # Check if it's a video file
                    elif _subject_ref_image.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                        # Load video frames
                        frames = get_video_frames_as_latent(_subject_ref_image, sample_size=[args.height, args.width], padding=True)
                        processed_refs.append(frames)
                    else:
                        # Load single image
                        img = get_image_latent(_subject_ref_image, sample_size=[args.height, args.width], padding=True)
                        processed_refs.append(img)
                
                subject_ref_images = torch.cat(processed_refs, dim=2)
            
            # Setup generator
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
            
            # Adjust video length for VAE temporal compression
            video_length = int((args.num_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.num_frames != 1 else 1
            
            # Generate
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
                    subject_ref_images=subject_ref_images.to(device=device, dtype=weight_dtype),
                    shift=args.shift,
                ).videos
            
            # Save results
            sample_dir = os.path.join(args.output_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save generated video
            if sample_video.shape[2] == 1:
                save_videos_grid(sample_video, os.path.join(sample_dir, "generated.png"))
            else:
                save_videos_grid(sample_video, os.path.join(sample_dir, "generated.mp4"), fps=args.fps)
            
            # Save reference images as grid or video
            ref_video = subject_ref_images
            if ref_video.shape[2] == 1:
                save_videos_grid(ref_video, os.path.join(sample_dir, "reference.png"))
            else:
                save_videos_grid(ref_video, os.path.join(sample_dir, "reference.mp4"), fps=args.fps)
            
            # Save comparison if requested
            if args.save_comparison:
                ref_matched, sample_matched = match_video_size(ref_video, sample_video)
                comparison_video = torch.cat([ref_matched, sample_matched], dim=0)
                
                if comparison_video.shape[2] == 1:
                    save_videos_grid(comparison_video, os.path.join(sample_dir, "comparison.png"))
                else:
                    save_videos_grid(comparison_video, os.path.join(sample_dir, "comparison.mp4"), fps=args.fps)
            
            # Save prompt
            with open(os.path.join(sample_dir, "prompt.txt"), 'w') as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"References:\n")
                for ref_path in ref_paths:
                    f.write(f"  - {ref_path}\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nValidation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

