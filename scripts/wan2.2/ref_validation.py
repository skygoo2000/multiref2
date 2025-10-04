import argparse
import json
import os
import sys
from pathlib import Path

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
    AutoencoderKLWan3_8,
    WanT5EncoderModel,
    AutoTokenizer,
    Wan2_2RefTransformer3DModel
)
from videox_fun.pipeline import Wan2_2MultiRefPipeline
from videox_fun.utils.utils import filter_kwargs, save_videos_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Validation script for multiref model")
    
    # Model paths
    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_civitai_5b.yaml", help="Path to config file")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/Wan2.2-TI2V-5B", help="Base model path")
    parser.add_argument("--custom_transformer_path", type=str, default=None, help="Custom transformer checkpoint path")
    parser.add_argument("--custom_transformer_high_path", type=str, default=None, help="Custom high noise transformer checkpoint path")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE checkpoint path")
    
    # Validation dataset
    parser.add_argument("--validation_json", type=str, default="datasets/synworld12/train.json", help="Path to validation json")
    parser.add_argument("--validation_samples", type=int, default=10, help="Number of samples to validate")
    parser.add_argument("--validation_ref_dir", type=str, default=None, help="Directory containing reference videos/images")
    
    # Generation parameters
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=448, help="Video width")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames")
    parser.add_argument("--fps", type=int, default=24, help="FPS for output video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guide_scale_text", type=float, default=5.0, help="Text guidance scale")
    parser.add_argument("--guide_scale_ref", type=float, default=5.0, help="Image guidance scale")
    parser.add_argument("--boundary", type=float, default=0.98, help="Boundary for transformer switching")
    parser.add_argument("--shift", type=int, default=5, help="Shift parameter for scheduler")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/ref_validation", help="Output directory")
    parser.add_argument("--save_comparison", action="store_true", help="Save comparison videos (ref + generated)")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Weight dtype")
    
    # Scheduler
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    
    return parser.parse_args()


def load_reference(ref_path, height, width):
    """Load reference image or video"""
    ref_ext = ref_path.lower().split('.')[-1]
    
    if ref_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
        # Load as image
        ref_image = Image.open(ref_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(min(height, width)),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
        ])
        validation_ref = transform(ref_image).unsqueeze(0)  # [1, C, H, W]
        print(f"Loaded reference image with shape: {validation_ref.shape}")
    else:
        # Load as video
        vr = VideoReader(ref_path)
        max_ref_frames = len(vr)
        ref_frames = vr.get_batch(list(range(max_ref_frames))).asnumpy()
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(min(height, width)),
            transforms.CenterCrop((height, width)),
            transforms.ToTensor(),
        ])
        
        ref_tensor_list = [transform(frame) for frame in ref_frames]
        validation_ref = torch.stack(ref_tensor_list).unsqueeze(0)  # [1, F, C, H, W]
        validation_ref = validation_ref.permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
        print(f"Loaded reference video with shape: {validation_ref.shape}")
    
    return validation_ref


def match_video_size(ref_video, sample_video):
    """Match reference and sample video sizes for comparison"""
    ref_frames = ref_video.shape[2]
    sample_frames = sample_video.shape[2]
    
    # Match frame count
    if ref_frames != sample_frames:
        if sample_frames < ref_frames:
            last_frame = sample_video[:, :, -1:, :, :]
            repeat_count = ref_frames - sample_frames
            repeated_frames = last_frame.repeat(1, 1, repeat_count, 1, 1)
            sample_video = torch.cat([sample_video, repeated_frames], dim=2)
        else:
            last_frame = ref_video[:, :, -1:, :, :]
            repeat_count = sample_frames - ref_frames
            repeated_frames = last_frame.repeat(1, 1, repeat_count, 1, 1)
            ref_video = torch.cat([ref_video, repeated_frames], dim=2)
    
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
    transformer_load_path = args.custom_transformer_path if args.custom_transformer_path is not None else os.path.join(
        args.model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    )
    print(f"Loading transformer from: {transformer_load_path}")
    
    transformer = Wan2_2RefTransformer3DModel.from_pretrained(
        transformer_load_path,
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
        transformer_high_load_path = args.custom_transformer_high_path if args.custom_transformer_high_path is not None else os.path.join(
            args.model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
        )
        print(f"Loading transformer_2 from: {transformer_high_load_path}")
        
        transformer_2 = Wan2_2RefTransformer3DModel.from_pretrained(
            transformer_high_load_path,
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer_2 = None
    
    # Load VAE
    Chosen_AutoencoderKL = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8
    }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
    
    vae = Chosen_AutoencoderKL.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    if args.vae_path is not None:
        print(f"Loading VAE from checkpoint: {args.vae_path}")
        state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        vae.load_state_dict(state_dict, strict=False)
    
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
    
    # Load scheduler
    from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
    from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    
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
    pipeline = Wan2_2MultiRefPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)
    
    # Load validation data
    print(f"Loading validation data from: {args.validation_json}")
    with open(args.validation_json, 'r') as f:
        validation_data = json.load(f)
    
    # Limit samples
    validation_data = validation_data[:args.validation_samples]
    
    # Setup generator
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    
    # Get base directory for reference paths
    json_base_dir = os.path.dirname(args.validation_json)
    
    # Run validation
    print(f"Running validation on {len(validation_data)} samples...")
    
    for idx, sample in enumerate(tqdm(validation_data)):
        try:
            prompt = sample.get('text', sample.get('cap', sample.get('caption', '')))
            ref_relative_path = sample.get('ref', sample.get('ref_path', sample.get('video_path', '')))
            
            # Construct full reference path
            if args.validation_ref_dir is not None:
                # Use custom reference directory
                ref_path = os.path.join(args.validation_ref_dir, os.path.basename(ref_relative_path))
            else:
                # Use path relative to JSON file location
                ref_path = os.path.join(json_base_dir, ref_relative_path)
            
            if not os.path.exists(ref_path):
                print(f"Warning: Reference not found: {ref_path}, skipping...")
                continue
            
            print(f"\n[{idx+1}/{len(validation_data)}] Processing: {os.path.basename(ref_path)}")
            print(f"Prompt: {prompt[:100]}...")
            
            # Load reference
            validation_ref = load_reference(ref_path, args.height, args.width)
            
            # Generate
            with torch.no_grad():
                sample_video = pipeline(
                    prompt,
                    num_frames=args.num_frames,
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    height=args.height,
                    width=args.width,
                    generator=generator,
                    guide_scale_text=args.guide_scale_text,
                    guide_scale_ref=args.guide_scale_ref,
                    num_inference_steps=args.num_inference_steps,
                    boundary=args.boundary,
                    ref_video=validation_ref.to(device=device, dtype=weight_dtype),
                    shift=args.shift,
                ).videos
            
            # Save results
            sample_dir = os.path.join(args.output_dir, f"sample_{idx:04d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save generated video
            if sample_video.shape[2] == 1:
                save_videos_grid(sample_video, os.path.join(sample_dir, "generated.gif"), fps=args.fps)
            else:
                save_videos_grid(sample_video, os.path.join(sample_dir, "generated.mp4"), fps=args.fps)
            
            # Save reference
            if validation_ref.dim() == 4:  # Image
                validation_ref = validation_ref.unsqueeze(2)  # Add frame dim
            save_videos_grid(validation_ref, os.path.join(sample_dir, "reference.gif" if validation_ref.shape[2] == 1 else "reference.mp4"), fps=args.fps)
            
            # Save comparison if requested
            if args.save_comparison:
                ref_matched, sample_matched = match_video_size(validation_ref, sample_video)
                comparison_video = torch.cat([ref_matched, sample_matched], dim=0)
                
                if comparison_video.shape[2] == 1:
                    save_videos_grid(comparison_video, os.path.join(sample_dir, "comparison.gif"), fps=args.fps)
                else:
                    save_videos_grid(comparison_video, os.path.join(sample_dir, "comparison.mp4"), fps=args.fps)
            
            # Save prompt
            with open(os.path.join(sample_dir, "prompt.txt"), 'w') as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Reference: {ref_path}\n")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nValidation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 