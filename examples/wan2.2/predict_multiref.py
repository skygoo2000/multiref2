import os
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from decord import VideoReader

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan3_8, AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer,
                              Wan2_2RefTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import Wan2_2MultiRefPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

# GPU memory mode, which can be chosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
GPU_memory_mode     = "model_full_load"
# Multi GPUs config
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
compile_dit         = False

# TeaCache config
enable_teacache     = True
teacache_threshold  = 0.10
num_skip_start_steps = 5
teacache_offload    = False

# Skip some cfg steps in inference
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.2/wan_civitai_5b.yaml"
model_name          = "models/Diffusion_Transformer/Wan2.2-TI2V-5B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
shift               = 5

# Load pretrained model if need
transformer_path        = None
transformer_high_path   = None
vae_path                = None
# Load lora model if need
lora_path               = None
lora_high_path          = None

# NEW: Custom transformer path
custom_transformer_path = "ckpts/0928_5B_overfit_lr2e-05_ref-t0_beforeconcat_selfattn_neg-rope_256p_nodrop_cfg/checkpoint-10000/transformer"
custom_transformer_high_path = None  # 例如: "path/to/custom/transformer_high"

# Other params
sample_size         = [256, 448]
video_length        = 49
fps                 = 24

# Use torch.float16 if GPU does not support torch.bfloat16
weight_dtype            = torch.bfloat16

# Reference image/video path (NEW: for multiref pipeline)
validation_ref_path     = "/project/liuyuan/zekai/datasets/openvidhd/synworld11k/fg_video/H7z_-9IjXBA_85_23to151_fg.mp4"  # img or video

# Input image for image-to-video (optional, set to None for text-to-video)
validation_image_start  = None  # "asset/1.png"

# prompts
prompt              = "The video shows a white pickup truck parked on a grassy area. The truck is a modern model with a large grille and black wheels. In the background, there is a red pickup truck parked next to the white truck. The scene appears to be set in a rural or semi-rural area, with a building and trees visible in the distance. The sky is partly cloudy, suggesting it might be a cool or overcast day. The style of the video is straightforward and documentary-like, with no visible text or additional graphics. The focus is on the trucks and their immediate surroundings, with no people or other significant activity visible in the frames."
negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
guidance_scale_text = 5.0
guidance_scale_img  = 5.0
seed                = 43
num_inference_steps = 50
lora_weight         = 0.55
lora_high_weight    = 0.55
save_path           = "samples/multiref"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)
boundary = 0.95

# Load transformer (use Wan2_2RefTransformer3DModel instead of Wan2_2Transformer3DModel)
transformer_load_path = custom_transformer_path if custom_transformer_path is not None else os.path.join(
    model_name, config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
)
print(f"Loading transformer from: {transformer_load_path}")

transformer = Wan2_2RefTransformer3DModel.from_pretrained(
    transformer_load_path,
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if config['transformer_additional_kwargs'].get('transformer_combination_type', 'single') == "moe":
    transformer_high_load_path = custom_transformer_high_path if custom_transformer_high_path is not None else os.path.join(
        model_name, config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
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

# Load additional transformer weights if provided (for fine-tuning checkpoints)
if transformer_path is not None:
    print(f"Loading additional weights from checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

if transformer_2 is not None and transformer_high_path is not None:
    print(f"Loading additional weights from checkpoint: {transformer_high_path}")
    if transformer_high_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_high_path)
    else:
        state_dict = torch.load(transformer_high_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer_2.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
Chosen_AutoencoderKL = {
    "AutoencoderKLWan": AutoencoderKLWan,
    "AutoencoderKLWan3_8": AutoencoderKLWan3_8
}[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
vae = Chosen_AutoencoderKL.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

# ... existing code for loading vae weights ...
if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Scheduler
Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Chosen_Scheduler(
    **filter_kwargs(Chosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline (use Wan2_2MultiRefPipeline)
pipeline = Wan2_2MultiRefPipeline(
    transformer=transformer,
    transformer_2=transformer_2,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)

# ... existing code for multi-GPU, FSDP, compile, etc. ...
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if transformer_2 is not None:
        transformer_2.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        if transformer_2 is not None:
            pipeline.transformer_2 = shard_fn(pipeline.transformer_2)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    if transformer_2 is not None:
        for i in range(len(pipeline.transformer_2.blocks)):
            pipeline.transformer_2.blocks[i] = torch.compile(pipeline.transformer_2.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    if transformer_2 is not None:
        replace_parameters_by_name(transformer_2, ["modulation",], device=device)
        transformer_2.freqs = transformer_2.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    if transformer_2 is not None:
        convert_model_weight_to_float8(transformer_2, exclude_module_name=["modulation",], device=device)
        convert_weight_dtype_wrapper(transformer_2, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

# ... existing code for TeaCache, cfg_skip, etc. ...
coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )
    if transformer_2 is not None:
        pipeline.transformer_2.share_teacache(transformer=pipeline.transformer)

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)
    if transformer_2 is not None:
        pipeline.transformer_2.share_cfg_skip(transformer=pipeline.transformer)

generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)
    if transformer_2 is not None:
        pipeline = merge_lora(pipeline, lora_high_path, lora_high_weight, device=device, sub_transformer_name="transformer_2")

with torch.no_grad():
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

    if enable_riflex:
        pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        if transformer_2 is not None:
            pipeline.transformer_2.enable_riflex(k = riflex_k, L_test = latent_frames)

    # Load reference image/video
    validation_ref = None
    if validation_ref_path is not None:
        ref_ext = validation_ref_path.lower().split('.')[-1]
        if ref_ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']:
            # Load as image
            ref_image = Image.open(validation_ref_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize(min(sample_size[0], sample_size[1])),
                transforms.CenterCrop((sample_size[0], sample_size[1])),
                transforms.ToTensor(),
            ])
            validation_ref = transform(ref_image).unsqueeze(0)  # [1, C, H, W]
            print(f"Loaded reference image with shape: {validation_ref.shape}")
        else:
            # Load as video
            vr = VideoReader(validation_ref_path)
            max_ref_frames = len(vr)
            ref_frames = vr.get_batch(list(range(max_ref_frames))).asnumpy()
            
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(min(sample_size[0], sample_size[1])),
                transforms.CenterCrop((sample_size[0], sample_size[1])),
                transforms.ToTensor(),
            ])
            
            ref_tensor_list = [transform(frame) for frame in ref_frames]
            validation_ref = torch.stack(ref_tensor_list).unsqueeze(0)  # [1, F, C, H, W]
            validation_ref = validation_ref.permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]
            print(f"Loaded reference video with shape: {validation_ref.shape}")

    # Load input image for image-to-video if provided
    if validation_image_start is not None:
        input_video, input_video_mask, clip_image = get_image_to_video_latent(
            validation_image_start, None, video_length=video_length, sample_size=sample_size
        )
    else:
        input_video, input_video_mask, clip_image = None, None, None

    # Generate video with reference
    sample = pipeline(
        prompt, 
        num_frames = video_length,
        negative_prompt = negative_prompt,
        height      = sample_size[0],
        width       = sample_size[1],
        generator   = generator,
        guide_scale_text = guidance_scale_text,
        guide_scale_img = guidance_scale_img,
        num_inference_steps = num_inference_steps,
        boundary = boundary,
        video      = input_video,
        mask_video = input_video_mask,
        ref_video  = validation_ref.to(device=device, dtype=weight_dtype) if validation_ref is not None else None,
        shift = shift,
    ).videos

if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)
    if transformer_2 is not None:
        pipeline = unmerge_lora(pipeline, lora_high_path, lora_high_weight, device=device, sub_transformer_name="transformer_2")

def save_results():
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)
    if video_length == 1:
        video_path = os.path.join(save_path, prefix + ".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(video_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)
    
    print(f"Video saved to: {video_path}")

if ulysses_degree * ring_degree > 1:
    import torch.distributed as dist
    if dist.get_rank() == 0:
        save_results()
else:
    save_results() 