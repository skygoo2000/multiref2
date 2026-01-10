# Modified from wan_transformer3d.py

import glob
import json
import math
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from .wan_transformer3d import (
    WanTransformer3DModel,
    WanRMSNorm,
    WanLayerNorm,
    WanSelfAttention,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    rope_apply_qk,
    sinusoidal_embedding_1d,
)


class PosWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        
        # pose encoder and projector (initialized to zero and identity)
        # Input is 16 (4x4 homogeneous coordinates flattened)
        self.pos_emb = nn.Linear(16, dim)
        self.projector = nn.Linear(dim, dim)
        # Zero initialization for pos_emb
        self.pos_emb.weight.data.zero_()
        self.pos_emb.bias.data.zero_()
        # Identity initialization for projector
        self.projector.weight = nn.Parameter(torch.eye(dim))
        self.projector.bias = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        dtype=torch.bfloat16,
        t=0,
        pose_emb=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            pose_emb(Tensor, optional): Shape [B, L, 16], pose embedding for each token (4x4 homogeneous coordinates flattened)
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:        
            e = (self.modulation + e).chunk(6, dim=1)

        # self-attention with pose embedding
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        
        # Add pose embedding before self-attention
        if pose_emb is not None:
            pose_encoded = self.pos_emb(pose_emb)
            temp_x = temp_x + pose_encoded

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t)
        
        # Apply projector after self-attention
        y = self.projector(y)
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            # cross-attention
            x = x + self.cross_attn(self.norm3(x), context, context_lens, dtype, t=t)

            # ffn function
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class MultiRefTransformer3DModel(WanTransformer3DModel):
    r"""
    Multi-reference Transformer3D model inherited from WanTransformer3DModel.
    """

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=16,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
        use_pose_embedding=True,
    ):
        r"""
        Initialize the multi-reference diffusion model backbone.
        
        Args:
            use_pose_embedding (bool): Whether to use pose embedding (pos_emb and projector)
        """
        self.use_pose_embedding = use_pose_embedding
        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=add_ref_conv,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type if cross_attn_type is not None else "cross_attn",
        )

        # Replace blocks with our own PosWanAttentionBlock
        if cross_attn_type is None:
            cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'cross_attn'
        
        self.blocks = nn.ModuleList([
            PosWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers

        if hasattr(self, "img_emb"):
            del self.img_emb

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded MultiRefTransformer3DModel from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['patch_embedding.weight'].size() != state_dict['patch_embedding.weight'].size():
            model.state_dict()['patch_embedding.weight'][:, :state_dict['patch_embedding.weight'].size()[1], :, :] = state_dict['patch_embedding.weight'][:, :model.state_dict()['patch_embedding.weight'].size()[1], :, :]
            model.state_dict()['patch_embedding.weight'][:, state_dict['patch_embedding.weight'].size()[1]:, :, :] = 0
            state_dict['patch_embedding.weight'] = model.state_dict()['patch_embedding.weight']
        
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        # Check if pos_emb and projector are missing, if so, initialize them to zero/identity
        missing_pos_emb = any('pos_emb' in key for key in m)
        missing_projector = any('projector' in key for key in m)
        
        if missing_pos_emb or missing_projector:
            print("### Initializing missing pos_emb and/or projector with zero/identity initialization")
            for block in model.blocks:
                if missing_pos_emb:
                    block.pos_emb.weight.data.zero_()
                    block.pos_emb.bias.data.zero_()
                if missing_projector:
                    block.projector.weight = nn.Parameter(torch.eye(block.dim))
                    block.projector.bias = nn.Parameter(torch.zeros(block.dim))
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        ref_pose=None,
        video_pose=None,
    ):
        r"""
        Forward pass through the multi-reference diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            y_camera (Tensor, *optional*):
                Camera control adapter input
            full_ref (Tensor, *optional*):
                Full reference frames
            subject_ref (Tensor, *optional*):
                Subject reference frames
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input
            ref_pose (Tensor, *optional*):
                Reference pose information, shape [B, F, 4, 4]
            video_pose (Tensor, *optional*):
                Video pose information, shape [B, F, 4, 4]

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # params
        device = self.patch_embedding.weight.device
        dtype = x.dtype if not isinstance(x, list) else x[0].dtype
        
        # Store original grid sizes for pose processing
        original_grid_sizes = None
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        
        # Store original grid sizes (before adding ref frames) for pose processing
        original_grid_sizes = grid_sizes.clone()

        x = [u.flatten(2).transpose(1, 2) for u in x]
        if self.ref_conv is not None and full_ref is not None:
            if full_ref.dim() > 4:
                # full_ref shape: [B, 16, F, H, W]
                full_ref_frames_num = full_ref.size(2)
                full_ref_list = []
                for f in range(full_ref_frames_num):
                    frame = full_ref[:, :, f, :, :]  # [B, 16, H, W]
                    frame_conv = self.ref_conv(frame)
                    full_ref_list.append(frame_conv)  # [B, model_dim(1536), H/2, H/2]
                full_ref = torch.stack(full_ref_list, dim=2)
                full_ref = full_ref.flatten(2).transpose(1, 2)  # [B, ref_seq_len, model_dim(1536)]
                grid_sizes = torch.stack([torch.tensor([u[0] + full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            else:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
                full_ref_frames_num = None
            seq_len += full_ref.size(1)

            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([padding, t], dim=1)

        # Track subject_ref frames for pose validation
        subject_ref_frames = 0
        if subject_ref is not None:
            subject_ref_frames = subject_ref.size(2)
            subject_ref = self.patch_embedding(subject_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += subject_ref.size(1)
            x = [torch.concat([u, _subject_ref.unsqueeze(0)], dim=1) for _subject_ref, u in zip(subject_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        
        # Process pose embeddings
        # Correspondence: video_pose -> x (main video), ref_pose -> subject_ref
        # Concatenation order should match: [x, subject_ref] -> [video_pose, ref_pose]
        pose_emb = None
        if ref_pose is not None or video_pose is not None:
            batch_size = x.size(0)
            h_tokens = original_grid_sizes[0][1].item()
            w_tokens = original_grid_sizes[0][2].item()
            tokens_per_frame = h_tokens * w_tokens
            
            # Get actual frame counts from original grid sizes and subject_ref
            x_frames = original_grid_sizes[0][0].item()  # x (main video) frames
            
            pose_list = []
            
            # Process video_pose first (corresponds to x, which comes first in concatenation)
            if video_pose is not None:
                video_pose_frames = video_pose.size(1)
                
                # Downsample video_pose to match x_frames (VAE compressed frames)
                # Poses are sampled from original video frames, but latents are VAE-compressed
                if video_pose_frames != x_frames:
                    # Use linspace to uniformly sample poses to match latent frames
                    indices = torch.linspace(0, video_pose_frames - 1, x_frames, device=video_pose.device).long()
                    video_pose = video_pose[:, indices, :, :]  # [B, x_frames, 4, 4]
                    video_pose_frames = x_frames
                
                # video_pose: [B, F, 4, 4] -> [B, F, 16]
                video_pose_flat = video_pose.reshape(batch_size, video_pose_frames, 16)
                # Broadcast to all tokens in each frame: [B, F, 16] -> [B, F*H*W, 16]
                video_pose_broadcast = video_pose_flat.unsqueeze(2).expand(batch_size, video_pose_frames, tokens_per_frame, 16)
                video_pose_broadcast = video_pose_broadcast.reshape(batch_size, video_pose_frames * tokens_per_frame, 16)
                pose_list.append(video_pose_broadcast)
            
            # Process ref_pose second (corresponds to subject_ref, which comes second in concatenation)
            if ref_pose is not None:
                ref_pose_frames = ref_pose.size(1)
                
                # Downsample ref_pose to match subject_ref_frames (VAE compressed frames)
                # Poses are sampled from original video frames, but latents are VAE-compressed
                if ref_pose_frames != subject_ref_frames:
                    # Use linspace to uniformly sample poses to match latent frames
                    indices = torch.linspace(0, ref_pose_frames - 1, subject_ref_frames, device=ref_pose.device).long()
                    ref_pose = ref_pose[:, indices, :, :]  # [B, subject_ref_frames, 4, 4]
                    ref_pose_frames = subject_ref_frames
                
                # ref_pose: [B, F, 4, 4] -> [B, F, 16]
                ref_pose_flat = ref_pose.reshape(batch_size, ref_pose_frames, 16)
                # Broadcast to all tokens in each frame: [B, F, 16] -> [B, F*H*W, 16]
                ref_pose_broadcast = ref_pose_flat.unsqueeze(2).expand(batch_size, ref_pose_frames, tokens_per_frame, 16)
                ref_pose_broadcast = ref_pose_broadcast.reshape(batch_size, ref_pose_frames * tokens_per_frame, 16)
                pose_list.append(ref_pose_broadcast)
            
            # Concatenate video and ref poses in correct order: [video_pose, ref_pose]
            if len(pose_list) > 0:
                pose_emb = torch.cat(pose_list, dim=1)  # [B, total_tokens, 16]
                
                # Pad to seq_len if necessary
                if pose_emb.size(1) < seq_len:
                    padding = torch.zeros(batch_size, seq_len - pose_emb.size(1), 16, 
                                        device=pose_emb.device, dtype=pose_emb.dtype)
                    pose_emb = torch.cat([pose_emb, padding], dim=1)

        # time embeddings
        with amp.autocast('cuda',dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
            if pose_emb is not None:
                pose_emb = torch.chunk(pose_emb, self.sp_world_size, dim=1)[self.sp_world_rank]
        
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t,
                            pose_emb=pose_emb
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            pose_emb,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t,
                            pose_emb=pose_emb
                        )
                        x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_token_num = full_ref.size(1)
            x = x[:, full_ref_token_num:]

            if full_ref_frames_num is not None:
                grid_sizes = torch.stack([torch.tensor([u[0] - full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            else:
                grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if subject_ref is not None:
            subject_ref_length = subject_ref.size(1)
            x = x[:, :-subject_ref_length]
            grid_sizes = torch.stack([torch.tensor([u[0] - subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


class CroodRefTransformer3DModel(WanTransformer3DModel):
    r"""
    CroodRef Transformer3D model inherited from WanTransformer3DModel.
    """

    @register_to_config
    def __init__(
        self,
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=48,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        in_channels=48,
        hidden_size=2048,
        add_control_adapter=False,
        in_dim_control_adapter=24,
        downscale_factor_control_adapter=8,
        add_ref_conv=False,
        in_dim_ref_conv=16,
        cross_attn_type=None,
    ):
        r"""
        Initialize the CroodRef diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 48):
                Input video channels (C_in) - 48 for CroodRef (16 ref + 32 coordmap)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__(
            model_type=model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            in_channels=in_channels,
            hidden_size=hidden_size,
            add_control_adapter=False,
            in_dim_control_adapter=in_dim_control_adapter,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            add_ref_conv=False,
            in_dim_ref_conv=in_dim_ref_conv,
            cross_attn_type=cross_attn_type,  # Let parent class decide based on model_type
        )

        # Remove components not used in CroodRef
        if hasattr(self, "control_adapter"):
            del self.control_adapter
        if hasattr(self, "ref_conv"):
            del self.ref_conv
        
        # Save the original patch_embedding weights (16 channels from parent class)
        original_patch_weight = self.patch_embedding.weight.data.clone()
        original_patch_bias = self.patch_embedding.bias.data.clone() if self.patch_embedding.bias is not None else None
        
        # Force recreate patch_embedding with 48 channels for CroodRef
        # This ensures patch_embedding always uses 48 channels regardless of loaded config
        self.patch_embedding = nn.Conv3d(
            48,  # Always 48 channels for CroodRef (16 ref + 32 coordmap)
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )
        
        with torch.no_grad():
            original_channels = original_patch_weight.size(1)
            self.patch_embedding.weight[:, :original_channels, :, :, :] = original_patch_weight
            self.patch_embedding.weight[:, original_channels:, :, :, :] = 0
            if original_patch_bias is not None:
                self.patch_embedding.bias.copy_(original_patch_bias)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded CroodRefTransformer3DModel from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".safetensors")

        if "dict_mapping" in transformer_additional_kwargs.keys():
            for key in transformer_additional_kwargs["dict_mapping"]:
                transformer_additional_kwargs[transformer_additional_kwargs["dict_mapping"][key]] = config[key]

        if low_cpu_mem_usage:
            try:
                import re

                from diffusers import __version__ as diffusers_version
                from diffusers.models.modeling_utils import \
                    load_model_dict_into_meta
                from diffusers.utils import is_accelerate_available
                if is_accelerate_available():
                    import accelerate
                
                # Instantiate model with empty weights
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **transformer_additional_kwargs)

                param_device = "cpu"
                if os.path.exists(model_file):
                    state_dict = torch.load(model_file, map_location="cpu")
                elif os.path.exists(model_file_safetensors):
                    from safetensors.torch import load_file, safe_open
                    state_dict = load_file(model_file_safetensors)
                else:
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
                    state_dict = {}
                    print(model_files_safetensors)
                    for _model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(_model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                if diffusers_version >= "0.33.0":
                    # Diffusers has refactored `load_model_dict_into_meta` since version 0.33.0 in this commit:
                    # https://github.com/huggingface/diffusers/commit/f5929e03060d56063ff34b25a8308833bec7c785.
                    load_model_dict_into_meta(
                        model,
                        state_dict,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )
                else:
                    model._convert_deprecated_attention_blocks(state_dict)
                    # move the params from meta device to cpu
                    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_path} because the following keys are"
                            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                            " those weights or else make sure your checkpoint file is correct."
                        )

                    unexpected_keys = load_model_dict_into_meta(
                        model,
                        state_dict,
                        device=param_device,
                        dtype=torch_dtype,
                        model_name_or_path=pretrained_model_path,
                    )

                    if cls._keys_to_ignore_on_load_unexpected is not None:
                        for pat in cls._keys_to_ignore_on_load_unexpected:
                            unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                    if len(unexpected_keys) > 0:
                        print(
                            f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                        )
                
                return model
            except Exception as e:
                print(
                    f"The low_cpu_mem_usage mode is not work because {e}. Use low_cpu_mem_usage=False instead."
                )
        
        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file, safe_open
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if 'patch_embedding.weight' in state_dict:
            pretrained_channels = state_dict['patch_embedding.weight'].size(1)
            model_channels = model.state_dict()['patch_embedding.weight'].size(1)
            
            if pretrained_channels != model_channels:
                print(f"### Expanding patch_embedding from {pretrained_channels} to {model_channels} channels")
                new_weight = torch.zeros_like(model.state_dict()['patch_embedding.weight'])
                new_weight[:, :pretrained_channels, :, :, :] = state_dict['patch_embedding.weight']
                state_dict['patch_embedding.weight'] = new_weight
                print(f"### First {pretrained_channels} channels copied from pretrained, remaining {model_channels - pretrained_channels} channels zero-initialized")
            elif pretrained_channels == model_channels == 48:
                # Both are 48 channels, load directly
                print(f"### Loading patch_embedding with {model_channels} channels (no expansion needed)")
        else:
            raise ValueError(f"patch_embedding.weight not found in checkpoint")
        
        tmp_state_dict = {} 
        for key in state_dict:
            # Skip ref_conv and control_adapter for CroodRef
            if key.startswith('ref_conv.') or key.startswith('control_adapter.'):
                print(key, "Skipping ref_conv/control_adapter for CroodRef")
                continue
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        params = [p.numel() if "attn1." in n else 0 for n, p in model.named_parameters()]
        print(f"### attn1 Parameters: {sum(params) / 1e6} M")
        
        model = model.to(torch_dtype)
        return model

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        full_ref=None,
        full_ref_crood=None,
        cond_flag=True,
    ):
        r"""
        Forward pass through the CroodRef diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs, 32 channels
            full_ref (Tensor, *optional*):
                Full reference frames, 16 channels
            full_ref_crood (Tensor, *optional*):
                Coordinate map for full reference frames, 16 channels
            cond_flag (`bool`, *optional*, defaults to True):
                Flag to indicate whether to forward the condition input

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        device = self.patch_embedding.weight.device
        dtype = x.dtype if not isinstance(x, list) else x[0].dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        full_ref_token_num = 0
        full_ref_frames_num = 0
        if full_ref is not None:
            if full_ref_crood is not None:
                # full_ref_crood is 16 channels, concat with zeros to make 32
                full_ref_crood = torch.cat([full_ref_crood, full_ref], dim=1)
            else:
                full_ref_crood = torch.cat([torch.zeros_like(full_ref), full_ref], dim=1)
            
            full_ref_combined = torch.cat([full_ref, full_ref_crood], dim=1)  # [B, 48, F, H, W] or [B, 48, H, W]
            
            full_ref_embedded = self.patch_embedding(full_ref_combined).flatten(2).transpose(1, 2)  # [B, seq_len, model_dim]
            
            if full_ref_combined.dim() > 4:
                full_ref_frames_num = full_ref_combined.size(2)
            else:
                full_ref_frames_num = 1
            
            full_ref_token_num = full_ref_embedded.size(1)
            seq_len += full_ref_token_num
        
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]).to(device)

        x = [u.flatten(2).transpose(1, 2) for u in x] # [B, seq_len, model_dim]
        
        # Concat full_ref in front of x
        if full_ref is not None:
            grid_sizes = torch.stack([torch.tensor([u[0] + full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref_embedded, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([padding, t], dim=1)
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long).to(device)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast('cuda',dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            # e0 = e0.to(dtype)
            # e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None and hasattr(self, 'img_emb'):
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
        
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t  
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        t=t  
                    )
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        # Remove full_ref tokens from output (use saved full_ref_token_num and full_ref_frames_num)
        if full_ref_token_num > 0:
            x = x[:, full_ref_token_num:]
            grid_sizes = torch.stack([torch.tensor([u[0] - full_ref_frames_num, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
