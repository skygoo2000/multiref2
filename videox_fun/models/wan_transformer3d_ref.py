# Modified from wan_transformer3d.py

import glob
import json
import math
import os
from typing import Any, Dict, Optional

import torch
import torch.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from .wan_transformer3d import (
    Wan2_2Transformer3DModel,
    WanAttentionBlock,
    WanSelfAttention,
    WanTransformer3DModel,
    sinusoidal_embedding_1d,
)


@amp.autocast('cuda',enabled=False)
@torch.compiler.disable()
def rope_apply_with_ref(x, grid_sizes, ref_grid_sizes, freqs, gap=1):
    """
    Apply RoPE to a tensor x which is a concatenation of a main part and a reference part.
    The reference part is given positional offsets after the main part with a gap of n frames.

    Args:
        x (Tensor): Input tensor, shape [B, L, num_heads, C / num_heads]. (main + ref)
        grid_sizes (Tensor): Shape [B, 3], grid sizes (F, H, W) for the main part.
        ref_grid_sizes (Tensor): Shape [B, 3], grid sizes (rF, rH, rW) for the reference part.
        freqs (Tensor): RoPE frequencies.
        gap (int): Frame gap between main and ref positions.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs for F, H, W dimensions
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (main_grid, ref_grid) in enumerate(zip(grid_sizes.tolist(), ref_grid_sizes.tolist())):
        f, h, w = main_grid
        rf, rh, rw = ref_grid
        
        main_seq_len = f * h * w
        ref_seq_len = rf * rh * rw
        seq_len = ref_seq_len + main_seq_len

        # Main part: positions 0 to f-1
        freqs_main_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(main_seq_len, 1, -1)

        # Ref part: positions f+gap to f+gap+rf-1 (in F dimension)
        freqs_f_ref = freqs[0][f+gap:f+gap+rf].to(device=freqs[0].device, dtype=freqs[0].dtype)
        freqs_h_ref = freqs[1][:rh].to(device=freqs[1].device, dtype=freqs[1].dtype)
        freqs_w_ref = freqs[2][:rw].to(device=freqs[2].device, dtype=freqs[2].dtype)

        freqs_ref_i = torch.cat([
            freqs_f_ref.view(rf, 1, 1, -1).expand(rf, rh, rw, -1),
            freqs_h_ref.view(1, rh, 1, -1).expand(rf, rh, rw, -1),
            freqs_w_ref.view(1, 1, rw, -1).expand(rf, rh, rw, -1)
        ], dim=-1).reshape(ref_seq_len, 1, -1)
        
        # Concatenate: main first, then ref (matching x's sequence order)
        freqs_i = torch.cat([freqs_main_i, freqs_ref_i], dim=0)

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float32).reshape(
            seq_len, n, -1, 2))
        
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        
        if seq_len < x.size(1):
            x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
        
    return torch.stack(output).to(x.dtype)


def rope_apply_qk_with_ref(q, k, grid_sizes, ref_grid_sizes, freqs):
    q = rope_apply_with_ref(q, grid_sizes, ref_grid_sizes, freqs)
    k = rope_apply_with_ref(k, grid_sizes, ref_grid_sizes, freqs)
    return q, k


class WanSelfAttentionWithRef(WanSelfAttention):
    """
    WanSelfAttention with support for reference frames using offset RoPE positions.
    """
    
    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0, ref_grid_sizes=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads] - concatenated (main + ref)
            seq_lens(Tensor): Shape [B] - total sequence lengths including ref
            grid_sizes(Tensor): Shape [B, 3], grid sizes (F, H, W) for the main part
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ref_grid_sizes(Tensor): Shape [B, 3], grid sizes (rF, rH, rW) for the reference part
        """
        from .attention_utils import attention
        from .wan_transformer3d import rope_apply_qk
        
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
            k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
            v = self.v(x.to(dtype)).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Apply RoPE with reference support if ref_grid_sizes is provided
        if ref_grid_sizes is not None:
            q, k = rope_apply_qk_with_ref(q, k, grid_sizes, ref_grid_sizes, freqs)
        else:
            q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v=v.to(dtype),
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.to(dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlockWithRef(WanAttentionBlock):
    """
    WanAttentionBlock with support for reference frames.
    """
    
    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        
        # Replace self_attn with the ref-aware version
        self.self_attn = WanSelfAttentionWithRef(dim, num_heads, window_size, qk_norm, eps)

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
        ref_grid_sizes=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            ref_grid_sizes(Tensor): Shape [B, 3], grid sizes (rF, rH, rW) for the reference part
        """
        if e.dim() > 3:
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            e = [e.squeeze(2) for e in e]
        else:        
            e = (self.modulation + e).chunk(6, dim=1)

        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype, t=t, ref_grid_sizes=ref_grid_sizes)
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


class Wan2_1RefTransformer3DModel(WanTransformer3DModel):
    r"""
    Wan 2.1 diffusion backbone with multi-frame reference support.
    Extends WanTransformer3DModel to support subject_ref with RoPE-based positioning.
    Uses subject_ref naming convention to match WanTransformer3DModel interface.
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
    ):
        # Call parent's __init__
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
        )
        
        # Replace blocks with ref-aware versions
        cross_attn_type = cross_attn_type or ('t2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn')
        self.blocks = nn.ModuleList([
            WanAttentionBlockWithRef(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers
    
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
    ):
        r"""
        Forward pass with multi-frame reference support.
        
        Args:
            subject_ref (Tensor, *optional*):
                Reference video/image tensor. Can be:
                - [B, C, H, W] for single frame reference
                - [B, C, F, H, W] for multi-frame reference
                Uses the same patch_embedding as main input x.
        """
        # Handle multi-frame subject_ref with RoPE positioning
        ref_frames = 0
        ref_grid_sizes = None
        
        device = self.patch_embedding.weight.device if hasattr(self.patch_embedding, 'weight') else next(self.parameters()).device
        dtype = x[0].dtype if isinstance(x, list) else x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings for main input
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        # Get original grid sizes for main content
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]).to(device)
        
        # Store original grid sizes for main content (without ref)
        x_grid_sizes = grid_sizes.clone()

        x = [u.flatten(2).transpose(1, 2) for u in x]
        
        # Handle full_ref (for compatibility, but not implemented)
        if full_ref is not None:
            if self.ref_conv is not None:
                full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
                grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
                seq_len += full_ref.size(1)
                x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]
                if t.dim() != 1 and t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([padding, t], dim=1)
        
        # Process subject_ref if provided
        if subject_ref is not None:
            if subject_ref.dim() == 4:
                # Single frame case: [B, C, H, W] -> [B, C, 1, H, W]
                subject_ref = subject_ref.unsqueeze(2)
                ref_frames = 1
            else:
                # Multi-frame case: [B, C, F, H, W]
                ref_frames = subject_ref.size(2)
            
            # Calculate ref_grid_sizes from subject_ref shape
            ref_grid_sizes = torch.stack([
                torch.tensor([ref_frames, x_grid_sizes[i][1], x_grid_sizes[i][2]], dtype=torch.long) 
                for i in range(len(x))
            ]).to(device=device, dtype=torch.long)
            
            subject_ref = self.patch_embedding(subject_ref).flatten(2).transpose(1, 2)  # [B, tokens, dim]
            
            # Update seq_len to include subject_ref tokens
            seq_len += subject_ref.size(1)
            
            # Concatenate subject_ref tokens to the end of each sequence
            x = [torch.concat([u, _subject_ref.unsqueeze(0)], dim=1) for u, _subject_ref in zip(x, subject_ref)]
            
            ### Use t=0 for subject_ref tokens ####
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                # Use t=0 for subject_ref tokens, append to the end
                padding_zero = t.new_zeros(t.size(0), pad_size)
                t = torch.cat([t, padding_zero], dim=1)
            
            # ### Use noisy subject_ref tokens ####
            # if t.dim() != 1 and t.size(1) < seq_len:
            #     pad_size = seq_len - t.size(1)
            #     last_elements = t[:, -1].unsqueeze(1)
            #     padding = last_elements.repeat(1, pad_size)
            #     t = torch.cat([t, padding], dim=1)
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
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

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

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
                            x_grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            ref_grid_sizes,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=x_grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t,
                            ref_grid_sizes=ref_grid_sizes
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
                        x_grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                        ref_grid_sizes,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=x_grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        t=t,
                        ref_grid_sizes=ref_grid_sizes
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

        # Remove subject_ref tokens from output
        if subject_ref is not None:
            subject_ref_length = subject_ref.size(1)
            x = x[:, :-subject_ref_length]  # Remove from end since subject_ref is concatenated at the end
            # Use x_grid_sizes for unpatchify
            grid_sizes = x_grid_sizes

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x


class Wan2_2RefTransformer3DModel(Wan2_2Transformer3DModel):
    r"""
    Wan 2.2 diffusion backbone with multi-frame reference support.
    Uses its own patch_embedding that accepts in_dim + 16 channels (16 extra channels for mask).
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
        mask_channels=16,  # Number of mask channels to add
        cross_attn_type=None,
    ):
        # Store mask_channels before calling parent
        self.mask_channels = mask_channels
        
        # Call parent's __init__
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
        )

        if hasattr(self, "img_emb"):
            del self.img_emb
        
        # Replace patch_embedding to accept in_dim + mask_channels
        self.patch_embedding = nn.Conv3d(
            in_dim + mask_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # Replace blocks with ref-aware versions
        cross_attn_type = "cross_attn"
        self.blocks = nn.ModuleList([
            WanAttentionBlockWithRef(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for layer_idx, block in enumerate(self.blocks):
            block.self_attn.layer_idx = layer_idx
            block.self_attn.num_layers = self.num_layers
        
        # Initialize the new patch_embedding
        self._init_patch_embedding()
    
    def _init_patch_embedding(self):
        """Initialize patch_embedding with Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
    
    @classmethod
    def from_pretrained(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        low_cpu_mem_usage=False, torch_dtype=torch.bfloat16
    ):
        """
        Override from_pretrained to handle patch_embedding dimension mismatch.
        If checkpoint has in_dim channels, we reinitialize patch_embedding.
        If checkpoint has in_dim + mask_channels, we load it directly.
        """
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded Wan2_2RefTransformer3DModel from {pretrained_model_path} ...")

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

        # Create model instance
        model = cls.from_config(config, **transformer_additional_kwargs)
        
        # Load state dict
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_files_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            print(model_files_safetensors)
            for _model_file_safetensors in model_files_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        # Check patch_embedding dimensions
        if 'patch_embedding.weight' in state_dict:
            ckpt_patch_embed_shape = state_dict['patch_embedding.weight'].shape
            model_patch_embed_shape = model.patch_embedding.weight.shape
            
            # ckpt_patch_embed_shape[1] is input channels
            if ckpt_patch_embed_shape[1] != model_patch_embed_shape[1]:
                print(f"Patch embedding input channels mismatch: checkpoint has {ckpt_patch_embed_shape[1]}, "
                      f"model expects {model_patch_embed_shape[1]}.")
                
                # Initialize patch_embedding to zeros
                new_patch_weight = torch.zeros(model_patch_embed_shape, dtype=state_dict['patch_embedding.weight'].dtype)
                
                # Load checkpoint weights to first 48 channels (or min of checkpoint and model channels)
                num_channels_to_copy = min(ckpt_patch_embed_shape[1], 48)
                new_patch_weight[:, :num_channels_to_copy, ...] = state_dict['patch_embedding.weight'][:, :num_channels_to_copy, ...]
                
                print(f"Loaded first {num_channels_to_copy} channels from checkpoint to patch_embedding, "
                      f"remaining channels initialized to zeros.")
                
                # Update state_dict with the new patch_embedding weight
                state_dict['patch_embedding.weight'] = new_patch_weight
                
                if 'patch_embedding.bias' in state_dict:
                    pass
            else:
                pass
        
        # Filter state_dict to only include matching keys
        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(f"{key}: Size don't match or not in model, skip")
                
        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(f"Missing keys: {m}")
        
        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")
        
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
    ):
        r"""
        Forward pass with multi-frame reference support.
        
        Args:
            full_ref (Tensor, *optional*):
                Reference video/image tensor. Can be:
                - [B, C, H, W] for single frame reference
                - [B, C, F, H, W] for multi-frame reference
                Uses the same patch_embedding as main input x.
        """
        # Handle multi-frame full_ref with mask channel
        ref_frames = 0
        ref_grid_sizes = None
        
        device = self.patch_embedding.weight.device if hasattr(self.patch_embedding, 'weight') else next(self.parameters()).device
        dtype = x[0].dtype if isinstance(x, list) else x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Process main input x: add mask channel of 0s
        x_with_mask = []
        for u in x:
            # u shape: [C, F, H, W]
            mask = torch.zeros(self.mask_channels, u.size(1), u.size(2), u.size(3), 
                             device=u.device, dtype=u.dtype)
            u_with_mask = torch.cat([u, mask], dim=0)  # [C+mask_channels, F, H, W]
            x_with_mask.append(u_with_mask)
        
        # embeddings for main input
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x_with_mask]
        
        # Process full_ref if provided: add mask channel of 1s
        if full_ref is not None:
            if full_ref.dim() == 4:
                # Single frame case: [B, C, H, W] -> [B, C, 1, H, W]
                full_ref = full_ref.unsqueeze(2)
                ref_frames = 1
            else:
                # Multi-frame case: [B, C, F, H, W]
                ref_frames = full_ref.size(2)
            
            # Add mask channel of 1s to full_ref
            # full_ref shape: [B, C, F, H, W]
            ref_mask = torch.ones(full_ref.size(0), self.mask_channels, full_ref.size(2), 
                                 full_ref.size(3), full_ref.size(4), 
                                 device=full_ref.device, dtype=full_ref.dtype)
            full_ref = torch.cat([full_ref, ref_mask], dim=1)  # [B, C+mask_channels, F, H, W]
            
            # Apply patch_embedding to full_ref
            full_ref = self.patch_embedding(full_ref)
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        # Get original grid sizes for main content
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]).to(device)
        
        # Store original grid sizes for main content (without ref)
        x_grid_sizes = grid_sizes.clone()

        x = [u.flatten(2).transpose(1, 2) for u in x]
        
        # concatenate full_ref to x sequences
        if full_ref is not None:
            full_ref = full_ref.flatten(2).transpose(1, 2)  # [B, tokens, dim]
            
            # Calculate ref_grid_sizes from full_ref shape
            ref_grid_sizes = torch.stack([
                torch.tensor([ref_frames, x_grid_sizes[i][1], x_grid_sizes[i][2]], dtype=torch.long) 
                for i in range(len(x))
            ]).to(device=device, dtype=torch.long)
            
            # Update grid_sizes to account for ref frames (total frames = main + ref)
            grid_sizes = torch.stack([torch.tensor([u[0] + ref_frames, u[1], u[2]]) for u in grid_sizes]).to(device=device, dtype=torch.long)
            seq_len += full_ref.size(1)
            # Concatenate ref tokens to the end of each sequence
            x = [torch.concat([u, _full_ref.unsqueeze(0)], dim=1) for u, _full_ref in zip(x, full_ref)]
            
            ### Use t=0 for full_ref tokens instead of last_elements ####
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                # Use t=0 for full_ref tokens, append to the end
                zero_timesteps = t.new_zeros(t.size(0), pad_size)
                t = torch.cat([t, zero_timesteps], dim=1)

            # ### Use noisy t for full_ref tokens instead of last_elements ####
            # if t.dim() != 1 and t.size(1) < seq_len:
            #     pad_size = seq_len - t.size(1)
            #     last_elements = t[:, -1]
            #     padding = last_elements.unsqueeze(1).repeat(1, pad_size)
            #     t = torch.cat([t, padding], dim=1)
        
        if subject_ref is not None:
            assert False, "multiref2 do not support subject_ref"
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
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

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

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
                            x_grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            dtype,
                            t,
                            ref_grid_sizes,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=x_grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            t=t,
                            ref_grid_sizes=ref_grid_sizes
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
                        x_grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        dtype,
                        t,
                        ref_grid_sizes,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=x_grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        t=t,
                        ref_grid_sizes=ref_grid_sizes
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

        # Remove ref tokens from output
        if full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, :-full_ref_length]  # Remove from end since full_ref is concatenated at the end
            # Use x_grid_sizes for unpatchify
            grid_sizes = x_grid_sizes

        if subject_ref is not None:
            assert False, "multiref2 do not support subject_ref"

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
