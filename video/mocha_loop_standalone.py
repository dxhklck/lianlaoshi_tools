import torch
from comfy import model_management as mm
import gc

def rope_params_mocha(max_seq_len, dim, theta=10000, L_test=25, k=0, start=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(start, max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.autocast(device_type=mm.get_autocast_device(mm.get_torch_device()), enabled=False)
def rope_apply_mocha(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    split_size_1 = c - 2 * (c // 3)
    split_size_2 = c // 3
    split_size_3 = c // 3
    freqs_0, freqs_1, freqs_2 = freqs.split([split_size_1, split_size_2, split_size_3], dim=1)

    batch_size = grid_sizes.size(0)
    outputs = []
    
    for i in range(batch_size):
        f, h, w = grid_sizes[i].tolist()
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].reshape(seq_len, n, -1, 2))

        sf = (f - 2) // 2

        freq0_start, freq0_end = 1, 1 + sf
        freq1_start, freq1_end = 1, 1 + h
        freq2_start, freq2_end = 1, 1 + w
        
        repeat_freqs = torch.cat([
            freqs_0[freq0_start:freq0_end].view(sf, 1, 1, -1).expand(sf, h, w, -1),
            freqs_1[freq1_start:freq1_end].view(1, h, 1, -1).expand(sf, h, w, -1),
            freqs_2[freq2_start:freq2_end].view(1, 1, w, -1).expand(sf, h, w, -1)
        ], dim=-1)

        mask_freqs = torch.cat([
            freqs_0[1:2].view(1, 1, 1, -1).expand(1, h, w, -1),
            freqs_1[freq1_start:freq1_end].view(1, h, 1, -1).expand(1, h, w, -1),
            freqs_2[freq2_start:freq2_end].view(1, 1, w, -1).expand(1, h, w, -1)
        ], dim=-1)

        img_freqs = torch.cat([
            freqs_0[0:1].view(1, 1, 1, -1).expand(1, h, w, -1),
            freqs_1[freq1_start:freq1_end].view(1, h, 1, -1).expand(1, h, w, -1),
            freqs_2[freq2_start:freq2_end].view(1, 1, w, -1).expand(1, h, w, -1)
        ], dim=-1)
    
        if f == 2 * sf + 2:
            freqs_i = torch.cat([repeat_freqs, repeat_freqs, mask_freqs, img_freqs], dim=0)
        else:
            bias_freqs = torch.cat([
                freqs_0[0:1].view(1, 1, 1, -1).expand(1, h, w, -1),
                freqs_1[(h+1):(2*h+1)].view(1, h, 1, -1).expand(1, h, w, -1),
                freqs_2[(w+1):(2*w+1)].view(1, 1, w, -1).expand(1, h, w, -1)
            ], dim=-1)
            freqs_i = torch.cat([repeat_freqs, repeat_freqs, mask_freqs, img_freqs, bias_freqs], dim=0)
        
        freqs_i = freqs_i.reshape(f * h * w, 1, -1).to(x.device)
        
        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]], dim=0)

        outputs.append(x_i)
    
    return torch.stack(outputs).to(x.dtype)


class MochaLoopEmbedsStandalone:
    """
    独立的MochaLoopEmbeds第三方节点
    支持长视频循环编码，包含上下文窗口选项
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("WANVAE", {"tooltip": "WANVAE 编码器（与模型匹配）"}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "编码结束后将 VAE 卸载到低显存设备"}),
                "input_video": ("IMAGE", {"tooltip": "输入视频（像素帧，RGBA时仅取RGB）"}),
                "mask": ("MASK", {"tooltip": "单帧掩码（与人物区域一致）"}),
                "ref1": ("IMAGE", {"tooltip": "参考图像 1（维持主体与色彩）"}),
                "chunk_frames": ("INT", {"default": 81, "min": 2, "max": 10000, "step": 1, "tooltip": "每批采样的像素帧数（建议 4n+1）"}),
            },
            "optional": {
                "ref2": ("IMAGE", {"tooltip": "参考图像 2（可选）"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "瓦片编码（显著降低显存占用）"}),
                "context_stride": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1, "tooltip": "像素步长（自动对齐为 4 的倍数）"}),
                "context_overlap": ("INT", {"default": 24, "min": 1, "max": 100, "step": 1, "tooltip": "像素重叠（越大边界更平滑；自动对齐为 4 的倍数）"}),
                "closed_loop": ("BOOLEAN", {"default": False, "tooltip": "循环遍历窗口（长视频闭环）"}),
                "freenoise": ("BOOLEAN", {"default": False, "tooltip": "打乱噪声提升局部多样性（一致性优先时建议关闭）"}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "输出窗口与融合的详细日志"}),
                "fuse_method": (["linear", "pyramid"], {"default": "pyramid", "tooltip": "窗口融合权重：pyramid 边界更平滑，linear 边界更锐"}),
                "use_first_frame_as_reference": ("BOOLEAN", {"default": True, "tooltip": "将有效首帧 latent 作为参考用于非0起始窗口"}),
                "pad_first_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, "tooltip": "首帧预热复制帧数（按 4 对齐；生成后自动裁掉）"}),
                "force_first_window_zero": ("BOOLEAN", {"default": True, "tooltip": "首窗口固定从第 0 帧开始（更稳）"}),
            }
        }
    
    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", "WANVIDCONTEXT")
    RETURN_NAMES = ("image_embeds", "context_options")

    FUNCTION = "process"
    CATEGORY = "lianlaoshi_tools/video"
    DESCRIPTION = "MoCha long video looped embeds with context options - 独立第三方版本"

    def process(self, vae, force_offload, input_video, mask, ref1, chunk_frames, ref2=None, tiled_vae=False, context_stride=4, context_overlap=16, closed_loop=True, freenoise=False, verbose=False, fuse_method="pyramid", use_first_frame_as_reference=True, pad_first_frames=0, force_first_window_zero=True):
        # 获取设备信息
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        W = input_video.shape[2]
        H = input_video.shape[1]
        F = input_video.shape[0]

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        # 处理首帧预热
        pad_head_frames = (max(0, pad_first_frames) // 4) * 4
        if pad_head_frames > 0:
            head = input_video[0:1].clone().repeat(pad_head_frames, 1, 1, 1)
            input_video = torch.cat([head, input_video.clone()], dim=0)
        F = input_video.shape[0]
        
        # 对齐帧数到4的倍数+1
        F = (F - 1) // 4 * 4 + 1
        input_video = input_video[: F]

        # 处理窗口参数
        chunk_frames = min(chunk_frames, F)
        chunk_frames = ((chunk_frames - 1) // 4) * 4 + 1
        
        def _m4(x, mn=4):
            return max(mn, (x // 4) * 4)
        
        context_stride = _m4(min(context_stride, chunk_frames))
        context_overlap = _m4(min(context_overlap, chunk_frames - 1))
        if context_overlap >= chunk_frames:
            context_overlap = max(4, chunk_frames - 4)

        # 清理缓存并移动VAE到设备
        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)

        # 处理输入数据
        input_video = input_video.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        ref1 = ref1.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)
        if ref2 is not None:
            ref2 = ref2.clone()[..., :3].to(device, vae.dtype).unsqueeze(0).permute(0, 4, 1, 2, 3)

        # VAE编码
        latents = vae.encode(input_video * 2.0 - 1.0, device, tiled=tiled_vae)
        
        ref_latents = vae.encode(ref1 * 2.0 - 1.0, device, tiled=tiled_vae)
        num_refs = 1
        if ref2 is not None:
            ref2_latents = vae.encode(ref2 * 2.0 - 1.0, device, tiled=tiled_vae)
            ref_latents = torch.cat([ref_latents, ref2_latents], dim=2)
            num_refs = 2
        
        # 处理掩码
        input_latent_mask = torch.nn.functional.interpolate(mask.unsqueeze(1).to(vae.dtype), size=(lat_h, lat_w), mode='nearest').unsqueeze(1)
        input_latent_mask = input_latent_mask.repeat(1, 16, 1, 1, 1).to(device, vae.dtype)

        input_latent_mask[input_latent_mask <= 0.5] = 0
        input_latent_mask[input_latent_mask > 0.5] = 1
        input_latent_mask[input_latent_mask == 0] = -1

        # 组合embeds
        mocha_embeds = torch.cat([latents, input_latent_mask, ref_latents], dim=2)
        mocha_embeds = mocha_embeds[0]

        target_shape = (16, (F - 1) // 4 + 1, lat_h, lat_w)
        seq_len = (target_shape[1] * 2 + 1 + num_refs) * (target_shape[2] * target_shape[3] // 4)

        # 清理资源
        if force_offload:
            vae.model.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()

        image_embeds = {
            "seq_len": seq_len,
            "mocha_embeds": mocha_embeds,
            "num_frames": F,
            "target_shape": target_shape,
            "mocha_num_refs": num_refs,
        }
        
        if pad_head_frames > 0:
            image_embeds["trim_head_frames"] = pad_head_frames

        ref_init = latents[:, :, :1] if use_first_frame_as_reference else None

        context_schedule = "static_standard" if force_first_window_zero else ("uniform_looped" if closed_loop else "uniform_standard")

        context_options = {
            "context_schedule": context_schedule,
            "context_frames": chunk_frames,
            "context_stride": context_stride,
            "context_overlap": context_overlap,
            "freenoise": freenoise,
            "verbose": verbose,
            "fuse_method": fuse_method,
            "reference_latent": ref_init,
        }

        return (image_embeds, context_options)


# 节点映射配置
NODE_CLASS_MAPPINGS = {
    "MochaLoopEmbedsStandalone": MochaLoopEmbedsStandalone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MochaLoopEmbedsStandalone": "lian MochaLoopEmbeds",
}