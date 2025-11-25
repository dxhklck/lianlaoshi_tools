import torch
import torch.nn.functional as F
import torchaudio

from comfy.utils import common_upscale
import comfy.model_management as mm

import logging
log = logging.getLogger(__name__)


device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

def _encode_ref_images(vae, images, enable_vae_tiling=False,
                       tile_x=272, tile_y=272, tile_stride_x=144, tile_stride_y=128,
                       latent_strength=1.0):
    """Internal helper to encode a batch of NHWC images into per-frame latents.
    Mirrors the behavior of WanVideoEncodeLatentBatch.encode but without node registration.
    Returns a tensor of shape [B, C, 1, H//8, W//8].
    """
    vae.to(device)

    images = images.clone()
    B, H, W, C = images.shape

    # Ensure dimensions are divisible by 16 for VAE
    if W % 16 != 0 or H % 16 != 0:
        new_height = (H // 16) * 16
        new_width = (W // 16) * 16
        log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
        images = common_upscale(images.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

    # Drop alpha channel if present
    if images.shape[-1] == 4:
        images = images[..., :3]

    # Match dtype and normalize to [-1, 1]
    images = images.to(vae.dtype).to(device) * 2.0 - 1.0

    latent_list = []
    for img in images:
        # Add batch and time dimensions, permute to [B, C, T, H, W]
        img_btchw = img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3)

        if enable_vae_tiling and tile_x is not None:
            latent = vae.encode(
                img_btchw,
                device=device,
                tiled=enable_vae_tiling,
                tile_size=(tile_x // vae.upsampling_factor, tile_y // vae.upsampling_factor),
                tile_stride=(tile_stride_x // vae.upsampling_factor, tile_stride_y // vae.upsampling_factor),
            )
        else:
            latent = vae.encode(img_btchw, device=device, tiled=enable_vae_tiling)

        latent = latent[0]  # [1, C, 1, H//8, W//8]
        if latent_strength != 1.0:
            latent = latent * latent_strength
        latent_list.append(latent.squeeze(0).cpu())  # [C, 1, H//8, W//8]

    latents_out = torch.stack(latent_list, dim=0)  # [B, C, 1, H//8, W//8]
    log.info(f"WanVideoEncode (internal): Encoded latents shape {latents_out.shape}")
    vae.to(offload_device)
    mm.soft_empty_cache()
    return latents_out

def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)  # [1, C, T]
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

def get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift=2):
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
        audio_emb_wind.append(wind_feat)
    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

    return audio_emb_wind, ed - audio_shift

class lianHuMoEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "num_frames": ("INT", {"default": 81, "min": -1, "max": 10000, "step": 1, "tooltip": "The total frame count to generate."}),
            "width": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
            "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the audio conditioning"}),
            "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
            "audio_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The percent of the video to start applying audio conditioning"}),
            "audio_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The percent of the video to stop applying audio conditioning"}),
            "motion_amplitude": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "Amplify gray-frame residual to fight 4-step slow-motion. 1.0=off, 1.15-1.3=safe speed-up, ≥1.5=experimental"})
        },
            "optional" : {
                "whisper_model": ("WHISPERMODEL",),
                "vae": ("WANVAE", ),
                "reference_images": ("IMAGE", {"tooltip": "reference images for the humo model"}),
                "start_image": ("IMAGE", {"tooltip": "optional start frame image for HuMo"}),
                "end_image": ("IMAGE", {"tooltip": "optional end frame image for HuMo"}),
                "audio": ("AUDIO",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds", )
    FUNCTION = "process"
    CATEGORY = "lianlaoshi/video"

    def process(self, num_frames, width, height, audio_scale, audio_cfg_scale, audio_start_percent, audio_end_percent, motion_amplitude, whisper_model=None, vae=None, reference_images=None, start_image=None, end_image=None, audio=None, tiled_vae=False):
        if reference_images is not None and vae is None:
            raise ValueError("VAE is required when reference images are provided")
        if whisper_model is None and audio is not None:
            raise ValueError("Whisper model is required when audio is provided")
        model = whisper_model["model"]
        feature_extractor = whisper_model["feature_extractor"]
        dtype = whisper_model["dtype"]

        sampling_rate = 16000

        if audio is not None:
            audio_input = audio["waveform"][0]
            sample_rate = audio["sample_rate"]

            if sample_rate != sampling_rate:
                audio_input = torchaudio.functional.resample(audio_input, sample_rate, sampling_rate)
            if audio_input.shape[1] == 2:
                audio_input = audio_input.mean(dim=0, keepdim=False)
            else:
                audio_input = audio_input[0]

            model.to(device)
            audio_len = len(audio_input) // 640

            # feature extraction
            audio_features = []
            window = 750*640
            for i in range(0, len(audio_input), window):
                audio_feature = feature_extractor(audio_input[i:i+window], sampling_rate=sampling_rate, return_tensors="pt").input_features
                audio_features.append(audio_feature)
            audio_features = torch.cat(audio_features, dim=-1).to(device, dtype)

            # preprocess
            window = 3000
            audio_prompts = []
            for i in range(0, audio_features.shape[-1], window):
                audio_prompt = model.encoder(audio_features[:,:,i:i+window], output_hidden_states=True).hidden_states
                audio_prompt = torch.stack(audio_prompt, dim=2)
                audio_prompts.append(audio_prompt)

            model.to(offload_device)

            audio_prompts = torch.cat(audio_prompts, dim=1)
            audio_prompts = audio_prompts[:,:audio_len*2]

            feat0 = linear_interpolation_fps(audio_prompts[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation_fps(audio_prompts[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation_fps(audio_prompts[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation_fps(audio_prompts[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation_fps(audio_prompts[:, :, 32], 50, 25)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]
            # 按开始/结束百分比与音频权重降低系数进行缩放
            t_len = audio_emb.shape[0]
            st = max(0, min(t_len, int(round(audio_start_percent * t_len))))
            ed = max(0, min(t_len, int(round(audio_end_percent * t_len))))
            if ed < st:
                st, ed = ed, st
            scale_vec = torch.zeros(t_len, dtype=audio_emb.dtype, device=audio_emb.device)
            scale_vec[st:ed] = float(audio_scale)
            audio_emb = audio_emb * scale_vec.view(t_len, 1, 1)
        else:
            audio_emb = torch.zeros(num_frames, 5, 1280, device=device)
            audio_len = num_frames
            
        pixel_frame_num = num_frames if num_frames != -1 else audio_len
        pixel_frame_num = 4 * ((pixel_frame_num - 1) // 4) + 1
        latent_frame_num = (pixel_frame_num - 1) // 4 + 1

        log.info(f"HuMo set to generate {pixel_frame_num} frames")

        #audio_emb, _ = get_audio_emb_window(audio_emb, pixel_frame_num, frame0_idx=0)

        # 处理参考图像与首尾帧
        num_refs = 0
        ref_batches = []
        has_start = False
        has_end = False
        if start_image is not None:
            has_start = True
            if start_image.shape[1] != height or start_image.shape[2] != width:
                start_in = common_upscale(start_image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            else:
                start_in = start_image
            ref_batches.append(start_in)
        if end_image is not None:
            has_end = True
            if end_image.shape[1] != height or end_image.shape[2] != width:
                end_in = common_upscale(end_image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            else:
                end_in = end_image
            ref_batches.append(end_in)
        if reference_images is not None:
            if reference_images.shape[1] != height or reference_images.shape[2] != width:
                reference_images_in = common_upscale(reference_images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            else:
                reference_images_in = reference_images
            ref_batches.append(reference_images_in)

        if len(ref_batches) > 0:
            # 合并所有参考批次到一个批次 [N, H, W, C]
            reference_all = torch.cat(ref_batches, dim=0)
            # 使用内部辅助函数进行参考图编码
            encoded_btchw = _encode_ref_images(
                vae,
                reference_all,
                enable_vae_tiling=tiled_vae,
                tile_x=272,
                tile_y=272,
                tile_stride_x=144,
                tile_stride_y=128,
                latent_strength=1.0,
            )  # [N, C, 1, H//8, W//8]
            # 转为 [C, N, H//8, W//8] 以便后续按参考索引访问
            samples = encoded_btchw.transpose(0, 1).squeeze(2)
            num_refs = samples.shape[1]

        vae.to(device)
        zero_frames = torch.zeros(1, 3, pixel_frame_num + 4*num_refs, height, width, device=device, dtype=vae.dtype)
        zero_latents = vae.encode(zero_frames, device=device, tiled=tiled_vae)[0].to(offload_device)
        
        # motion_amplitude 残差放大：对抗 4-step 慢动作
        if motion_amplitude != 1.0:
            # 提取第一帧 latent 作为“基准”
            first_latent = zero_latents[:, :, :1]  # [C,1,H,W]
            residual = zero_latents - first_latent
            residual = residual * motion_amplitude
            # clamp 防止溢出，保持颜色稳定
            residual = torch.clamp(residual, -5.0, 5.0)
            zero_latents = first_latent + residual
        
        vae.to(offload_device)
        mm.soft_empty_cache()

        target_shape = (16, latent_frame_num + num_refs, height // 8, width // 8)

        mask = torch.ones(4, target_shape[1], target_shape[2], target_shape[3], device=offload_device, dtype=vae.dtype)
        if num_refs > 0:
            mask[:,:-num_refs] = 0
            # 基于首尾帧对时间轴进行线性引导，提升图像权重
            guided_time = zero_latents[:, :latent_frame_num].clone()
            # 提取首尾参考 latent（按 ref_batches 追加顺序：start -> end -> others）
            ref_idx = 0
            if has_start:
                start_latent = samples[:, ref_idx]
                ref_idx += 1
                w_start = torch.linspace(1.0, 0.0, steps=latent_frame_num, device=offload_device, dtype=vae.dtype)
                guided_time = guided_time + start_latent.unsqueeze(1) * w_start.view(1, -1, 1, 1)
            if has_end:
                end_latent = samples[:, ref_idx]
                ref_idx += 1
                w_end = torch.linspace(0.0, 1.0, steps=latent_frame_num, device=offload_device, dtype=vae.dtype)
                guided_time = guided_time + end_latent.unsqueeze(1) * w_end.view(1, -1, 1, 1)
            # 拼接时间轴引导 + 参考集合
            image_cond = torch.cat([guided_time, samples], dim=1)
        else:
            # 无参考时保持原逻辑：不进行图像引导
            image_cond = zero_latents
            mask = torch.zeros_like(mask)
        image_cond = torch.cat([mask, image_cond], dim=0)
        image_cond_neg = torch.cat([mask, zero_latents], dim=0)

        embeds = {
            "humo_audio_emb": audio_emb,
            "humo_audio_emb_neg": torch.zeros_like(audio_emb, dtype=audio_emb.dtype, device=audio_emb.device),
            "humo_image_cond": image_cond,
            "humo_image_cond_neg": image_cond_neg,
            "humo_reference_count": num_refs,
            "humo_has_start_image": int(has_start),
            "humo_has_end_image": int(has_end),
            "humo_start_end_schedule": "linear",
            "target_shape": target_shape,
            "num_frames": pixel_frame_num,
            "humo_audio_scale": audio_scale,
            "humo_audio_cfg_scale": audio_cfg_scale,
            "humo_start_percent": audio_start_percent,
            "humo_end_percent": audio_end_percent,
        }
        
        return (embeds, )
    
NODE_CLASS_MAPPINGS = {
    "lianHuMoEmbeds": lianHuMoEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "lianHuMoEmbeds": "lian HuMoEmbeds",
}
