"""
WanVideo Animate Embeds - Standalone Third-party Node
This is a standalone version of WanVideoAnimateEmbeds that can be used as a third-party plugin
without modifying the original ComfyUI-WanVideoWrapper code.
"""

import torch
import math
import gc
import os
import sys
from pathlib import Path
import torch.nn.functional as F

# Add the parent directory to path to import from ComfyUI-WanVideoWrapper
comfyui_wanvideo_path = Path(__file__).parent
if str(comfyui_wanvideo_path) not in sys.path:
    sys.path.insert(0, str(comfyui_wanvideo_path))

# 颜色匹配算法实现 - 始终可用
class ColorMatcher:
    """颜色匹配算法类，用于防止长视频颜色漂移"""
    
    @staticmethod
    def reinhard_match(source, target):
        """Reinhard颜色匹配算法"""
        # 转换到LAB色彩空间
        source_lab = ColorMatcher.rgb_to_lab(source)
        target_lab = ColorMatcher.rgb_to_lab(target)
        
        # 匹配均值和方差
        matched = ColorMatcher.match_statistics(source_lab, target_lab)
        
        # 转换回RGB
        return ColorMatcher.lab_to_rgb(matched)
    
    @staticmethod
    def mkl_match(source, target):
        """MKL颜色匹配算法"""
        # 简化的MKL匹配实现
        source_mean = source.mean(dim=[-2, -1], keepdim=True)
        target_mean = target.mean(dim=[-2, -1], keepdim=True)
        
        source_std = source.std(dim=[-2, -1], keepdim=True)
        target_std = target.std(dim=[-2, -1], keepdim=True)
        
        # 标准化并重新缩放
        source_normalized = (source - source_mean) / (source_std + 1e-8)
        matched = source_normalized * target_std + target_mean
        
        return torch.clamp(matched, 0, 1)
    
    @staticmethod
    def adaptive_match(source, target, consistency_weight=0.7):
        """自适应颜色匹配，结合多种算法防止颜色漂移"""
        if source.shape != target.shape:
            # 调整尺寸匹配
            target_resized = F.interpolate(target, size=source.shape[-2:], mode='bilinear', align_corners=False)
        else:
            target_resized = target
        
        # 时间一致性检查 - 防止长视频颜色漂移
        if len(source.shape) == 4 and source.shape[1] > 1:  # 视频序列
            # 计算相邻帧的颜色变化
            source_diff = torch.abs(source[:, 1:] - source[:, :-1]).mean()
            target_diff = torch.abs(target_resized[:, 1:] - target_resized[:, :-1]).mean()
            
            # 如果目标变化过大，增强一致性权重
            if target_diff > source_diff * 1.5:
                consistency_weight = min(consistency_weight * 1.3, 0.95)
        
        # 混合多种匹配算法
        reinhard_result = ColorMatcher.reinhard_match(source, target_resized)
        mkl_result = ColorMatcher.mkl_match(source, target_resized)
        
        # 自适应权重混合
        final_result = consistency_weight * mkl_result + (1 - consistency_weight) * reinhard_result
        
        # 保持亮度一致性
        if len(source.shape) == 4:
            # [B, C, H, W] 形状
            if source.shape[1] == 1:
                source_luminance = source[:, 0]
            else:
                source_luminance = 0.299 * source[:, 0] + 0.587 * source[:, 1] + 0.114 * source[:, 2]
            result_luminance = 0.299 * final_result[:, 0] + 0.587 * final_result[:, 1] + 0.114 * final_result[:, 2]
            luminance_ratio = source_luminance / (result_luminance + 1e-8)
            final_result = final_result * luminance_ratio.unsqueeze(1)
        else:
            # [C, H, W] 形状
            if source.shape[0] == 1:
                source_luminance = source[0]
            else:
                source_luminance = 0.299 * source[0] + 0.587 * source[1] + 0.114 * source[2]
            result_luminance = 0.299 * final_result[0] + 0.587 * final_result[1] + 0.114 * final_result[2]
            luminance_ratio = source_luminance / (result_luminance + 1e-8)
            final_result = final_result * luminance_ratio
        
        return torch.clamp(final_result, 0, 1)
    
    @staticmethod
    def rgb_to_lab(rgb):
        """RGB到LAB色彩空间转换"""
        # 处理不同形状的张量
        original_shape = rgb.shape
        is_single_channel = False
        
        # 如果输入是4D张量 [B, C, H, W]
        if len(rgb.shape) == 4:
            if rgb.shape[1] == 1:  # 单通道，转换为3通道
                rgb = rgb.expand(-1, 3, -1, -1)
                is_single_channel = True
            elif rgb.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb.shape[1]}")
            # 重塑为 [B, H, W, C] 以便按通道处理
            rgb = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
        elif len(rgb.shape) == 3:
            if rgb.shape[0] == 1:  # [1, H, W]
                rgb = rgb.expand(3, -1, -1)
                is_single_channel = True
            elif rgb.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {rgb.shape[0]}")
            # 重塑为 [H*W, C] 以便按像素处理
            rgb = rgb.permute(1, 2, 0).reshape(-1, 3)
        else:
            raise ValueError(f"Expected RGB shape [3, H, W] or [B, C, H, W], got {original_shape}")
        
        # 简化的RGB到LAB转换
        # 首先转换到XYZ
        rgb = torch.clamp(rgb, 0, 1)
        mask = rgb > 0.04045
        rgb = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
        
        # 转换矩阵 (sRGB到XYZ)
        x = 0.4124564 * rgb[:, 0] + 0.3575761 * rgb[:, 1] + 0.1804375 * rgb[:, 2]
        y = 0.2126729 * rgb[:, 0] + 0.7151522 * rgb[:, 1] + 0.0721750 * rgb[:, 2]
        z = 0.0193339 * rgb[:, 0] + 0.1191920 * rgb[:, 1] + 0.9503041 * rgb[:, 2]
        
        # XYZ到LAB
        x = x / 0.95047
        z = z / 1.08883
        
        # 计算LAB值
        def f(t):
            return torch.where(t > 0.008856, torch.pow(t, 1/3), (7.787 * t + 16/116))
        
        fx = f(x)
        fy = f(y)
        fz = f(z)
        
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        lab = torch.stack([L, a, b], dim=1)  # [N, 3]
        
        # 恢复原始形状
        if len(original_shape) == 4:
            B, H, W = original_shape[0], original_shape[2], original_shape[3]
            lab = lab.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
        else:
            H, W = original_shape[1], original_shape[2]
            lab = lab.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        
        return lab
    
    @staticmethod
    def lab_to_rgb(lab):
        """LAB到RGB色彩空间转换"""
        # 处理不同形状的张量
        original_shape = lab.shape
        
        # 如果输入是4D张量 [B, C, H, W]
        if len(lab.shape) == 4:
            if lab.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {lab.shape[1]}")
            # 重塑为 [B, H, W, C] 以便按像素处理
            lab = lab.permute(0, 2, 3, 1).reshape(-1, 3)
        elif len(lab.shape) == 3:
            if lab.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {lab.shape[0]}")
            # 重塑为 [H*W, C] 以便按像素处理
            lab = lab.permute(1, 2, 0).reshape(-1, 3)
        else:
            raise ValueError(f"Expected LAB shape [3, H, W] or [B, 3, H, W], got {original_shape}")
        
        # 简化的LAB到RGB转换
        L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
        
        # LAB到XYZ
        y = (L + 16) / 116
        x = a / 500 + y
        z = y - b / 200
        
        # 计算XYZ值
        def f_inv(t):
            return torch.where(t**3 > 0.008856, t**3, (t - 16/116) / 7.787)
        
        x = f_inv(x)
        y = f_inv(y)
        z = f_inv(z)
        
        x = x * 0.95047
        z = z * 1.08883
        
        # XYZ到RGB
        r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
        g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
        b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
        
        # 伽马校正
        r = torch.clamp(r, 0, 1)
        g = torch.clamp(g, 0, 1)
        b = torch.clamp(b, 0, 1)
        
        mask_r = r > 0.0031308
        mask_g = g > 0.0031308
        mask_b = b > 0.0031308
        
        r = torch.where(mask_r, 1.055 * torch.pow(r, 1/2.4) - 0.055, 12.92 * r)
        g = torch.where(mask_g, 1.055 * torch.pow(g, 1/2.4) - 0.055, 12.92 * g)
        b = torch.where(mask_b, 1.055 * torch.pow(b, 1/2.4) - 0.055, 12.92 * b)
        
        rgb = torch.stack([r, g, b], dim=1)  # [N, 3]
        
        # 恢复原始形状
        if len(original_shape) == 4:
            B, H, W = original_shape[0], original_shape[2], original_shape[3]
            rgb = rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
        else:
            H, W = original_shape[1], original_shape[2]
            rgb = rgb.reshape(H, W, 3).permute(2, 0, 1)  # [3, H, W]
        
        return rgb
    
    @staticmethod
    def match_statistics(source, target):
        """匹配源图像和目标图像的统计特性"""
        source_mean = source.mean(dim=[-2, -1], keepdim=True)
        source_std = source.std(dim=[-2, -1], keepdim=True)
        target_mean = target.mean(dim=[-2, -1], keepdim=True)
        target_std = target.std(dim=[-2, -1], keepdim=True)
        
        # 标准化源图像
        source_normalized = (source - source_mean) / (source_std + 1e-8)
        
        # 重新缩放以匹配目标统计
        matched = source_normalized * target_std + target_mean
        
        return matched
    
    @staticmethod
    def apply_color_matching(images, reference, method='adaptive', consistency_weight=0.7):
        """应用颜色匹配到图像序列"""
        if images is None or reference is None:
            return images
        
        # 确保图像在正确范围
        images = torch.clamp(images, 0, 1)
        reference = torch.clamp(reference, 0, 1)
        
        if method == 'adaptive':
            return ColorMatcher.adaptive_match(images, reference, consistency_weight)
        elif method == 'reinhard':
            return ColorMatcher.reinhard_match(images, reference)
        elif method == 'mkl':
            return ColorMatcher.mkl_match(images, reference)
        else:
            return images  # disabled

try:
    # Try to import from the original location
    from comfy.utils import common_upscale
    import comfy.model_management as mm
    offload_device = mm.unet_offload_device()
except ImportError:
    # Fallback if comfy is not available
    try:
        # Try to import from ComfyUI-WanVideoWrapper utils
        from utils import log
        # Mock comfy functions for standalone operation
        class MockModelManagement:
            @staticmethod
            def get_torch_device():
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            @staticmethod
            def unet_offload_device():
                return torch.device("cpu")
            
            @staticmethod
            def soft_empty_cache():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        mm = MockModelManagement()
        offload_device = mm.unet_offload_device()
        
        # Mock common_upscale function
        def common_upscale(samples, width, height, upscale_method, crop):
            return torch.nn.functional.interpolate(samples, size=(height, width), mode='bilinear', align_corners=False)
            """颜色匹配算法类，用于防止长视频颜色漂移"""
            
            @staticmethod
            def reinhard_match(source, target):
                """Reinhard颜色匹配算法"""
                # 转换到LAB色彩空间
                source_lab = ColorMatcher.rgb_to_lab(source)
                target_lab = ColorMatcher.rgb_to_lab(target)
                
                # 匹配均值和方差
                matched = ColorMatcher.match_statistics(source_lab, target_lab)
                
                # 转换回RGB
                return ColorMatcher.lab_to_rgb(matched)
            
            @staticmethod
            def mkl_match(source, target):
                """MKL颜色匹配算法"""
                # 简化的MKL匹配实现
                source_mean = source.mean(dim=[-2, -1], keepdim=True)
                target_mean = target.mean(dim=[-2, -1], keepdim=True)
                
                source_std = source.std(dim=[-2, -1], keepdim=True)
                target_std = target.std(dim=[-2, -1], keepdim=True)
                
                # 标准化并重新缩放
                source_normalized = (source - source_mean) / (source_std + 1e-8)
                matched = source_normalized * target_std + target_mean
                
                return torch.clamp(matched, 0, 1)
            
            @staticmethod
            def adaptive_match(source, target, consistency_weight=0.7):
                """自适应颜色匹配，结合多种算法防止颜色漂移"""
                if source.shape != target.shape:
                    # 调整尺寸匹配
                    target_resized = F.interpolate(target, size=source.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    target_resized = target
                
                # 时间一致性检查 - 防止长视频颜色漂移
                if len(source.shape) == 4 and source.shape[1] > 1:  # 视频序列
                    # 计算相邻帧的颜色变化
                    source_diff = torch.abs(source[:, 1:] - source[:, :-1]).mean()
                    target_diff = torch.abs(target_resized[:, 1:] - target_resized[:, :-1]).mean()
                    
                    # 如果目标变化过大，增强一致性权重
                    if target_diff > source_diff * 1.5:
                        consistency_weight = min(consistency_weight * 1.3, 0.95)
                
                # 混合多种匹配算法
                reinhard_result = ColorMatcher.reinhard_match(source, target_resized)
                mkl_result = ColorMatcher.mkl_match(source, target_resized)
                
                # 自适应权重混合
                final_result = consistency_weight * mkl_result + (1 - consistency_weight) * reinhard_result
                
                # 保持亮度一致性
                source_luminance = 0.299 * source[:, 0] + 0.587 * source[:, 1] + 0.114 * source[:, 2]
                result_luminance = 0.299 * final_result[:, 0] + 0.587 * final_result[:, 1] + 0.114 * final_result[:, 2]
                
                luminance_ratio = source_luminance / (result_luminance + 1e-8)
                final_result = final_result * luminance_ratio.unsqueeze(1)
                
                return torch.clamp(final_result, 0, 1)
            
            @staticmethod
            def rgb_to_lab(rgb):
                """RGB到LAB色彩空间转换"""
                # 简化的RGB到LAB转换
                # 首先转换到XYZ
                rgb = torch.clamp(rgb, 0, 1)
                mask = rgb > 0.04045
                rgb = torch.where(mask, torch.pow((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
                
                # 转换矩阵 (sRGB到XYZ)
                x = 0.4124564 * rgb[:, 0] + 0.3575761 * rgb[:, 1] + 0.1804375 * rgb[:, 2]
                y = 0.2126729 * rgb[:, 0] + 0.7151522 * rgb[:, 1] + 0.0721750 * rgb[:, 2]
                z = 0.0193339 * rgb[:, 0] + 0.1191920 * rgb[:, 1] + 0.9503041 * rgb[:, 2]
                
                # XYZ到LAB
                x = x / 0.95047
                z = z / 1.08883
                
                mask = torch.stack([x, y, z]) > 0.008856
                x, y, z = torch.stack([x, y, z])
                
                x = torch.where(mask, torch.pow(x, 1/3), (7.787 * x + 16/116))
                y = torch.where(mask, torch.pow(y, 1/3), (7.787 * y + 16/116))
                z = torch.where(mask, torch.pow(z, 1/3), (7.787 * z + 16/116))
                
                L = 116 * y - 16
                a = 500 * (x - y)
                b = 200 * (y - z)
                
                return torch.stack([L, a, b], dim=1)
            
            @staticmethod
            def lab_to_rgb(lab):
                """LAB到RGB色彩空间转换"""
                # 简化的LAB到RGB转换
                L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
                
                # LAB到XYZ
                y = (L + 16) / 116
                x = a / 500 + y
                z = y - b / 200
                
                x = torch.where(x**3 > 0.008856, x**3, (x - 16/116) / 7.787)
                y = torch.where(y**3 > 0.008856, y**3, (y - 16/116) / 7.787)
                z = torch.where(z**3 > 0.008856, z**3, (z - 16/116) / 7.787)
                
                x = x * 0.95047
                z = z * 1.08883
                
                # XYZ到RGB
                r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
                g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
                b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
                
                # 伽马校正
                mask = torch.stack([r, g, b]) > 0.0031308
                r, g, b = torch.stack([r, g, b])
                
                r = torch.where(mask, 1.055 * torch.pow(r, 1/2.4) - 0.055, 12.92 * r)
                g = torch.where(mask, 1.055 * torch.pow(g, 1/2.4) - 0.055, 12.92 * g)
                b = torch.where(mask, 1.055 * torch.pow(b, 1/2.4) - 0.055, 12.92 * b)
                
                return torch.stack([r, g, b], dim=1)
            
            @staticmethod
            def match_statistics(source, target):
                """匹配源图像和目标图像的统计特性"""
                source_mean = source.mean(dim=[-2, -1], keepdim=True)
                source_std = source.std(dim=[-2, -1], keepdim=True)
                target_mean = target.mean(dim=[-2, -1], keepdim=True)
                target_std = target.std(dim=[-2, -1], keepdim=True)
                
                # 标准化源图像
                source_normalized = (source - source_mean) / (source_std + 1e-8)
                
                # 重新缩放以匹配目标统计
                matched = source_normalized * target_std + target_mean
                
                return matched
            
            @staticmethod
            def apply_color_matching(images, reference, method='adaptive', consistency_weight=0.7):
                """应用颜色匹配到图像序列"""
                if images is None or reference is None:
                    return images
                
                # 确保图像在正确范围
                images = torch.clamp(images, 0, 1)
                reference = torch.clamp(reference, 0, 1)
                
                if method == 'adaptive':
                    return ColorMatcher.adaptive_match(images, reference, consistency_weight)
                elif method == 'reinhard':
                    return ColorMatcher.reinhard_match(images, reference)
                elif method == 'mkl':
                    return ColorMatcher.mkl_match(images, reference)
                else:
                    return images  # disabled
    
    except ImportError as e:
        print(f"Warning: Could not import required dependencies: {e}")
        print("This node requires ComfyUI-WanVideoWrapper to be installed in the same custom_nodes directory")

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {}

class WanVideoAnimateEmbedsStandalone:
    """
    Standalone version of WanVideoAnimateEmbeds node.
    
    This node provides video animation capabilities with advanced color matching
    and consistency features, designed to work as a third-party plugin.
    
    Features:
    - Adaptive color matching to prevent color drift in long videos
    - Automatic color consistency optimization  
    - Configurable color consistency weight
    - Smart looping video detection
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vae": ("WANVAE",),
            "width": ("INT", {"default": 832, "min": 64, "max": 8096, "step": 8, "tooltip": "Width of the image to encode"}),
            "height": ("INT", {"default": 480, "min": 64, "max": 8096, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
            "force_offload": ("BOOLEAN", {"default": True}),
            "frame_window_size": ("INT", {"default": 77, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to use for temporal attention window"}),
            "color_consistency_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Weight for color consistency in adaptive matching (higher = more consistent but less flexible)"}),
            "colormatch": (
            [   
                'disabled',
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
                'adaptive',  # 新增自适应颜色匹配
            ], {
               "default": 'adaptive', "tooltip": "Color matching method to use between the windows. 'adaptive' is recommended for looping videos to prevent color drift."
            }),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the pose"}),
            "face_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "tooltip": "Additional multiplier for the face"}),
            "auto_color_match": ("BOOLEAN", {"default": True, "tooltip": "Automatically optimize color matching settings for long videos"}),
            },
            "optional": {
                "clip_embeds": ("WANVIDIMAGE_CLIPEMBEDS", {"tooltip": "Clip vision encoded image"}),
                "ref_images": ("IMAGE", {"tooltip": "Image to encode"}),
                "pose_images": ("IMAGE", {"tooltip": "end frame"}),
                "face_images": ("IMAGE", {"tooltip": "end frame"}),
                "bg_images": ("IMAGE", {"tooltip": "background images"}),
                "mask": ("MASK", {"tooltip": "mask"}),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "lianlaoshi_tools/video"

    def process(self, vae, width, height, num_frames, force_offload, frame_window_size, colormatch, pose_strength, face_strength, color_consistency_weight=0.7, auto_color_match=True,
                ref_images=None, pose_images=None, face_images=None, clip_embeds=None, tiled_vae=False, bg_images=None, mask=None):
        """
        Process video animation with advanced color matching and consistency features.
        
        This is the main processing function that handles video encoding with
        intelligent color matching to prevent drift in long videos.
        
        Args:
            vae: VAE model for encoding
            width: Target width (will be aligned to 16-pixel boundary)
            height: Target height (will be aligned to 16-pixel boundary)  
            num_frames: Number of frames (will be aligned to 4-frame boundary)
            force_offload: Whether to force model offloading after processing
            frame_window_size: Size of temporal attention window
            colormatch: Color matching method ('adaptive' recommended for looping videos)
            pose_strength: Strength multiplier for pose conditioning
            face_strength: Strength multiplier for face conditioning
            color_consistency_weight: Weight for color consistency (0.7 default, higher = more consistent)
            auto_color_match: Enable automatic color optimization for long videos
            ref_images: Reference images for encoding
            pose_images: Pose images for conditioning
            face_images: Face images for conditioning
            clip_embeds: CLIP vision embeddings
            tiled_vae: Use tiled VAE encoding for memory efficiency
            bg_images: Background images
            mask: Mask tensor for selective processing
            
        Returns:
            Dictionary containing processed embeddings and metadata with keys:
            - 'latent': Encoded latent representations
            - 'pose_latent': Pose conditioning latents
            - 'bg_latent': Background latents
            - 'ref_images': Reference images
            - 'resized_ref_images': Resized reference images
            - 'resized_pose_images': Resized pose images
            - 'resized_face_images': Resized face images
            - 'resized_bg_images': Resized background images
            - 'num_refs': Number of reference images
            - 'num_frames': Number of frames
            - 'latent_window_size': Size of latent attention window
            - 'frame_window_size': Size of frame attention window
            - 'looping': Whether video requires looping processing
            - 'vae': VAE model reference
            - 'tiled_vae': Tiled VAE flag
            - 'colormatch': Color matching method
            - 'color_consistency_weight': Color consistency weight
            - 'face_strength': Face conditioning strength
            - 'pose_strength': Pose conditioning strength
        """
        
        device = mm.get_torch_device()
        
        # Align dimensions to VAE requirements
        W = (width // 16) * 16
        H = (height // 16) * 16

        lat_h = H // vae.upsampling_factor
        lat_w = W // vae.upsampling_factor

        num_refs = ref_images.shape[0] if ref_images is not None else 0
        num_frames = ((num_frames - 1) // 4) * 4 + 1  # Align to 4-frame boundary

        looping = num_frames > frame_window_size
        
        # Smart color matching optimization
        if auto_color_match and looping:
            if colormatch == "disabled":
                colormatch = "adaptive"
                print(f"[WanVideoAnimateEmbeds] Auto-enabled adaptive color matching for looping video (frames: {num_frames}, window: {frame_window_size})")
            
            # Adjust color consistency weight based on video length
            if num_frames > 800 and color_consistency_weight < 0.9:
                color_consistency_weight = 0.9
                print(f"[WanVideoAnimateEmbeds] Increased color consistency weight to {color_consistency_weight} for extra-long video")
            elif num_frames > 500 and color_consistency_weight < 0.8:
                color_consistency_weight = 0.8
                print(f"[WanVideoAnimateEmbeds] Increased color consistency weight to {color_consistency_weight} for long video")
            elif num_frames > 200 and color_consistency_weight < 0.7:
                color_consistency_weight = 0.7
                print(f"[WanVideoAnimateEmbeds] Increased color consistency weight to {color_consistency_weight} for medium-length video")
            
            # 为超长视频强制启用更强的颜色一致性
            if num_frames > 1000:
                color_consistency_weight = max(color_consistency_weight, 0.95)
                print(f"[WanVideoAnimateEmbeds] Ultra-long video detected, forcing high color consistency: {color_consistency_weight}")
                
        # 颜色漂移预检测
        if looping and ref_images is not None and ref_images.shape[0] > 1:
            # 检查参考图像之间的颜色一致性
            ref_colors = ref_images[:, :3].mean(dim=[1, 2])  # 每帧的平均颜色
            color_variance = ref_colors.std(dim=0).mean().item()
            
            if color_variance > 0.1:  # 颜色变化较大
                print(f"[WanVideoAnimateEmbeds] Detected significant color variance in reference images: {color_variance:.3f}")
                # 增强颜色一致性权重
                color_consistency_weight = min(color_consistency_weight * 1.2, 0.95)
                print(f"[WanVideoAnimateEmbeds] Adjusted color consistency weight to {color_consistency_weight}")

        if num_frames < frame_window_size:
            frame_window_size = num_frames

        target_shape = (16, (num_frames - 1) // 4 + 1 + num_refs, lat_h, lat_w)
        latent_window_size = ((frame_window_size - 1) // 4)

        if not looping:
            num_frames = num_frames + num_refs * 4
        else:
            latent_window_size = latent_window_size + 1

        mm.soft_empty_cache()
        gc.collect()
        vae.to(device)
        
        # Initialize variables for processing
        pose_latents = ref_latent = None
        resized_pose_images = None
        resized_bg_images = None
        resized_ref_images = None
        resized_face_images = None
        
        # Process pose images if provided
        if pose_images is not None:
            pose_images = pose_images[..., :3]  # Take only RGB channels
            if pose_images.shape[1] != H or pose_images.shape[2] != W:
                resized_pose_images = common_upscale(pose_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_pose_images = pose_images.permute(3, 0, 1, 2)  # C, T, H, W
            resized_pose_images = resized_pose_images * 2 - 1  # Normalize to [-1, 1]
            
            # 应用颜色匹配到姿态图像
            if looping and colormatch != 'disabled' and ref_images is not None:
                print(f"[WanVideoAnimateEmbeds] Applying {colormatch} color matching to pose images")
                pose_for_matching = (resized_pose_images + 1) / 2
                
                # 使用参考图像的第一帧作为颜色参考
                if ref_images is not None and ref_images.shape[0] > 0:
                    # 重新处理参考图像的第一帧作为颜色参考
                    ref_first = ref_images[0:1, :H, :W, :3].permute(0, 3, 1, 2)  # 1, C, H, W
                    ref_first = (ref_first * 2 - 1 + 1) / 2  # 转换到[0,1]
                    
                    # 对姿态图像应用颜色匹配
                    for i in range(pose_for_matching.shape[1]):
                        matched_pose = ColorMatcher.apply_color_matching(
                            pose_for_matching[:, i:i+1], 
                            ref_first, 
                            method=colormatch, 
                            consistency_weight=color_consistency_weight
                        )
                        pose_for_matching[:, i:i+1] = matched_pose
                
                resized_pose_images = pose_for_matching * 2 - 1
            
            if not looping:
                pose_latents = vae.encode([resized_pose_images.to(device, vae.dtype)], device, tiled=tiled_vae)
                pose_latents = pose_latents.to(offload_device)
                
                # Pad pose latents if needed
                if pose_latents.shape[2] < latent_window_size:
                    print(f"[WanVideoAnimateEmbeds] Padding pose latents from {pose_latents.shape} to length {latent_window_size}")
                    pad_len = latent_window_size - pose_latents.shape[2]
                    pad = torch.zeros(pose_latents.shape[0], pose_latents.shape[1], pad_len, pose_latents.shape[3], pose_latents.shape[4], device=pose_latents.device, dtype=pose_latents.dtype)
                    pose_latents = torch.cat([pose_latents, pad], dim=2)
                del resized_pose_images
            else:
                resized_pose_images = resized_pose_images.to(offload_device, dtype=vae.dtype)

        # Process background images if provided
        bg_latents = None
        if bg_images is not None:
            if bg_images.shape[1] != H or bg_images.shape[2] != W:
                resized_bg_images = common_upscale(bg_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_bg_images = bg_images.permute(3, 0, 1, 2)  # C, T, H, W
            resized_bg_images = (resized_bg_images[:3] * 2 - 1)  # Normalize and take only RGB
            
            # 应用颜色匹配到背景图像
            if looping and colormatch != 'disabled' and ref_images is not None:
                print(f"[WanVideoAnimateEmbeds] Applying {colormatch} color matching to background images")
                bg_for_matching = (resized_bg_images + 1) / 2
                
                # 使用参考图像的第一帧作为颜色参考
                if ref_images is not None and ref_images.shape[0] > 0:
                    ref_first = ref_images[0:1, :H, :W, :3].permute(0, 3, 1, 2)  # 1, C, H, W
                    ref_first = (ref_first * 2 - 1 + 1) / 2  # 转换到[0,1]
                    
                    # 对背景图像应用颜色匹配
                    for i in range(bg_for_matching.shape[1]):
                        matched_bg = ColorMatcher.apply_color_matching(
                            bg_for_matching[:, i:i+1], 
                            ref_first, 
                            method=colormatch, 
                            consistency_weight=color_consistency_weight
                        )
                        bg_for_matching[:, i:i+1] = matched_bg
                
                resized_bg_images = bg_for_matching * 2 - 1

        # Generate background latents
        if not looping:
            if bg_images is None:
                resized_bg_images = torch.zeros(3, num_frames - num_refs, H, W, device=device, dtype=vae.dtype)
            bg_latents = vae.encode([resized_bg_images.to(device, vae.dtype)], device, tiled=tiled_vae)[0].to(offload_device)
            del resized_bg_images
        elif bg_images is not None:
            resized_bg_images = resized_bg_images.to(offload_device, dtype=vae.dtype)

        # Process reference images if provided
        if ref_images is not None:
            if ref_images.shape[1] != H or ref_images.shape[2] != W:
                resized_ref_images = common_upscale(ref_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_ref_images = ref_images.permute(3, 0, 1, 2)  # C, T, H, W
            resized_ref_images = resized_ref_images[:3] * 2 - 1  # Normalize and take only RGB

            # 应用颜色匹配以防止长视频颜色漂移
            if looping and colormatch != 'disabled':
                print(f"[WanVideoAnimateEmbeds] Applying {colormatch} color matching to reference images")
                # 将参考图像转换为[0,1]范围进行颜色匹配
                ref_for_matching = (resized_ref_images + 1) / 2
                
                # 如果有多个参考帧，使用第一帧作为颜色参考
                if ref_for_matching.shape[1] > 1:
                    color_reference = ref_for_matching[:, :1]  # 使用第一帧作为颜色参考
                    # 对所有帧应用颜色匹配
                    for i in range(ref_for_matching.shape[1]):
                        matched_frame = ColorMatcher.apply_color_matching(
                            ref_for_matching[:, i:i+1], 
                            color_reference, 
                            method=colormatch, 
                            consistency_weight=color_consistency_weight
                        )
                        ref_for_matching[:, i:i+1] = matched_frame
                
                # 转换回[-1,1]范围
                resized_ref_images = ref_for_matching * 2 - 1

            ref_latent = vae.encode([resized_ref_images.to(device, vae.dtype)], device, tiled=tiled_vae)[0]
            msk = torch.zeros(4, 1, lat_h, lat_w, device=device, dtype=vae.dtype)
            msk[:, :num_refs] = 1
            ref_latent_masked = torch.cat([msk, ref_latent], dim=0).to(offload_device)  # 4+C 1 H W

            # Process mask if provided
            if mask is None:
                bg_mask = torch.zeros(1, num_frames, lat_h, lat_w, device=offload_device, dtype=vae.dtype)
            else:
                # 确保mask是正确的维度格式
                if mask.dim() == 2:  # [H, W] -> [1, H, W]
                    mask = mask.unsqueeze(0)
                elif mask.dim() == 3 and mask.shape[0] == 1:  # [1, H, W] 已经是正确格式
                    pass
                elif mask.dim() == 3:  # [H, W, 1] -> [1, H, W]
                    mask = mask.permute(2, 0, 1)
                
                # 调整mask尺寸到latent空间
                if mask.shape[1] != lat_h or mask.shape[2] != lat_w:
                    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(lat_h, lat_w), mode='bilinear', align_corners=False).squeeze(0)
                
                # 扩展到时间维度 - 修复维度匹配问题
                # mask 现在是 [1, lat_h, lat_w]，需要扩展到 [1, num_frames, lat_h, lat_w]
                bg_mask = mask.unsqueeze(0).expand(1, num_frames, -1, -1).to(offload_device, dtype=vae.dtype)
        else:
            ref_latent_masked = None
            bg_mask = torch.zeros(1, num_frames, lat_h, lat_w, device=offload_device, dtype=vae.dtype)

        # Process face images if provided
        if face_images is not None:
            if face_images.shape[1] != H or face_images.shape[2] != W:
                resized_face_images = common_upscale(face_images.movedim(-1, 1), W, H, "lanczos", "disabled").movedim(0, 1)
            else:
                resized_face_images = face_images.permute(3, 0, 1, 2)  # C, T, H, W
            
            # 应用颜色匹配到人脸图像
            if looping and colormatch != 'disabled' and ref_images is not None:
                print(f"[WanVideoAnimateEmbeds] Applying {colormatch} color matching to face images")
                face_for_matching = (resized_face_images + 1) / 2
                
                # 使用参考图像的第一帧作为颜色参考
                if ref_images is not None and ref_images.shape[0] > 0:
                    ref_first = ref_images[0:1, :H, :W, :3].permute(0, 3, 1, 2)  # 1, C, H, W
                    ref_first = (ref_first * 2 - 1 + 1) / 2  # 转换到[0,1]
                    
                    # 对人脸图像应用颜色匹配
                    for i in range(face_for_matching.shape[1]):
                        matched_face = ColorMatcher.apply_color_matching(
                            face_for_matching[:, i:i+1], 
                            ref_first, 
                            method=colormatch, 
                            consistency_weight=color_consistency_weight
                        )
                        face_for_matching[:, i:i+1] = matched_face
                
                resized_face_images = face_for_matching * 2 - 1
            
            resized_face_images = resized_face_images.to(offload_device, dtype=vae.dtype)

        # Offload VAE if requested
        if force_offload:
            vae.to(offload_device)

        # Build output dictionary
        image_embeds = {
            "latent": bg_latents,
            "pose_latent": pose_latents,
            "bg_latent": bg_latents,
            "ref_images": ref_images,
            "resized_ref_images": resized_ref_images,
            "resized_pose_images": resized_pose_images,
            "resized_face_images": resized_face_images,
            "resized_bg_images": resized_bg_images,
            "num_refs": num_refs,
            "num_frames": num_frames,
            "latent_window_size": latent_window_size,
            "frame_window_size": frame_window_size,
            "looping": looping,
            "vae": vae,
            "tiled_vae": tiled_vae,
            "colormatch": colormatch,
            "color_consistency_weight": color_consistency_weight,
            "face_strength": face_strength,
            "pose_strength": pose_strength,
        }

        # Add CLIP embeddings if provided
        if clip_embeds is not None:
            image_embeds["clip_embeds"] = clip_embeds

        # Add reference latent if available
        if ref_latent_masked is not None:
            image_embeds["ref_latent"] = ref_latent_masked
            image_embeds["bg_mask"] = bg_mask

        print(f"[WanVideoAnimateEmbeds] Processed {num_frames} frames with {colormatch} color matching")
        if looping:
            print(f"[WanVideoAnimateEmbeds] Looping mode enabled with window size {frame_window_size}")
        
        return (image_embeds,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "WanVideoAnimateEmbedsStandalone": WanVideoAnimateEmbedsStandalone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoAnimateEmbedsStandalone": "lian WAnimate Embeds",
}

