import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List


class VideoTransitionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video1": ("IMAGE", {"tooltip": "第一个视频帧序列 [N,H,W,C]"}),
                "video2": ("IMAGE", {"tooltip": "第二个视频帧序列 [N,H,W,C]"}),
                "transition_mode": ([
                    "crossfade",
                    "fade_to_black",
                    "fade_to_white",
                ], {"default": "crossfade"}),
                "transition_frames": ("INT", {"default": 5, "min": 1, "max": 600, "step": 1}),
            },
            "optional": {
                "target_width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "输出宽度；0表示取video1的宽度"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1, "tooltip": "输出高度；0表示取video1的高度"}),
                "length_mode": (["overlap", "preserve_total", "insert"], {"default": "preserve_total", "tooltip": "转场长度策略"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply_transition"
    CATEGORY = "lianlaoshi/video"

    def _resize_frame(self, frame: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        # frame: [H,W,C] float 0..1
        h, w, c = frame.shape
        th, tw = size_hw
        if h == th and w == tw:
            return frame
        frame_nchw = frame.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        resized = F.interpolate(frame_nchw, size=(th, tw), mode="bilinear", align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0)

    def _ensure_float(self, img: torch.Tensor | None) -> torch.Tensor | None:
        if img is None:
            return None
        if img.dtype != torch.float32:
            return img.float()
        return img

    def _blend(self, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        return a * (1.0 - alpha) + b * alpha

    def apply_transition(
        self,
        video1: torch.Tensor,
        video2: torch.Tensor,
        transition_mode: str,
        transition_frames: int,
        target_width: int = 0,
        target_height: int = 0,
        length_mode: str = "preserve_total",
    ) -> Tuple[torch.Tensor]:
        # 输入期望: [N,H,W,C]、范围0..1
        v1 = self._ensure_float(video1)
        v2 = self._ensure_float(video2)

        # 兼容 None 输入，安全获取形状与通道数
        if v1 is not None:
            n1, h1, w1, c1 = v1.shape
        else:
            n1, h1, w1, c1 = 0, 0, 0, (v2.shape[-1] if v2 is not None else 3)

        if v2 is not None:
            n2, h2, w2, c2 = v2.shape
        else:
            n2, h2, w2, c2 = 0, 0, 0, (v1.shape[-1] if v1 is not None else 3)

        # 若两者都有帧，通道需一致；建立通道基准 c
        if n1 > 0 and n2 > 0:
            assert c1 == c2, "两个视频的通道数不一致"
        c = c1 if n1 > 0 else c2

        # 输出分辨率：优先用传入尺寸，其次 video1；若 video1 为空则用 video2
        base_w = w1 if n1 > 0 else (w2 if n2 > 0 else 1)
        base_h = h1 if n1 > 0 else (h2 if n2 > 0 else 1)
        out_w = target_width if target_width > 0 else base_w
        out_h = target_height if target_height > 0 else base_h

        # 统一尺寸，兼容空序列
        v1_resized = (
            torch.stack([self._resize_frame(v1[i], (out_h, out_w)) for i in range(n1)], dim=0)
            if n1 > 0 else torch.empty((0, out_h, out_w, c), dtype=torch.float32, device=(v1.device if v1 is not None else (v2.device if v2 is not None else "cpu")))
        )
        v2_resized = (
            torch.stack([self._resize_frame(v2[i], (out_h, out_w)) for i in range(n2)], dim=0)
            if n2 > 0 else torch.empty((0, out_h, out_w, c), dtype=torch.float32, device=(v2.device if v2 is not None else (v1.device if v1 is not None else "cpu")))
        )

        # 任一输入为空：不进行 crossfade，直接拼接
        if n1 == 0 or n2 == 0:
            output = torch.cat([v1_resized, v2_resized], dim=0)
            return (output,)

        # 计算过渡帧数（受两段长度限制）
        T = min(max(1, transition_frames), n1, n2)

        # 取尾部/头部用于转场混合
        t1 = v1_resized[n1 - T:]
        t2 = v2_resized[:T]

        # 生成转场帧
        transition = []
        for i in range(T):
            alpha = i / (T - 1) if T > 1 else 1.0
            f1 = t1[i]
            f2 = t2[i]
            if transition_mode == "crossfade":
                blended = self._blend(f1, f2, alpha)
            elif transition_mode == "fade_to_black":
                if alpha < 0.5:
                    a = 1.0 - (alpha * 2.0)
                    blended = self._blend(f1, torch.zeros_like(f1), 1.0 - a)
                else:
                    b = (alpha - 0.5) * 2.0
                    blended = self._blend(torch.zeros_like(f2), f2, b)
            elif transition_mode == "fade_to_white":
                white = torch.ones_like(f1)
                if alpha < 0.5:
                    a = 1.0 - (alpha * 2.0)
                    blended = self._blend(f1, white, 1.0 - a)
                else:
                    b = (alpha - 0.5) * 2.0
                    blended = self._blend(white, f2, b)
            else:
                blended = self._blend(f1, f2, alpha)
            transition.append(blended)
        transition_tensor = torch.stack(transition, dim=0)

        if length_mode == "overlap":
            # 原逻辑：减少总帧数
            prefix = v1_resized[:n1 - T] if n1 > T else torch.empty((0, out_h, out_w, c), dtype=v1_resized.dtype, device=v1_resized.device)
            suffix = v2_resized[T:] if n2 > T else torch.empty((0, out_h, out_w, c), dtype=v2_resized.dtype, device=v2_resized.device)
            output = torch.cat([prefix, transition_tensor, suffix], dim=0)
            return (output,)
        elif length_mode == "insert":
            # 插入：总帧数增加
            output = torch.cat([v1_resized, transition_tensor, v2_resized], dim=0)
            return (output,)
        else:
            # preserve_total：总帧数等于 n1 + n2
            K = T // 2          # 从 v1 尾部让出 K 帧
            R = T - K           # 从 v2 头部让出 R 帧
            prefix = v1_resized[:max(0, n1 - K)] if (n1 - K) > 0 else torch.empty((0, out_h, out_w, c), dtype=v1_resized.dtype, device=v1_resized.device)
            suffix = v2_resized[R:] if R < n2 else torch.empty((0, out_h, out_w, c), dtype=v2_resized.dtype, device=v2_resized.device)
            output = torch.cat([prefix, transition_tensor, suffix], dim=0)
            return (output,)


NODE_CLASS_MAPPINGS = {
    "VideoTransitionNode": VideoTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoTransitionNode": "lian Video Transition",
}