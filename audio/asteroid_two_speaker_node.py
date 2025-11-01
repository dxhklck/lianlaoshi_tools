import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from typing import Dict, Any, List, Tuple
from comfy.utils import ProgressBar
import folder_paths
import warnings


class AsteroidTwoSpeakerSeparationNode:
    """
    使用 Asteroid 两人分离模型（例如 DPRNN-TasNet / DPTNet，16k）对清唱中的重叠人声进行分离。

    说明：
    - 为保证离线可用，节点优先加载本地权重（scripted或state_dict）。
    - 若找不到权重，将返回友好错误信息并提示权重路径。
    - 将自动重采样到 16k 以匹配常见训练采样率，分离后再升采样回原采样率。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "输入人声清唱音频"}),
                "model_name": (["dprnn_tasnet_16k", "dptnet_16k"], {
                    "default": "dprnn_tasnet_16k",
                    "tooltip": "选择Asteroid两人分离模型（需本地权重）"
                }),
            },
            "optional": {
                "use_gpu": ("BOOLEAN", {"default": True, "tooltip": "是否使用GPU加速（如可用）"}),
                "energy_threshold": ("FLOAT", {"default": 1e-3, "min": 0.0, "max": 0.1, "step": 1e-3,
                                                "tooltip": "低能量过滤阈值（避免空轨）"}),
                "max_speakers": ("INT", {"default": 2, "min": 2, "max": 2, "step": 1,
                                            "tooltip": "固定为两人分离"}),
            }
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT", "STRING")
    RETURN_NAMES = ("separated_tracks", "speaker_count", "info")
    FUNCTION = "separate_two"
    CATEGORY = "lianlaoshi/audio"

    def __init__(self):
        # 优先使用 ComfyUI models 目录
        self.local_dir = os.path.join(folder_paths.models_dir, "asteroid")
        # 记录最近加载模型采样率（用于输入重采样）
        self._last_model_sr = None

    def _load_local_model(self, model_name: str, device: torch.device):
        """
        优先加载本地权重。支持两种形式：
        - scripted: model_scripted.pt (torch.jit)
        - state dict: 如存在 model_state.pt / checkpoint.pt / *.pt，则尝试自动构建并导出 TorchScript
        """
        model_dir = os.path.join(self.local_dir, model_name)
        scripted_path = os.path.join(model_dir, "model_scripted.pt")
        if os.path.isfile(scripted_path) and os.path.getsize(scripted_path) > 0:
            print(f"从本地加载Asteroid scripted模型: {scripted_path}")
            model = torch.jit.load(scripted_path, map_location=device)
            model.eval()
            return model
        # 自动发现并尝试从 state_dict 导出 TorchScript
        candidate_names = [
            "model_state.pt", "checkpoint.pt", "state_dict.pt", "checkpoint.pth", "model.pth"
        ]
        # 扩展：扫描目录中的 .pt/.ckpt 文件
        # 扫描目录及子目录，寻找权重文件
        found_paths: List[str] = []
        if os.path.isdir(model_dir):
            for root, dirs, files in os.walk(model_dir):
                for fname in files:
                    if fname.lower().endswith((".pt", ".ckpt", ".pth", ".bin")):
                        full = os.path.join(root, fname)
                        found_paths.append(full)
        # 将常见文件名优先级排前
        ordered_paths = []
        priority_names = ["model_state.pt", "checkpoint.pt", "state_dict.pt", "checkpoint.pth", "model.pth"]
        for p in found_paths:
            name = os.path.basename(p)
            if name in priority_names:
                ordered_paths.insert(0, p)
            else:
                ordered_paths.append(p)
        # 追加根目录下的候选名（兼容旧逻辑）
        for cand in candidate_names:
            cand_path = os.path.join(model_dir, cand)
            if os.path.isfile(cand_path) and os.path.getsize(cand_path) > 0:
                ordered_paths.insert(0, cand_path)
        # 逐个尝试自动导出
        for cand_path in ordered_paths:
            if os.path.isfile(cand_path) and os.path.getsize(cand_path) > 0:
                print(f"发现候选权重: {cand_path}，尝试自动加载或导出 TorchScript...")
                try:
                    # HuggingFace 预训练配置（pytorch.bin）：包含 model_args 与 state_dict
                    base = os.path.basename(cand_path).lower()
                    if base == "pytorch.bin" or base.endswith(".bin"):
                        import torch
                        conf = torch.load(cand_path, map_location="cpu")
                        model_args = conf.get("model_args", {})
                        sr_conf = int(model_args.get("sample_rate", 16000))
                        self._last_model_sr = sr_conf
                        arch = model_name
                        if arch.startswith("dprnn"):
                            try:
                                from asteroid.models import DPRNNTasNet
                            except Exception:
                                from asteroid.models.dprnn import DPRNNTasNet
                            model = DPRNNTasNet.from_pretrained(cand_path)
                        elif arch.startswith("dptnet"):
                            try:
                                from asteroid.models import DPTNet
                            except Exception:
                                from asteroid.models.dptnet import DPTNet
                            model = DPTNet.from_pretrained(cand_path)
                        else:
                            raise RuntimeError(f"未知架构: {arch}")
                        model.eval()
                        return model
                    # 读取可选超参
                    hparams_path = os.path.join(model_dir, "hparams.json")
                    hparams = None
                    if os.path.isfile(hparams_path):
                        import json
                        with open(hparams_path, "r", encoding="utf-8") as f:
                            hparams = json.load(f)
                    # 构建模型架构
                    arch = model_name
                    default_hparams = {
                        "n_src": 2,
                        "sample_rate": 16000,
                        "n_filters": 64,
                        "kernel_size": 16,
                        "stride": 8,
                        "bn_chan": 128,
                        "hid_size": 128,
                        "chunk_size": 100,
                        "n_repeats": 6,
                        "mask_act": "relu",
                    }
                    # 如果存在 config.yaml（例如 ESPNet 导出），尝试解析采样率
                    cfg_path = os.path.join(model_dir, "exp", "enh_train_enh_dprnn_tasnet_raw", "config.yaml")
                    sample_rate_hint = None
                    try:
                        if os.path.isfile(cfg_path):
                            with open(cfg_path, "r", encoding="utf-8") as cf:
                                text = cf.read()
                            import re
                            m = re.search(r"\bfs\s*:\s*(\d+)", text)
                            if m:
                                sample_rate_hint = int(m.group(1))
                                print(f"从 config.yaml 解析到采样率 fs={sample_rate_hint}")
                    except Exception as cfg_err:
                        print(f"解析 config.yaml 失败: {cfg_err}")
                    hp = hparams or default_hparams
                    if sample_rate_hint in (8000, 16000, 44100, 48000):
                        hp["sample_rate"] = sample_rate_hint
                        self._last_model_sr = sample_rate_hint
                    try:
                        if arch.startswith("dprnn"):
                            try:
                                from asteroid.models import DPRNNTasNet
                            except Exception:
                                from asteroid.models.dprnn import DPRNNTasNet
                            model_build = DPRNNTasNet(
                                n_src=hp.get("n_src", 2), sample_rate=hp.get("sample_rate", 16000),
                                n_filters=hp.get("n_filters", 64), kernel_size=hp.get("kernel_size", 16),
                                stride=hp.get("stride", 8), bn_chan=hp.get("bn_chan", 128),
                                hid_size=hp.get("hid_size", 128), chunk_size=hp.get("chunk_size", 100),
                                n_repeats=hp.get("n_repeats", 6), mask_act=hp.get("mask_act", "relu")
                            )
                        elif arch.startswith("dptnet"):
                            try:
                                from asteroid.models import DPTNet
                            except Exception:
                                from asteroid.models.dptnet import DPTNet
                            model_build = DPTNet(
                                n_src=hp.get("n_src", 2), sample_rate=hp.get("sample_rate", 16000),
                                n_filters=hp.get("n_filters", 64), kernel_size=hp.get("kernel_size", 16),
                                stride=hp.get("stride", 8), bn_chan=hp.get("bn_chan", 128),
                                hid_size=hp.get("hid_size", 128), chunk_size=hp.get("chunk_size", 100),
                                n_repeats=hp.get("n_repeats", 6), mask_act=hp.get("mask_act", "relu")
                            )
                        else:
                            raise RuntimeError(f"未知架构: {arch}")
                        model_build.eval()
                    except Exception as build_err:
                        raise RuntimeError(f"构建架构失败: {build_err}")

                    # 加载 state_dict
                    state = torch.load(cand_path, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    new_state = {}
                    for k, v in state.items():
                        nk = k
                        for prefix in ("model.", "net.", "module."):
                            if nk.startswith(prefix):
                                nk = nk[len(prefix):]
                        new_state[nk] = v
                    missing, unexpected = model_build.load_state_dict(new_state, strict=False)
                    print(f"加载权重完成 (missing: {missing}, unexpected: {unexpected})")

                    # 导出 TorchScript 并缓存
                    scripted = torch.jit.script(model_build)
                    scripted.eval()
                    os.makedirs(model_dir, exist_ok=True)
                    scripted.save(scripted_path)
                    print(f"已导出 TorchScript 到: {scripted_path}")
                    model = torch.jit.load(scripted_path, map_location=device)
                    model.eval()
                    return model
                except Exception as export_err:
                    print(f"从 {cand_path} 自动导出失败: {export_err}")
                    continue
        raise FileNotFoundError(
            f"未找到模型权重: {scripted_path}。可将 state_dict 放到该目录并命名为 model_state.pt，或任意 *.pt/.ckpt，并附带 hparams.json；系统会尝试自动导出。"
        )

    def separate_two(self, audio: Dict[str, Any], model_name: str, use_gpu: bool = True,
                     energy_threshold: float = 1e-3, max_speakers: int = 2) -> Tuple[List[Dict[str, Any]], int, str]:
        progress_bar = ProgressBar(5)
        progress_bar.update_absolute(1, 5)

        # 设备
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        # 读取与降采样到16k
        W = audio["waveform"]
        sr = int(audio["sample_rate"])
        if W.ndim == 2:
            W = W.mean(dim=0)
        W = W.contiguous().to(device)

        # 根据模型采样率推断（优先使用最近加载的配置；否则解析目录；默认16k）
        target_sr = self._last_model_sr if getattr(self, "_last_model_sr", None) else 16000
        try:
            model_dir = os.path.join(self.local_dir, model_name)
            cfg_path = os.path.join(model_dir, "exp", "enh_train_enh_dprnn_tasnet_raw", "config.yaml")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    text = cf.read()
                import re
                m = re.search(r"\bfs\s*:\s*(\d+)", text)
                if m:
                    target_sr = int(m.group(1))
                    print(f"检测到模型采样率: {target_sr}Hz (来自 config.yaml)")
        except Exception as sr_err:
            print(f"采样率推断失败，使用默认 {target_sr}Hz: {sr_err}")
        resampler_down = None
        resampler_up = None
        if sr != target_sr:
            print(f"重采样到模型采样率: {sr}Hz -> {target_sr}Hz")
            resampler_down = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr).to(device)
            resampler_up = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=sr).to(device)
            W16k = resampler_down(W.unsqueeze(0)).squeeze(0)
        else:
            W16k = W

        progress_bar.update_absolute(2, 5)

        # 加载模型（仅使用 Asteroid，不回退）
        try:
            model = self._load_local_model(model_name, device)
            # 显式将模型移动到当前设备，避免 CPU/GPU 混用
            if hasattr(model, "to"):
                model = model.to(device)
        except Exception as e:
            msg = (
                f"Asteroid模型加载失败: {e}。\n"
                f"请确保在目录 `{os.path.join(self.local_dir, model_name)}` 放置与 Asteroid 架构兼容的两人分离权重：\n"
                f"- 优先 `model_scripted.pt` (TorchScript)。\n"
                f"- 或兼容的 `state_dict`（*.pt/*.pth/*.ckpt）及可选 `hparams.json`，系统将自动导出为 TorchScript。\n"
                f"注意：ESPNet 训练工程的权重（如 `exp/.../96epoch.pth`）与 Asteroid 命名不一致，无法直接加载。"
            )
            print(msg)
            return ([], 0, msg)

        progress_bar.update_absolute(3, 5)

        # 分离（仅 Asteroid）
        try:
            with torch.no_grad():
                # 模型期望输入形状：通常为 [B, T] 或 [B, 1, T]，此处尝试 [1, T]
                inp = W16k.unsqueeze(0)
                est_sources = model(inp)  # 期望输出 [B, N, T]
            if isinstance(est_sources, (list, tuple)):
                est_sources = est_sources[0]
            if est_sources.ndim == 2:
                # [N, T]
                est_sources = est_sources.unsqueeze(0)
            print(f"分离输出形状: {tuple(est_sources.shape)} (期望 [1, 2, T])")
        except Exception as e:
            msg = f"两人分离过程失败: {e}"
            print(msg)
            return ([], 0, msg)

        progress_bar.update_absolute(4, 5)

        # --- 保质掩模重建（默认） ---
        # 思路：用分离结果的幅度比生成软掩模（在模型采样率），再插值到原采样率，在原始整轨上重建两条音轨。
        B, N, T = est_sources.shape
        if N < 2:
            print(f"警告：分离结果仅包含 {N} 条源，仍按两人分离处理")
        # 取前两源并计算幅度
        s1 = est_sources[0, 0]
        s2 = est_sources[0, 1] if N >= 2 else torch.zeros_like(s1)
        abs_s1 = torch.abs(s1)
        abs_s2 = torch.abs(s2)
        eps = 1e-8
        denom = abs_s1 + abs_s2 + eps
        mask1_model = abs_s1 / denom
        mask2_model = abs_s2 / denom if N >= 2 else (1.0 - mask1_model)
        # 插值到原采样率
        if sr != target_sr:
            print(f"插值掩模到原采样率: {target_sr}Hz -> {sr}Hz")
            up = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=sr).to(device)
            mask1_up = up(mask1_model.unsqueeze(0)).squeeze(0)
            mask2_up = up(mask2_model.unsqueeze(0)).squeeze(0)
        else:
            mask1_up = mask1_model
            mask2_up = mask2_model
        # 初步平滑掩模以减少抖动（~20ms窗口）
        def _smooth(m: torch.Tensor, win_ms: float = 20.0) -> torch.Tensor:
            # 规范为1D向量，避免出现 [B, C, S] 等多维输入导致 conv1d 报错
            if m is None:
                return m
            m = m.detach()
            if m.dim() == 0:
                return m
            if m.dim() > 1:
                m = m.reshape(-1)
            L = m.numel()
            if L <= 2:
                return m
            k_len = max(3, int((win_ms / 1000.0) * sr))
            k_len = min(k_len, max(3, L // 4))  # 内核不超过长度的1/4，避免过度平滑与Pad错误
            if k_len <= 1:
                return m
            k = torch.ones(1, 1, k_len, device=device) / float(k_len)
            x = m.view(1, 1, L)
            pad = min(k_len // 2, max(0, L - 1))
            x = F.pad(x, (pad, pad), mode="reflect")
            y = F.conv1d(x, k)
            y = y.view(-1)
            # 对齐回原长度
            if y.numel() > L:
                y = y[:L]
            return y
        mask1_up = _smooth(mask1_up, 20.0)
        mask2_up = _smooth(mask2_up, 20.0)
        # 对齐长度到原始整轨
        orig_wave = audio["waveform"].detach()
        # 统一形状为 [C, S]
        if orig_wave.dim() == 3:
            # [B, C, S] -> 取第一个批次
            orig_wave = orig_wave[0]
        elif orig_wave.dim() == 1:
            orig_wave = orig_wave.unsqueeze(0)
        elif orig_wave.dim() != 2:
            orig_wave = orig_wave.view(1, -1)
        C, S = orig_wave.shape
        def align_len(m: torch.Tensor, target_len: int) -> torch.Tensor:
            if m.numel() >= target_len:
                return m[:target_len]
            pad_len = target_len - m.numel()
            if pad_len > 0:
                last = m[-1] if m.numel() > 0 else torch.tensor(0.0, device=m.device)
                pad = last.repeat(pad_len)
                return torch.cat([m, pad], dim=0)
            return m
        mask1_up = align_len(mask1_up, S)
        mask2_up = align_len(mask2_up, S)
        # 仅在强非重叠段保留原音（主讲者=1，次讲者=0），在重叠段使用软掩模
        dom_th = 0.85
        min_th = 0.15
        dom1 = (mask1_up >= dom_th) & (mask2_up <= min_th)
        dom2 = (mask2_up >= dom_th) & (mask1_up <= min_th)
        overlap = ~(dom1 | dom2)
        # 应用门控：非重叠段直接设为1/0，重叠段保留当前值
        mask1_up = torch.where(dom1, torch.ones_like(mask1_up), mask1_up)
        mask2_up = torch.where(dom1, torch.zeros_like(mask2_up), mask2_up)
        mask1_up = torch.where(dom2, torch.zeros_like(mask1_up), mask1_up)
        mask2_up = torch.where(dom2, torch.ones_like(mask2_up), mask2_up)
        # 统计日志
        dom1_ratio = float(dom1.sum().item()) / float(S) if S > 0 else 0.0
        dom2_ratio = float(dom2.sum().item()) / float(S) if S > 0 else 0.0
        overlap_ratio = float(overlap.sum().item()) / float(S) if S > 0 else 0.0
        print(f"段落占比：主轨1={dom1_ratio:.2%}, 主轨2={dom2_ratio:.2%}, 重叠={overlap_ratio:.2%}")
        # 二次平滑（~10ms）并归一化，减少边界伪影
        mask1_up = _smooth(mask1_up, 10.0)
        mask2_up = _smooth(mask2_up, 10.0)
        sum_mask = mask1_up + mask2_up
        sum_mask = torch.clamp(sum_mask, min=eps)
        mask1_up = torch.clamp(mask1_up / sum_mask, 0.0, 1.0)
        mask2_up = torch.clamp(mask2_up / sum_mask, 0.0, 1.0)
        # 能量检查（掩模层面）
        m1_energy = torch.std(mask1_up).item()
        m2_energy = torch.std(mask2_up).item()
        print(f"掩模能量：mask1={m1_energy:.6f}, mask2={m2_energy:.6f}")
        # 在原始整轨上重建两条音轨（按声道广播）
        mask1_b = mask1_up.unsqueeze(0).expand(C, -1)
        mask2_b = mask2_up.unsqueeze(0).expand(C, -1)
        y1 = (mask1_b * orig_wave.to(device)).to("cpu")
        y2 = (mask2_b * orig_wave.to(device)).to("cpu")
        # 输出准备：形状统一为 [1, C, S]
        out_list: List[Dict[str, Any]] = [
            {"waveform": y1.unsqueeze(0), "sample_rate": sr},
            {"waveform": y2.unsqueeze(0), "sample_rate": sr},
        ]
        progress_bar.update_absolute(5, 5)
        info = f"Two-speaker separation ({model_name}, quality-preserving mask reconstruction), returned {len(out_list)} tracks"
        return (out_list, len(out_list), info)


NODE_CLASS_MAPPINGS = {
    "AsteroidTwoSpeakerSeparationNode": AsteroidTwoSpeakerSeparationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsteroidTwoSpeakerSeparationNode": "lian Asteroid Two-Speaker Separation",
}