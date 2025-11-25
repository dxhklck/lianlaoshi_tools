import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from typing import Dict, Any, List, Tuple
from comfy.utils import ProgressBar
import folder_paths


_VAD_MODEL = None
_VAD_REPO = "snakers4/silero-vad"
_VAD_VERSION = "silero_vad"

def _get_silero_vad():
    global _VAD_MODEL
    if _VAD_MODEL is None:
        cache_dir = os.path.join(folder_paths.models_dir, "whisper")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "silero_vad.jit")
        if not os.path.isfile(model_path):
            try:
                loaded = torch.hub.load(_VAD_REPO, _VAD_VERSION, trust_repo=True)
                _VAD_MODEL = loaded[0] if isinstance(loaded, tuple) else loaded
                try:
                    torch.jit.save(torch.jit.script(_VAD_MODEL), model_path)
                except Exception:
                    pass
            except Exception:
                try:
                    import importlib
                    sv = importlib.import_module("silero_vad")
                    _VAD_MODEL = sv.load_silero_vad()
                    try:
                        torch.jit.save(torch.jit.script(_VAD_MODEL), model_path)
                    except Exception:
                        pass
                except Exception as ee:
                    raise RuntimeError(f"Silero VAD 获取失败: {ee}")
        else:
            _VAD_MODEL = torch.jit.load(model_path, map_location="cpu")
        _VAD_MODEL.eval()
    return _VAD_MODEL

def _vad_segments_with_spans(audio, sr, min_silence_duration=0.15, threshold=0.5, window_ms=30, hop_ms=10, min_segment_duration=0.5):
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    x = audio.squeeze().cpu()
    
    # 检查音频长度是否足够 - 避免Silero VAD的"Input audio chunk is too short"错误
    # Silero VAD要求采样率/音频长度 <= 31.25，即音频长度 >= 采样率/31.25
    min_length = int(16000 / 31.25) + 1  # 约512个样本
    if x.numel() < min_length:
        # 音频太短，直接返回整个音频作为一个段
        return [{"start": 0, "end": len(x), "seg": x}], len(x)
    
    model = _get_silero_vad()
    win = max(1, int((window_ms / 1000.0) * 16000))
    hop = max(1, int((hop_ms / 1000.0) * 16000))
    
    # 确保窗口大小不超过音频长度
    win = min(win, x.numel())
    hop = min(hop, x.numel())
    
    probs = []
    with torch.no_grad():
        for s in range(0, max(1, len(x) - win + 1), hop):
            f = x[s:s + win].unsqueeze(0)
            if f.shape[-1] < win:
                f = torch.nn.functional.pad(f, (0, win - f.shape[-1]))
            
            # 额外的安全检查：确保音频块满足Silero VAD的最小长度要求
            if f.shape[-1] >= min_length:
                try:
                    prob = model(f, 16000).item()  # Silero VAD需要传入采样率16000
                    probs.append(prob)
                except RuntimeError as e:
                    if "Input audio chunk is too short" in str(e):
                        # 如果单个块仍然太短，使用保守的语音概率
                        probs.append(0.8)  # 假设为语音
                    else:
                        raise
            else:
                # 对于太短的块，假设为语音
                probs.append(0.8)
    # 如果没有有效的概率值，直接返回整个音频
    if not probs:
        return [{"start": 0, "end": len(x), "seg": x}], len(x)
    
    mask = (np.array(probs) >= threshold)
    changes = np.diff(np.concatenate(([0], (~mask).astype(int), [0])))
    silence_starts = np.where(changes == 1)[0] * hop
    silence_ends = np.where(changes == -1)[0] * hop
    min_samples = int(min_silence_duration * 16000)
    long_silences = (silence_ends - silence_starts) >= min_samples
    silence_starts = silence_starts[long_silences]
    silence_ends = silence_ends[long_silences]
    segments = []
    start = 0
    for ss, se in zip(silence_starts, silence_ends):
        seg = x[start:ss]
        if seg.numel() >= int(min_segment_duration * 16000):
            segments.append({"start": start, "end": ss, "seg": seg})
        start = se
    tail = x[start:]
    if tail.numel() >= int(min_segment_duration * 16000):
        segments.append({"start": start, "end": len(x), "seg": tail})
    
    # 如果没有任何段被检测到，或者只有一个段且音频较长，尝试强制分割
    if not segments and int(x.numel()) > 0:
        segments = [{"start": 0, "end": len(x), "seg": x}]
    elif len(segments) == 1 and len(x) > 16000 * 1.0:  # 如果音频超过1秒但只检测到一个段
        # 尝试强制分割成多个较短的段
        original_seg = segments[0]
        seg_length = len(original_seg["seg"])
        target_seg_length = max(int(0.5 * 16000), int(seg_length // 3))  # 目标每段约0.5秒或总长的1/3
        
        if seg_length > target_seg_length * 1.5:  # 只有原段足够长才分割
            new_segments = []
            for i in range(0, seg_length, target_seg_length):
                end_idx = min(i + target_seg_length, seg_length)
                if end_idx - i >= int(0.3 * 16000):  # 确保每段至少有0.3秒
                    new_segments.append({
                        "start": original_seg["start"] + i,
                        "end": original_seg["start"] + end_idx,
                        "seg": original_seg["seg"][i:end_idx]
                    })
            if len(new_segments) > 1:
                segments = new_segments
                print(f"[TwoSpeakerVAD] Forced segmentation: split 1 long segment into {len(segments)} segments")
    return segments, len(x)

def _embed_segment(seg, method):
    arr = seg.detach().cpu().numpy().astype(np.float32)
    
    # 确保音频段足够长以提取有意义的特征
    if len(arr) < 160:  # 至少10ms的音频（16000Hz * 0.01s）
        # 音频太短，使用基础统计特征但包含频域信息
        m = np.mean(arr) if len(arr) > 0 else 0.0
        s = np.std(arr) if len(arr) > 0 else 0.0
        zc = np.mean(np.abs(np.diff(np.sign(arr)))) if len(arr) > 1 else 0.0
        # 添加频域特征来区分男女声
        if len(arr) > 32:
            fft = np.fft.fft(arr)
            magnitude = np.abs(fft[:len(arr)//2])
            if len(magnitude) > 0:
                # 计算频谱质心（对男女声区分很重要）
                freqs = np.fft.fftfreq(len(arr), 1/16000)[:len(arr)//2]
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
                # 计算不同频段的能量（男女声在不同频段能量分布不同）
                low_freq_energy = np.sum(magnitude[freqs < 500])    # 低频（男声特征）
                mid_freq_energy = np.sum(magnitude[(freqs >= 500) & (freqs < 2000)])  # 中频
                high_freq_energy = np.sum(magnitude[freqs >= 2000])   # 高频（女声特征）
                total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-8
                
                return np.array([m, s, zc, spectral_centroid/1000, 
                               low_freq_energy/total_energy, mid_freq_energy/total_energy, high_freq_energy/total_energy], 
                              dtype=np.float32)
        return np.array([m, s, zc, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    if method == "resemblyzer":
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            wav = preprocess_wav(arr, sample_rate=16000)
            enc = VoiceEncoder()
            emb = enc.embed_utterance(wav)
            return emb.astype(np.float32)
        except Exception:
            pass
    
    try:
        import librosa
        # 增强的特征提取，专门针对男女声区分优化
        
        # 1. MFCC特征（更多系数和统计量）
        mfcc = librosa.feature.mfcc(y=arr, sr=16000, n_mfcc=20)  # 增加到20个MFCC系数
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 计算多阶统计特征
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        mfcc_skew = np.array([np.mean((mfcc[i] - mfcc_mean[i])**3) / (mfcc_std[i]**3 + 1e-8) for i in range(len(mfcc_mean))])
        mfcc_kurt = np.array([np.mean((mfcc[i] - mfcc_mean[i])**4) / (mfcc_std[i]**4 + 1e-8) for i in range(len(mfcc_mean))])
        
        delta_mean = delta_mfcc.mean(axis=1)
        delta_std = delta_mfcc.std(axis=1)
        delta2_mean = delta2_mfcc.mean(axis=1)
        delta2_std = delta2_mfcc.std(axis=1)
        
        # 2. 频谱特征（对男女声区分很重要）
        # 计算频谱特征，确保数组长度一致
        spectral_centroids = librosa.feature.spectral_centroid(y=arr, sr=16000)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=arr, sr=16000, roll_percent=0.85)[0]  # 使用85%滚降点，更敏感
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=arr, sr=16000)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(arr)[0]
        
        # 确保所有特征数组长度一致
        min_length = min(len(spectral_centroids), len(spectral_rolloff), len(spectral_bandwidth), len(zero_crossing_rate))
        spectral_centroids = spectral_centroids[:min_length]
        spectral_rolloff = spectral_rolloff[:min_length]
        spectral_bandwidth = spectral_bandwidth[:min_length]
        zero_crossing_rate = zero_crossing_rate[:min_length]
        
        # 计算多个滚降点百分比，增强男女声区分能力
        rolloff_25 = librosa.feature.spectral_rolloff(y=arr, sr=16000, roll_percent=0.25)[0]
        rolloff_50 = librosa.feature.spectral_rolloff(y=arr, sr=16000, roll_percent=0.50)[0]
        rolloff_75 = librosa.feature.spectral_rolloff(y=arr, sr=16000, roll_percent=0.75)[0]
        rolloff_90 = librosa.feature.spectral_rolloff(y=arr, sr=16000, roll_percent=0.90)[0]
        
        # 统一滚降点数组长度
        min_rolloff_length = min(len(rolloff_25), len(rolloff_50), len(rolloff_75), len(rolloff_90))
        rolloff_25 = rolloff_25[:min_rolloff_length]
        rolloff_50 = rolloff_50[:min_rolloff_length]
        rolloff_75 = rolloff_75[:min_rolloff_length]
        rolloff_90 = rolloff_90[:min_rolloff_length]
        
        # 与其他特征保持一致
        min_length = min(min_length, min_rolloff_length)
        spectral_centroids = spectral_centroids[:min_length]
        spectral_rolloff = spectral_rolloff[:min_length]
        spectral_bandwidth = spectral_bandwidth[:min_length]
        zero_crossing_rate = zero_crossing_rate[:min_length]
        rolloff_25 = rolloff_25[:min_length]
        rolloff_50 = rolloff_50[:min_length]
        rolloff_75 = rolloff_75[:min_length]
        rolloff_90 = rolloff_90[:min_length]
        
        # 3. 音调相关特征（男女声基频差异明显）
        pitches, magnitudes = librosa.piptrack(y=arr, sr=16000)
        if pitches.size > 0:
            valid_pitches = pitches[pitches > 0]
            if len(valid_pitches) > 0:
                pitch_mean = np.mean(valid_pitches)
                pitch_std = np.std(valid_pitches)
                pitch_median = np.median(valid_pitches)
            else:
                pitch_mean = pitch_std = pitch_median = 0
        else:
            pitch_mean = pitch_std = pitch_median = 0
        
        # 4. 频带能量比特征（男女声在不同频段能量分布不同）
        stft = librosa.stft(arr)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=16000)
        
        # 计算不同频段的能量
        low_freq_idx = freqs < 500  # 低频（男声特征）
        mid_freq_idx = (freqs >= 500) & (freqs < 2000)  # 中频
        high_freq_idx = freqs >= 2000  # 高频（女声特征）
        
        low_energy = np.mean(magnitude[low_freq_idx, :])
        mid_energy = np.mean(magnitude[mid_freq_idx, :])
        high_energy = np.mean(magnitude[high_freq_idx, :])
        total_energy = low_energy + mid_energy + high_energy + 1e-8
        
        # 5. 频谱对比度（音色特征）
        spectral_contrast = librosa.feature.spectral_contrast(y=arr, sr=16000)
        contrast_mean = spectral_contrast.mean(axis=1)
        contrast_std = spectral_contrast.std(axis=1)
        
        # 6. 组合所有特征
        features = np.concatenate([
            # MFCC特征（80维）
            mfcc_mean, mfcc_std, mfcc_skew, mfcc_kurt,  # 80维
            # Delta特征（40维）
            delta_mean, delta_std, delta2_mean, delta2_std,  # 40维
            # 频谱特征（16维）- 增强频谱滚降点特征
            [spectral_centroids.mean(), spectral_centroids.std()],
            [spectral_rolloff.mean(), spectral_rolloff.std()],
            [rolloff_25.mean(), rolloff_25.std()],
            [rolloff_50.mean(), rolloff_50.std()],
            [rolloff_75.mean(), rolloff_75.std()],
            [rolloff_90.mean(), rolloff_90.std()],
            [spectral_bandwidth.mean(), spectral_bandwidth.std()],
            [zero_crossing_rate.mean(), zero_crossing_rate.std()],
            # 音调特征（3维）
            [pitch_mean, pitch_std, pitch_median],
            # 频带能量比（3维）
            [low_energy/total_energy, mid_energy/total_energy, high_energy/total_energy],
            # 频谱对比度（14维，7个频带*2）
            contrast_mean, contrast_std  # 14维
        ], axis=0)
        
        return features.astype(np.float32)
        
    except Exception:
        # 降级到基础特征，但包含频域信息
        m = np.mean(arr)
        s = np.std(arr)
        zc = np.mean(np.abs(np.diff(np.sign(arr))))
        
        # 计算基础频域特征
        if len(arr) > 32:
            fft = np.fft.fft(arr)
            magnitude = np.abs(fft[:len(arr)//2])
            if len(magnitude) > 0:
                freqs = np.fft.fftfreq(len(arr), 1/16000)[:len(arr)//2]
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
                # 计算不同频段的能量
                low_energy = np.sum(magnitude[freqs < 500])
                mid_energy = np.sum(magnitude[(freqs >= 500) & (freqs < 2000)])
                high_energy = np.sum(magnitude[freqs >= 2000])
                total_energy = low_energy + mid_energy + high_energy + 1e-8
                
                return np.array([m, s, zc, spectral_centroid/1000, 
                               low_energy/total_energy, mid_energy/total_energy, high_energy/total_energy], 
                              dtype=np.float32)
        
        return np.array([m, s, zc, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))

class TwoSpeakerVADClusterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "需要分离的原始音频文件，支持各种常见音频格式。音频质量越好，分离效果越佳。"
                }),
                "embedding_method": (["resemblyzer", "mfcc"], {
                    "default": "resemblyzer",
                    "tooltip": "选择说话人特征提取方式：\n resemblyzer: 基于深度学习，效果更好但需要额外依赖包\n mfcc: 基于传统音频特征，更稳定但效果稍逊\n建议：如果已安装resemblyzer则优先使用，否则使用mfcc"
                }),
            },
            "optional": {
                "register_a": ("AUDIO", {
                    "tooltip": "【可选】用于识别的参考音频（说话人A），提供清晰的人声样本可显著提高识别准确率。建议提供10-30秒的干净人声。"
                }),
                "register_b": ("AUDIO", {
                    "tooltip": "【可选】用于识别的参考音频（说话人B），提供清晰的人声样本可显著提高识别准确率。建议提供10-30秒的干净人声。"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否使用GPU加速处理。如果系统有NVIDIA显卡且已安装CUDA，建议开启以获得更快的处理速度。"
                }),
                "min_silence_duration": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 2.0, "step": 0.01,
                    "tooltip": "判定为静音的最小持续时间（秒）。\n值越小：检测越敏感，可能将呼吸声误判为语音\n值越大：检测越宽松，可能漏掉短停顿\n建议：0.1-0.3秒之间，根据音频质量调整"
                }),
                "vad_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "语音活动检测的敏感度阈值。\n 0.1-0.3: 检测敏感，适合清晰音频\n 0.4-0.6: 平衡设置，适合一般音频\n 0.7-0.9: 检测严格，适合嘈杂音频\n建议：从0.5开始，根据分离效果微调"
                }),
                "window_ms": ("INT", {
                    "default": 30, "min": 10, "max": 100, "step": 1,
                    "tooltip": "VAD分析的窗口大小（毫秒）。\n 较小值（10-20ms）: 时间精度高，但可能不稳定\n 中等值（30-50ms）: 平衡选择，推荐设置\n 较大值（60-100ms）: 稳定性好，但时间精度低\n建议：保持默认30ms"
                }),
                "hop_ms": ("INT", {
                    "default": 10, "min": 5, "max": 50, "step": 1,
                    "tooltip": "VAD分析的步长（毫秒）。\n 较小值（5-10ms）: 分析密集，精度高但速度慢\n 较大值（10-30ms）: 分析稀疏，速度快但可能漏检\n建议：设置为窗口大小的1/3，如30ms窗口对应10ms步长"
                }),
                "min_segment_duration": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "保留的最小语音段长度（秒），用于过滤掉太短的语音片段。\n 较小值（0.1-0.3s）: 保留短语音，但可能包含噪音\n 较大值（0.5-1.0s）: 过滤效果好，但可能丢失短词\n建议：0.3-0.8秒，根据音频内容调整"
                }),
                "smooth_ms": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "对分离结果进行平滑处理的时间窗口（毫秒），用于减少说话人切换时的噪音。\n 0ms: 无平滑，切换快速但可能有爆音\n 10-30ms: 轻度平滑，保持自然过渡\n 40-100ms: 强度平滑，过渡柔和但可能模糊边界\n建议：15-30ms，根据音频质量调整"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT", "STRING")
    RETURN_NAMES = ("separated_tracks", "speaker_count", "info")
    FUNCTION = "separate_two"
    CATEGORY = "lianlaoshi/audio"
    DESCRIPTION = "基于VAD（语音活动检测）和聚类算法的双说话人音频分离工具。通过分析音频中的语音特征，自动识别并分离出不同的说话人声音。支持提供参考音频来提高识别准确率。"

    def separate_two(self, audio: Dict[str, Any], embedding_method: str,
                     register_a: Dict[str, Any] = None, register_b: Dict[str, Any] = None,
                     use_gpu: bool = True, min_silence_duration: float = 0.15, vad_threshold: float = 0.5,
                     window_ms: int = 30, hop_ms: int = 10, min_segment_duration: float = 0.5, smooth_ms: float = 20.0) -> Tuple[List[Dict[str, Any]], int, str]:
        """
        双说话人音频分离主函数
        
        参数说明:
        ----------
        audio : Dict[str, Any]
            输入音频字典，包含音频数据和元信息
        embedding_method : str
            说话人特征提取方法，支持"resemblyzer"或"mfcc"
        register_a : Optional[Dict[str, Any]], optional
            说话人A的参考音频，提供可提高识别准确率，默认为None
        register_b : Optional[Dict[str, Any]], optional
            说话人B的参考音频，提供可提高识别准确率，默认为None
        use_gpu : bool, optional
            是否使用GPU加速，默认为True
        min_silence_duration : float, optional
            最小静音持续时间（秒），用于VAD检测，默认为0.15
        vad_threshold : float, optional
            VAD检测阈值，范围0.1-0.9，默认为0.5
        window_ms : int, optional
            VAD分析窗口大小（毫秒），默认为30
        hop_ms : int, optional
            VAD分析步长（毫秒），默认为10
        min_segment_duration : float, optional
            保留的最小语音段长度（秒），默认为0.5
        smooth_ms : float, optional
            平滑处理时间窗口（毫秒），默认为20.0
            
        返回:
        -------
        Tuple[List[Dict[str, Any]], int, str]
            - 分离后的音频列表，每个元素是一个说话人的音频
            - 检测到的说话人数量
            - 处理信息字符串，包含详细的过程信息
            
        使用建议:
        ----------
        1. 音频质量：输入音频越清晰，分离效果越好
        2. 参考音频：提供10-30秒的干净人声作为参考可显著提高准确率
        3. 参数调节：从默认参数开始，根据分离效果逐步微调
        4. 处理时间：长音频需要更多处理时间，GPU可显著加速
        """
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
        W = audio["waveform"].squeeze(0)
        sr = int(audio["sample_rate"])
        if W.ndim > 1:
            W_mono = W.mean(dim=0)
        else:
            W_mono = W
        # VAD segmentation - 尝试不同参数以确保检测到足够语音段
        segs, total_16k = _vad_segments_with_spans(W_mono, sr, min_silence_duration, vad_threshold, window_ms, hop_ms, min_segment_duration)
        print(f"[TwoSpeakerVAD] Detected {len(segs)} speech segments")
        
        # 如果检测到的语音段太少，尝试更宽松的参数
        if len(segs) < 3:
            print(f"[TwoSpeakerVAD] Too few segments ({len(segs)}), trying relaxed parameters...")
            relaxed_segs, relaxed_total = _vad_segments_with_spans(
                W_mono, sr,
                max(0.05, min_silence_duration - 0.15),  # 显著降低最小静音持续时间
                max(0.3, vad_threshold - 0.2),          # 显著降低阈值
                window_ms, hop_ms,
                max(0.1, min_segment_duration - 0.2)    # 降低最小段时长
            )
            if len(relaxed_segs) > len(segs):
                segs, total_16k = relaxed_segs, relaxed_total
                print(f"[TwoSpeakerVAD] Using relaxed parameters, now {len(segs)} segments")
        if not segs:
            return ([], 0, "no speech segments")
        
        # 调试信息：输出检测到的语音段数量
        print(f"[TwoSpeakerVAD] Detected {len(segs)} speech segments")
        embs = [ _embed_segment(s["seg"], embedding_method) for s in segs ]
        proto = None
        if register_a is not None and register_b is not None:
            def proto_emb(aud):
                w = aud["waveform"].squeeze(0)
                r = int(aud["sample_rate"])
                if w.ndim > 1:
                    w = w.mean(dim=0)
                w16 = torchaudio.functional.resample(w, r, 16000) if r != 16000 else w
                return _embed_segment(w16, embedding_method)
            ea = proto_emb(register_a)
            eb = proto_emb(register_b)
            proto = (ea.astype(np.float32), eb.astype(np.float32))
        labels = []
        if proto is not None:
            pa, pb = proto
            for e in embs:
                ca = _cosine(e, pa)
                cb = _cosine(e, pb)
                labels.append(0 if ca >= cb else 1)
        else:
            # 检查是否有足够的样本进行聚类
            if len(embs) < 2:
                # 样本太少，无法聚类，默认只有一个说话人
                return ([{"waveform": audio["waveform"], "sample_rate": sr}], 1, "Only one speech segment found, cannot separate speakers")
            
            # 调试信息：输出特征向量维度
            print(f"[TwoSpeakerVAD] Feature extraction successful, feature dimension: {embs[0].shape if embs else 'None'}")
            print(f"[TwoSpeakerVAD] First few segment features (first 5 values): {[e[:5] if len(e) > 5 else e for e in embs[:3]]}")
            
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 标准化特征向量，提高聚类效果
            embs_array = np.stack(embs, axis=0)
            if embs_array.shape[0] > 2:  # 只有样本足够多才进行标准化
                scaler = StandardScaler()
                embs_scaled = scaler.fit_transform(embs_array)
            else:
                embs_scaled = embs_array
            
            # 使用更多的随机初始化来提高聚类稳定性
            best_labels = None
            best_inertia = float('inf')
            best_silhouette = -1
            
            # 多次运行K-means，选择轮廓系数最好的结果（轮廓系数衡量聚类质量）
            for random_state in [0, 42, 123, 456, 789]:
                km = KMeans(n_clusters=2, n_init=30, random_state=random_state, max_iter=1000)
                try:
                    y = km.fit_predict(embs_scaled)
                    # 计算轮廓系数（如果样本足够）
                    if len(embs_scaled) > 2:
                        from sklearn.metrics import silhouette_score
                        try:
                            silhouette = silhouette_score(embs_scaled, y)
                            # 优先选择轮廓系数更好的结果
                            if silhouette > best_silhouette:
                                best_silhouette = silhouette
                                best_labels = y
                                best_inertia = km.inertia_
                        except:
                            # 如果轮廓系数计算失败，使用惯性
                            if km.inertia_ < best_inertia:
                                best_inertia = km.inertia_
                                best_labels = y
                    else:
                        if km.inertia_ < best_inertia:
                            best_inertia = km.inertia_
                            best_labels = y
                except:
                    continue
            
            if best_labels is None:
                # 如果所有聚类都失败，使用改进的启发式方法
                # 基于频谱特征进行区分（男女声的主要区别）
                if embs_array.shape[1] >= 8:  # 确保有足够的特征
                    # 使用多个频谱特征的组合来区分男女声
                    # 频谱质心（索引4-5）、频谱滚降点（索引6-7）、多个滚降点（索引8-13）
                    
                    # 计算频谱质心特征（男女声差异明显）
                    centroid_feature = embs_array[:, 4] if embs_array.shape[1] > 4 else embs_array[:, 0]
                    
                    # 计算频谱滚降点特征（关键区分特征）
                    rolloff_feature = embs_array[:, 6] if embs_array.shape[1] > 6 else centroid_feature
                    
                    # 计算多个滚降点的组合特征
                    if embs_array.shape[1] > 13:
                        # 使用25%、50%、75%、90%滚降点的组合
                        combined_rolloff = (embs_array[:, 8] + embs_array[:, 10] + embs_array[:, 12]) / 3
                    else:
                        combined_rolloff = rolloff_feature
                    
                    # 组合频谱特征进行区分
                    combined_spectral = 0.4 * centroid_feature + 0.6 * combined_rolloff
                    median_score = np.median(combined_spectral)
                    best_labels = (combined_spectral > median_score).astype(int)
                    
                    print(f"[TwoSpeakerVAD] Using spectral-based heuristic: centroid+rolloff combination")
                else:
                    # 降级到简单的频谱质心方法
                    if embs_array.shape[1] >= 4:
                        centroid_scores = embs_array[:, 3] if embs_array.shape[1] > 3 else embs_array[:, 0]
                        median_score = np.median(centroid_scores)
                        best_labels = (centroid_scores > median_score).astype(int)
                        print(f"[TwoSpeakerVAD] Using basic centroid heuristic")
                    else:
                        # 降级到简单的顺序分配
                        best_labels = np.array([i % 2 for i in range(len(embs))])
                        print(f"[TwoSpeakerVAD] Using alternating assignment fallback")
            
            labels = list(map(int, best_labels.tolist()))
        # 处理只有一个说话人的情况
        if len(labels) == 1 or all(l == labels[0] for l in labels):
            # 所有段都属于同一个说话人，但如果有足够段数，尝试强制分离
            if len(labels) >= 2:
                print(f"[TwoSpeakerVAD] All segments classified as speaker {labels[0] if labels else 'unknown'}, forcing separation with alternating assignment")
                # 强制基于时间顺序交替分配标签
                labels = [i % 2 for i in range(len(labels))]
                print(f"[TwoSpeakerVAD] Forced alternating assignment: Speaker0={labels.count(0)}, Speaker1={labels.count(1)}")
            else:
                # 所有段都属于同一个说话人，无法分离
                print(f"[TwoSpeakerVAD] All segments classified as speaker {labels[0] if labels else 'unknown'}, cannot separate")
                return ([{"waveform": audio["waveform"], "sample_rate": sr}], 1, "Only one speaker detected, cannot separate")
        
        # 确保有两个说话人被识别
        unique_speakers = set(labels)
        if len(unique_speakers) < 2:
            print(f"[TwoSpeakerVAD] Only {len(unique_speakers)} speaker(s) detected, forcing 2 speakers...")
            # 强制创建两个说话人，基于时间或特征差异
            if len(labels) >= 2:
                # 基于时间顺序交替分配
                labels = [i % 2 for i in range(len(labels))]
                print(f"[TwoSpeakerVAD] Forced alternating assignment: {dict(zip(*np.unique(labels, return_counts=True)))}")
            else:
                return ([], 1, "insufficient segments for 2 speakers")
        
        # 输出聚类结果统计
        speaker_counts = [labels.count(0), labels.count(1)]
        print(f"[TwoSpeakerVAD] Speaker distribution: Speaker0={speaker_counts[0]}, Speaker1={speaker_counts[1]}")
        
        # 如果聚类结果极度不平衡（比如90%都在一个类别），尝试不同的聚类策略
        if max(speaker_counts) / len(labels) > 0.8:  # 放宽阈值到80%
            print(f"[TwoSpeakerVAD] Clustering too imbalanced ({max(speaker_counts)}/{len(labels)}), trying alternative approach")
            # 尝试基于时间顺序的交替分配，假设男女声会交替说话
            alt_labels = [i % 2 for i in range(len(segs))]
            # 只有在交替分配能改善平衡度时才使用
            alt_counts = [alt_labels.count(0), alt_labels.count(1)]
            if abs(alt_counts[0] - alt_counts[1]) < abs(speaker_counts[0] - speaker_counts[1]):
                labels = alt_labels
                print(f"[TwoSpeakerVAD] Using alternating assignment: Speaker0={alt_counts[0]}, Speaker1={alt_counts[1]}")
            else:
                # 尝试基于特征距离的简单阈值分割
                if len(embs) > 2 and embs[0].shape[0] > 0:
                    # 计算特征向量的主要变化方向
                    embs_array = np.stack(embs, axis=0)
                    if embs_array.shape[1] > 3:  # 确保有足够的特征维度
                        # 使用第一个主要特征维度进行排序和分割
                        primary_feature = embs_array[:, 3] if embs_array.shape[1] > 3 else embs_array[:, 0]
                        sorted_indices = np.argsort(primary_feature)
                        # 将排序后的段分成两组
                        threshold_labels = np.zeros(len(embs))
                        threshold_labels[sorted_indices[len(sorted_indices)//2:]] = 1
                        threshold_counts = [int(np.sum(threshold_labels == 0)), int(np.sum(threshold_labels == 1))]
                        # 如果这种分割更平衡，则使用它
                        if abs(threshold_counts[0] - threshold_counts[1]) < abs(speaker_counts[0] - speaker_counts[1]):
                            labels = threshold_labels.tolist()
                            print(f"[TwoSpeakerVAD] Using feature-based threshold assignment: Speaker0={threshold_counts[0]}, Speaker1={threshold_counts[1]}")
        
        mask_a = torch.zeros(total_16k, dtype=torch.float32)
        mask_b = torch.zeros(total_16k, dtype=torch.float32)
        for s, lab in zip(segs, labels):
            st = int(s["start"]) ; ed = int(s["end"]) ; ed = max(st, min(ed, total_16k))
            if lab == 0:
                mask_a[st:ed] = 1.0
            else:
                mask_b[st:ed] = 1.0
        def _smooth_mask(m: torch.Tensor, sr_hz: int, win_ms_f: float) -> torch.Tensor:
            L = m.numel()
            k_len = max(3, int((win_ms_f/1000.0) * sr_hz))
            k_len = min(k_len, max(3, L//4))
            if k_len <= 1:
                return m
            k = torch.ones(1,1,k_len) / float(k_len)
            x = m.view(1,1,L)
            pad = min(k_len//2, max(0, L-1))
            x = torch.nn.functional.pad(x, (pad, pad), mode="reflect")
            y = torch.nn.functional.conv1d(x, k)
            y = y.view(-1)
            if y.numel() > L:
                y = y[:L]
            return y
        mask_a = _smooth_mask(mask_a, 16000, smooth_ms)
        mask_b = _smooth_mask(mask_b, 16000, smooth_ms)
        if sr != 16000:
            up = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sr)
            mask_a = up(mask_a.unsqueeze(0)).squeeze(0)
            mask_b = up(mask_b.unsqueeze(0)).squeeze(0)
        orig = audio["waveform"][0] if audio["waveform"].dim() == 3 else audio["waveform"]
        if orig.dim() == 1:
            orig = orig.unsqueeze(0)
        
        # 确保掩码和原始音频的长度匹配
        orig_len = orig.shape[-1]
        mask_len = mask_a.shape[-1]
        if orig_len != mask_len:
            # 调整掩码长度以匹配原始音频
            if mask_len > orig_len:
                mask_a = mask_a[:orig_len]
                mask_b = mask_b[:orig_len]
            else:
                # 如果掩码较短，用0填充
                pad_len = orig_len - mask_len
                mask_a = torch.nn.functional.pad(mask_a, (0, pad_len), mode='constant', value=0)
                mask_b = torch.nn.functional.pad(mask_b, (0, pad_len), mode='constant', value=0)
        
        Ca, Cb = mask_a.unsqueeze(0).expand(orig.shape[0], -1), mask_b.unsqueeze(0).expand(orig.shape[0], -1)
        y1 = (Ca * orig).to("cpu")
        y2 = (Cb * orig).to("cpu")
        out_list = [
            {"waveform": y1.unsqueeze(0), "sample_rate": sr},
            {"waveform": y2.unsqueeze(0), "sample_rate": sr},
        ]
        info = "Two-speaker non-overlap separation via VAD+embedding clustering"
        return (out_list, 2, info)

NODE_CLASS_MAPPINGS = {
    "TwoSpeakerVADClusterNode": TwoSpeakerVADClusterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TwoSpeakerVADClusterNode": "lian Two-Speaker Separation (VAD+Cluster)",  
}