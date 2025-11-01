import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
import scipy.signal
from scipy.fft import fft, fftfreq

class AudioSplitNode:
    """
    Audio Split Node - Intelligently splits audio at silence points after target duration
    
    This node intelligently splits audio by finding silence points after a target duration,
    avoiding cutting in the middle of continuous audio segments.
    
    ComfyUI AUDIO structure compatibility:
    - Input: {"waveform": torch.Tensor[...], "sample_rate": int}
      Supports shapes [B, S] / [B, C, S] / [S], will be normalized to [B, C, S]
    - Output: {"waveform": torch.Tensor[N, C, L], "sample_rate": int}
      N: number of segments after splitting; C: channels; L: unified length (zero-padded to max segment length)
    
    Algorithm:
    - Adopt core technique from SingingVoiceSplitter_VAD_Simple:
      * Smooth absolute amplitude envelope
      * Silence detection by threshold
      * Remove short silences to avoid over-splitting
      * Detect speech segments and insert boundary corrections
      * Build split points at long silences
      * Greedy merge to ensure each output segment >= target duration
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node"""
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "输入音频对象，包含波形数据和采样率"
                }),
                "split_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 600.0,
                    "step": 0.1,
                    "tooltip": "目标分割时长（秒），达到此时长后寻找静音点进行分割"
                }),
            },
            "optional": {
                "min_silence_duration": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "最小静音持续时长（秒），低于此时长的静音将被忽略"
                }),
                "silence_threshold": ("FLOAT", {
                    "default": 0.008,
                    "min": 0.005,
                    "max": 0.05,
                    "step": 0.001,
                    "display": "slider",
                    "tooltip": "静音检测阈值（平均绝对振幅），低于此值认为是静音"
                }),
                "max_silence_search": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "达到目标时长后的最大静音搜索时长（秒）"
                }),
                "noise_reduction": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用噪声抑制预处理，提高在噪声环境下的分割精度"
                }),
                "adaptive_threshold": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用自适应阈值调整，根据音频特征动态调整静音检测阈值"
                }),
                "spectral_analysis": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用频谱分析，在噪声环境下提供更准确的静音检测"
                }),
                "high_precision": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用高精度模式，使用多频段分析提供最佳分割效果"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audio_list", "count")
    FUNCTION = "split"
    CATEGORY = "lianlaoshi/audio"

    def split(self, audio: Dict[str, Any], split_duration: float,
              min_silence_duration: float = 0.15,
              silence_threshold: float = 0.008,
              max_silence_search: float = 0.5,
              noise_reduction: bool = True,
              adaptive_threshold: bool = True,
              spectral_analysis: bool = False,
              high_precision: bool = False):
        """
        Split audio using VAD-like technique with enhanced noise handling and precision
        
        Args:
            audio: Input audio object {"waveform": torch.Tensor[...], "sample_rate": int}
            split_duration: Minimum duration for each output segment (seconds)
            min_silence_duration: Minimum silence required to create a split (seconds)
            silence_threshold: Amplitude threshold for silence detection
            max_silence_search: Unused in VAD mode (kept for compatibility)
            noise_reduction: Enable noise reduction preprocessing
            adaptive_threshold: Enable adaptive threshold adjustment
            spectral_analysis: Enable spectral analysis for better noise handling
            high_precision: Enable multi-band analysis for higher precision
            
        Returns:
            Tuple of (audio_list, segment_count)
        """
        # Validate input
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            return ([], 0)
        
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = int(audio["sample_rate"]) if audio["sample_rate"] is not None else 0
        
        if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
            return ([], 0)
    
        # Normalize to [B, C, S] format
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, S]
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)               # [B, 1, S]
        elif waveform.ndim == 3:
            pass
        else:
            return ([], 0)
    
        B, C, S = waveform.shape
        # Map parameters to VAD technique
        smoothing_window_ms = 30  # fixed smoothing window (ms)
        min_silence_gap = max(0.05, min(min_silence_duration * 0.3, 0.3))
        min_segment_duration = max(0.5, min(split_duration * 0.3, 10.0))
        target_min_duration = max(0.1, split_duration)
        
        all_segments: List[torch.Tensor] = []
    
        for b in range(B):
            W = waveform[b].contiguous().cpu()  # [C, S]
            W_np = W.numpy()
            total_samples = W_np.shape[-1]
            
            # Build mono reference for envelope
            y = W_np.mean(axis=0) if W_np.ndim > 1 else W_np
            max_val = np.abs(y).max() if total_samples > 0 else 1.0
            y = y / (max_val + 1e-8)
            
            # Enhanced preprocessing for noise reduction and precision
            if noise_reduction or spectral_analysis or high_precision:
                y = self._enhanced_preprocessing(y, sample_rate, noise_reduction, 
                                               spectral_analysis, high_precision)
            
            y_env = np.abs(y)
            
            # Adaptive threshold adjustment
            if adaptive_threshold:
                silence_threshold = self._adaptive_threshold_adjustment(y_env, silence_threshold)
            
            # High precision multi-band analysis
            if high_precision:
                y_env = self._multi_band_envelope(y, sample_rate)
            
            # 1) Enhanced smoothing with adaptive window
            if high_precision:
                window_size = max(int(15 * sample_rate / 1000), 3)  # Smaller window for precision
            else:
                window_size = max(int(smoothing_window_ms * sample_rate / 1000), 3)
            
            kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
            y_env_smooth = np.convolve(y_env, kernel, mode='same')
            
            # Edge fix with improved boundary handling
            if window_size <= len(y_env):
                head = np.convolve(y_env[:window_size], kernel, mode='valid')[0]
                tail = np.convolve(y_env[-window_size:], kernel, mode='valid')[-1]
                y_env_smooth[:window_size // 2] = head
                y_env_smooth[-(window_size // 2):] = tail
            
            # 2) Silence detection
            is_silence = y_env_smooth < float(silence_threshold)
            
            # 3) Remove short silences
            min_gap_samples = int(min_silence_gap * sample_rate)
            diff = np.diff(is_silence.astype(np.int8))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if is_silence[0]:
                starts = np.insert(starts, 0, 0)
            if is_silence[-1]:
                ends = np.append(ends, len(is_silence))
            for s_idx, e_idx in zip(starts, ends):
                if (e_idx - s_idx) < min_gap_samples:
                    is_silence[s_idx:e_idx] = False
            
            # 4) Speech segments from transitions
            diff2 = np.diff(is_silence.astype(np.int8))
            speech_starts = np.where(diff2 == -1)[0] + 1
            speech_ends = np.where(diff2 == 1)[0] + 1
            
            # Boundary corrections
            lookahead = int(0.3 * sample_rate)
            early_energy = float(np.mean(y_env_smooth[:min(lookahead, total_samples)])) if total_samples > 0 else 0.0
            if (len(speech_starts) == 0 or speech_starts[0] > lookahead) and (early_energy >= float(silence_threshold) * 0.7):
                speech_starts = np.insert(speech_starts, 0, 0)
            if not is_silence[0] and (len(speech_starts) == 0 or speech_starts[0] != 0):
                speech_starts = np.insert(speech_starts, 0, 0)
            if total_samples > 0 and not is_silence[-1]:
                speech_ends = np.append(speech_ends, total_samples)
            if len(speech_starts) > len(speech_ends):
                speech_ends = np.append(speech_ends, total_samples)
            elif len(speech_ends) > len(speech_starts):
                speech_starts = np.insert(speech_starts, 0, 0)
            
            # 5) Filter too-short segments
            min_speech_samples = int(min_segment_duration * sample_rate)
            raw_segments: List[Dict[str, int]] = []
            for st, ed in zip(speech_starts, speech_ends):
                if ed <= st:
                    continue
                if (ed - st) >= max(1, min_speech_samples):
                    raw_segments.append({"start": int(st), "end": int(ed)})
            if len(raw_segments) == 0:
                # Fallback: single segment full audio
                raw_segments.append({"start": 0, "end": total_samples})
            
            # 6) Split points at long silences between raw segments
            split_points: List[int] = []
            for i in range(len(raw_segments) - 1):
                p_start = raw_segments[i]["end"]
                p_end = raw_segments[i + 1]["start"]
                pause_samples = max(0, p_end - p_start)
                if pause_samples >= int(min_silence_duration * sample_rate):
                    cut_point = (p_start + p_end) // 2
                    split_points.append(int(cut_point))
            
            # 7) Build boundaries and temp segments
            boundaries = [0] + sorted(split_points) + [total_samples]
            temp_segments: List[torch.Tensor] = []
            for i in range(len(boundaries) - 1):
                st = boundaries[i]
                ed = boundaries[i + 1]
                if ed <= st:
                    continue
                seg = W[:, st:ed]  # [C, L]
                if seg.shape[1] > 0:
                    temp_segments.append(seg)
            if len(temp_segments) == 0:
                # No valid segments, keep full audio
                temp_segments.append(W)
            
            # 8) Greedy merge to satisfy min output duration
            merged: List[torch.Tensor] = []
            acc = None
            acc_len = 0
            for seg in temp_segments:
                if acc is None:
                    acc = seg
                    acc_len = seg.shape[1]
                else:
                    acc = torch.cat([acc, seg], dim=1)
                    acc_len = acc.shape[1]
                if (acc_len / sample_rate) >= target_min_duration:
                    merged.append(acc)
                    acc = None
                    acc_len = 0
            if acc is not None:
                if len(merged) > 0:
                    merged[-1] = torch.cat([merged[-1], acc], dim=1)
                else:
                    merged.append(acc)
            
            # Collect merged segments
            all_segments.extend(merged)
    
        # Build output
        if len(all_segments) == 0:
            return ([], 0)
        
        audio_list: List[Dict[str, Any]] = []
        for seg in all_segments:
            seg = seg.contiguous().cpu()
            audio_list.append({
                "waveform": seg.unsqueeze(0),  # [1, C, L]
                "sample_rate": sample_rate
            })
        
        return (audio_list, len(audio_list))

    def _enhanced_preprocessing(self, y: np.ndarray, sample_rate: int, 
                               noise_reduction: bool, spectral_analysis: bool, 
                               high_precision: bool) -> np.ndarray:
        """Enhanced preprocessing with noise reduction and spectral analysis"""
        if len(y) == 0:
            return y
            
        # Noise reduction using spectral subtraction
        if noise_reduction:
            # Estimate noise from first 0.5 seconds (assumed to contain noise)
            noise_samples = min(int(0.5 * sample_rate), len(y) // 4)
            if noise_samples > 0:
                noise_spectrum = np.abs(fft(y[:noise_samples]))
                noise_floor = np.mean(noise_spectrum)
                
                # Apply spectral subtraction
                Y = fft(y)
                Y_mag = np.abs(Y)
                Y_phase = np.angle(Y)
                
                # Subtract noise floor with over-subtraction factor
                alpha = 2.0  # Over-subtraction factor
                Y_mag_clean = Y_mag - alpha * noise_floor
                Y_mag_clean = np.maximum(Y_mag_clean, 0.1 * Y_mag)  # Prevent over-subtraction
                
                # Reconstruct signal
                Y_clean = Y_mag_clean * np.exp(1j * Y_phase)
                y = np.real(np.fft.ifft(Y_clean))
        
        # High-pass filter to remove low-frequency noise
        if spectral_analysis or high_precision:
            # Design high-pass filter (cutoff at 80 Hz)
            nyquist = sample_rate / 2
            cutoff = 80 / nyquist
            if cutoff < 0.99:  # Ensure valid cutoff
                b, a = scipy.signal.butter(4, cutoff, btype='high')
                y = scipy.signal.filtfilt(b, a, y)
        
        return y
    
    def _adaptive_threshold_adjustment(self, y_env: np.ndarray, base_threshold: float) -> float:
        """Adaptively adjust silence threshold based on audio characteristics"""
        if len(y_env) == 0:
            return base_threshold
            
        # Calculate statistics
        mean_energy = np.mean(y_env)
        std_energy = np.std(y_env)
        percentile_10 = np.percentile(y_env, 10)
        percentile_90 = np.percentile(y_env, 90)
        
        # Dynamic range
        dynamic_range = percentile_90 - percentile_10
        
        # Adjust threshold based on dynamic range and noise floor
        if dynamic_range > 0.1:  # High dynamic range
            # Use lower threshold for clear audio
            adjusted_threshold = max(base_threshold * 0.5, percentile_10 * 1.5)
        elif dynamic_range < 0.02:  # Low dynamic range (noisy)
            # Use higher threshold for noisy audio
            adjusted_threshold = min(base_threshold * 2.0, mean_energy * 0.3)
        else:
            # Moderate adjustment
            noise_factor = percentile_10 / (mean_energy + 1e-8)
            adjusted_threshold = base_threshold * (1 + noise_factor)
        
        return float(adjusted_threshold)
    
    def _multi_band_envelope(self, y: np.ndarray, sample_rate: int) -> np.ndarray:
        """Multi-band envelope analysis for higher precision"""
        if len(y) == 0:
            return np.abs(y)
            
        # Define frequency bands
        bands = [
            (80, 250),    # Low frequencies
            (250, 1000),  # Mid-low frequencies  
            (1000, 4000), # Mid-high frequencies
            (4000, 8000)  # High frequencies
        ]
        
        nyquist = sample_rate / 2
        band_envelopes = []
        
        for low, high in bands:
            if high > nyquist:
                high = nyquist * 0.95
            if low >= high:
                continue
                
            # Design bandpass filter
            low_norm = low / nyquist
            high_norm = high / nyquist
            
            try:
                b, a = scipy.signal.butter(4, [low_norm, high_norm], btype='band')
                y_band = scipy.signal.filtfilt(b, a, y)
                band_envelopes.append(np.abs(y_band))
            except:
                # Fallback if filter design fails
                band_envelopes.append(np.abs(y))
        
        if not band_envelopes:
            return np.abs(y)
        
        # Combine band envelopes with weighted average
        # Higher weight for mid frequencies (speech range)
        weights = [0.2, 0.4, 0.3, 0.1]
        weights = weights[:len(band_envelopes)]
        weights = np.array(weights) / np.sum(weights)
        
        combined_envelope = np.zeros_like(band_envelopes[0])
        for env, weight in zip(band_envelopes, weights):
            combined_envelope += weight * env
            
        return combined_envelope


class AudioEvenSplitNode:
    """
    Audio Even Split Node - Splits audio into equal duration segments
    
    This node splits audio into segments of equal duration based on the specified time interval.
    Unlike the intelligent split node, this performs simple time-based splitting without 
    considering silence or speech boundaries.
    
    ComfyUI AUDIO structure compatibility:
    - Input: {"waveform": torch.Tensor[...], "sample_rate": int}
      Supports shapes [B, S] / [B, C, S] / [S], will be normalized to [B, C, S]
    - Output: {"waveform": torch.Tensor[N, C, L], "sample_rate": int}
      N: number of segments after splitting; C: channels; L: segment length (last segment may be shorter)
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node"""
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "输入音频对象，包含波形数据和采样率"
                }),
                "segment_duration": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 600.0,
                    "step": 0.1,
                    "tooltip": "每个音频片段的时长（秒）"
                }),
            },
            "optional": {
                "overlap_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "音频片段之间的重叠时长（秒）。设置重叠可以避免在片段边界处丢失信息，但会产生重复的音频内容"
                }),
                "include_remainder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否包含最后一个不足指定时长的音频片段"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO_LIST", "INT")
    RETURN_NAMES = ("audio_list", "count")
    FUNCTION = "split_even"
    CATEGORY = "lianlaoshi/audio"

    def split_even(self, audio: Dict[str, Any], segment_duration: float,
                   overlap_duration: float = 0.0, include_remainder: bool = True):
        """
        Split audio into equal duration segments
        
        Args:
            audio: Input audio object {"waveform": torch.Tensor[...], "sample_rate": int}
            segment_duration: Duration of each segment in seconds
            overlap_duration: Overlap between consecutive segments in seconds
            include_remainder: Whether to include the last segment if shorter than segment_duration
            
        Returns:
            Tuple of (audio_list, segment_count)
        """
        # Validate input
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            return ([], 0)
        
        waveform: torch.Tensor = audio["waveform"]
        sample_rate: int = int(audio["sample_rate"]) if audio["sample_rate"] is not None else 0
        
        if not isinstance(waveform, torch.Tensor) or sample_rate <= 0:
            return ([], 0)
    
        # Normalize to [B, C, S] format
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, S]
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)               # [B, 1, S]
        elif waveform.ndim == 3:
            pass
        else:
            return ([], 0)
    
        B, C, S = waveform.shape
        
        # Convert durations to samples
        segment_samples = int(segment_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        # Validate parameters
        if segment_samples <= 0:
            return ([], 0)
        
        if overlap_samples >= segment_samples:
            overlap_samples = segment_samples - 1
        
        step_samples = segment_samples - overlap_samples
        
        all_segments: List[torch.Tensor] = []
    
        for b in range(B):
            W = waveform[b].contiguous().cpu()  # [C, S]
            total_samples = W.shape[-1]
            
            # Calculate segment positions
            segments: List[torch.Tensor] = []
            start_pos = 0
            
            while start_pos < total_samples:
                end_pos = min(start_pos + segment_samples, total_samples)
                
                # Extract segment
                segment = W[:, start_pos:end_pos]  # [C, L]
                
                # Check if we should include this segment
                segment_length = segment.shape[1]
                if segment_length > 0:
                    if include_remainder or segment_length == segment_samples:
                        segments.append(segment)
                
                # Move to next position
                start_pos += step_samples
                
                # Break if we've reached the end and the last segment was full length
                if end_pos >= total_samples:
                    break
            
            # Add segments from this batch
            all_segments.extend(segments)
    
        # Build output
        if len(all_segments) == 0:
            return ([], 0)
        
        audio_list: List[Dict[str, Any]] = []
        for seg in all_segments:
            seg = seg.contiguous().cpu()
            audio_list.append({
                "waveform": seg.unsqueeze(0),  # [1, C, L]
                "sample_rate": sample_rate
            })
        
        return (audio_list, len(audio_list))


# Node registration
NODE_CLASS_MAPPINGS = {
    "AudioSplitNode": AudioSplitNode,
    "AudioEvenSplitNode": AudioEvenSplitNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSplitNode": "lian 智能音频分割节点",
    "AudioEvenSplitNode": "lian 音频均分分割节点",
}