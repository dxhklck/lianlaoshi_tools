import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List

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
                    "tooltip": "Input audio object containing waveform and sample rate"
                }),
                "split_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.1,
                    "max": 600.0,
                    "step": 0.1,
                    "tooltip": "Target duration in seconds before looking for silence"
                }),
            },
            "optional": {
                "min_silence_duration": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.05,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Minimum duration required to consider a segment as silence"
                }),
                "silence_threshold": ("FLOAT", {
                    "default": 0.008,
                    "min": 0.005,
                    "max": 0.05,
                    "step": 0.001,
                    "display": "slider",
                    "tooltip": "Silence detection threshold (average absolute amplitude)"
                }),
                "max_silence_search": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Maximum search duration after target time to find silence"
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
              max_silence_search: float = 0.5):
        """
        Split audio using VAD-like technique adapted from SingingVoiceSplitter_VAD_Simple
        while preserving this node's output contract.
        
        Args:
            audio: Input audio object {"waveform": torch.Tensor[...], "sample_rate": int}
            split_duration: Minimum duration for each output segment (seconds)
            min_silence_duration: Minimum silence required to create a split (seconds)
            silence_threshold: Amplitude threshold for silence detection
            max_silence_search: Unused in VAD mode (kept for compatibility)
            
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
            y_env = np.abs(y)
            
            # 1) Smoothing (moving average)
            window_size = max(int(smoothing_window_ms * sample_rate / 1000), 3)
            kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
            y_env_smooth = np.convolve(y_env, kernel, mode='same')
            # Edge fix similar to reference implementation
            if window_size <= len(y_env):
                head = np.convolve(y_env[:window_size], kernel, mode='valid')[0]
                y_env_smooth[:window_size // 2] = head
            
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


# Node registration
NODE_CLASS_MAPPINGS = {
    "AudioSplitNode": AudioSplitNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSplitNode": "lian 智能音频分割节点",
}