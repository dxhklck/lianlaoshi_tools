"""
Whisper 音频处理节点
版本: 1.1.0
更新日志:
- 修复了WhisperTranscribeNode中_add_internal_punctuation方法的正则表达式错误（不平衡括号）
- 改进了WhisperVADSplitNode的模型加载逻辑，添加了详细的日志输出和更完善的错误处理
- 增强了模型下载功能，添加了进度显示和备选目录支持
- 修复了代码中多余的异常跟踪语句
"""
import os
import time
import torch
import numpy as np
import soundfile as sf
from scipy import signal
# 导入Whisper相关模块
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

class WhisperVADSplitNode:
    """
    智能音频分段节点，基于VAD技术检测静音并在静音中部进行切割，
    支持按参考时长合并相邻分段以减少分段数量。
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),  # 输入音频
                "min_silence_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 2.0, "step": 0.1}),  # 最小静音时长(秒)
                "vad_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1}),  # VAD检测阈值
                "window_ms": ("INT", {"default": 25, "min": 10, "max": 50, "step": 5}),  # VAD窗口大小(毫秒)
                "hop_ms": ("INT", {"default": 10, "min": 5, "max": 20, "step": 1}),  # VAD步长(毫秒)
                "min_segment_duration": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),  # 最小分段时长(秒)
                "min_split_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),  # 最小切割长度(秒)
                "target_segment_duration": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 15.0, "step": 0.5}),  # 目标分段时长(秒)
                "vad_model_choice": (["silero-vad.jit"],),  # VAD模型选择
            },
        }
    
    RETURN_TYPES = ("AUDIO", "INT", "AUDIO_LIST")
    RETURN_NAMES = ("audio_segments", "count", "audio_segments_list")
    
    FUNCTION = "split"
    CATEGORY = "lianlaoshi/audio"
    
    def __init__(self):
        print(f"[VAD] 🔄 初始化WhisperVADSplitNode节点...")
        self.vad_model = None
        self.sample_rate = 16000  # VAD模型通常使用16kHz采样率
        # 记录初始化时间用于性能分析
        self.init_time = time.time()
        self._load_vad_model()
        print(f"[VAD] ✅ WhisperVADSplitNode节点初始化完成，耗时: {time.time() - self.init_time:.2f}秒")
    
    def _load_vad_model(self):
        """
        加载VAD模型，从ComfyUI的models/whisper目录加载，如果不存在则下载
        """
        print(f"[VAD] 📂 开始加载VAD模型...")
        start_time = time.time()
        
        try:
            # 检查CUDA可用性
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            device = "cuda" if has_cuda else "mps" if has_mps else "cpu"
            print(f"[VAD] 💻 设备信息: CUDA={has_cuda}, MPS={has_mps}, 将使用: {device}")
            
            # 首先尝试从ComfyUI的models/whisper目录加载模型
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "whisper")
            model_path = os.path.join(models_dir, "silero-vad.jit")
            print(f"[VAD] 🔍 尝试从ComfyUI模型目录加载: {model_path}")
            
            # 确保路径计算正确
            print(f"[VAD] 📍 当前文件路径: {os.path.abspath(__file__)}")
            print(f"[VAD] 📁 模型目录路径: {os.path.abspath(models_dir)}")
            
            # 检查目录是否存在
            if not os.path.exists(models_dir):
                print(f"[VAD] ⚠️ 模型目录不存在: {models_dir}")
            
            # 检查模型文件是否存在
            if os.path.exists(model_path):
                print(f"[VAD] 发现模型文件，正在加载...")
                try:
                    self.vad_model = torch.jit.load(model_path)
                    self.vad_model.eval()
                    print(f"[VAD] ✅ 成功加载模型: {model_path}")
                    return
                except Exception as load_error:
                    print(f"[VAD] ❌ 模型文件存在但加载失败: {str(load_error)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[VAD] ⚠️ 模型文件不存在: {model_path}")
            
            # 确保models/whisper目录存在
            print(f"[VAD] 准备创建模型目录（如果不存在）...")
            try:
                os.makedirs(models_dir, exist_ok=True)
                print(f"[VAD] ✅ 模型目录已确保存在: {models_dir}")
            except Exception as dir_error:
                print(f"[VAD] ❌ 创建模型目录失败: {str(dir_error)}")
                import traceback
                traceback.print_exc()
                
                # 尝试使用备选目录
                alt_dir = os.path.join(os.path.dirname(__file__), "models")
                print(f"[VAD] 🔄 尝试使用备选目录: {alt_dir}")
                try:
                    os.makedirs(alt_dir, exist_ok=True)
                    models_dir = alt_dir
                    model_path = os.path.join(models_dir, "silero-vad.jit")
                    print(f"[VAD] ✅ 备选目录创建成功，将使用: {model_path}")
                except Exception as alt_dir_error:
                    print(f"[VAD] ❌ 备选目录创建也失败: {str(alt_dir_error)}")
                    traceback.print_exc()
            
            # 下载模型
            print(f"[VAD] 📥 准备下载模型到: {model_path}")
            try:
                import requests
                import shutil
                model_url = "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.jit"
                print(f"[VAD] 📥 正在从官方GitHub下载silero-vad.jit模型...")
                print(f"[VAD] 📥 下载地址: {model_url}")
                print(f"[VAD] 💡 提示：下载可能需要一些时间，请耐心等待...")
                
                # 使用streaming下载大文件
                with requests.get(model_url, stream=True, timeout=30) as r:
                    r.raise_for_status()  # 检查下载是否成功
                    total_size = int(r.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(model_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                # 显示下载进度
                                if total_size > 0:
                                    progress = (downloaded_size / total_size) * 100
                                    print(f"[VAD] 📊 下载进度: {progress:.1f}%")
                
                print(f"[VAD] ✅ 模型下载完成: {model_path}")
                print(f"[VAD] 📁 下载文件大小: {os.path.getsize(model_path)/1024:.1f} KB")
                
                # 加载下载的模型
                print(f"[VAD] 📂 正在加载下载的模型...")
                self.vad_model = torch.jit.load(model_path)
                self.vad_model.eval()
                print(f"[VAD] ✅ 成功加载下载的模型")
            
            except requests.exceptions.RequestException as req_error:
                print(f"[VAD] ❌ 网络请求失败: {str(req_error)}")
                print(f"[VAD] 💡 请检查网络连接是否正常")
                import traceback
                traceback.print_exc()
            except Exception as download_error:
                print(f"[VAD] ❌ 模型下载失败: {str(download_error)}")
                import traceback
                traceback.print_exc()
                print(f"[VAD] 💡 请手动下载模型文件并放到: {model_path}")
                print(f"[VAD] 💡 下载地址: {model_url}")
                
                # 尝试从当前目录加载作为备选
                fallback_path = os.path.join(os.path.dirname(__file__), "silero-vad.jit")
                print(f"[VAD] 🔄 尝试从当前目录加载备选模型: {fallback_path}")
                if os.path.exists(fallback_path):
                    try:
                        self.vad_model = torch.jit.load(fallback_path)
                        self.vad_model.eval()
                        print(f"[VAD] ✅ 成功加载备选模型")
                    except Exception as fallback_error:
                        print(f"[VAD] ❌ 备选模型加载失败: {str(fallback_error)}")
                else:
                    print(f"[VAD] ❌ 备选模型文件不存在: {fallback_path}")
        
        except Exception as e:
            print(f"[VAD] ❌ 加载模型过程中发生未知错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
        # 最终检查模型是否成功加载
        if self.vad_model is None:
            print(f"[VAD] ⚠️ 警告：模型加载失败，将在需要时尝试重新加载")
        else:
            print(f"[VAD] ✅ 模型加载状态检查通过")
    
    def split(self, audio, min_silence_duration, vad_threshold, window_ms, hop_ms, 
              min_segment_duration, min_split_duration, target_segment_duration, vad_model_choice):
        """
        主函数：对音频进行分段处理
        """
        print(f"[VAD] 🎯 开始音频分段处理")
        start_time = time.time()
        
        # 详细打印所有参数
        print(f"[VAD] ⚙️ 参数设置:")
        print(f"[VAD]    - 最小静音时长: {min_silence_duration}秒")
        print(f"[VAD]    - VAD阈值: {vad_threshold}")
        print(f"[VAD]    - 窗口大小: {window_ms}毫秒")
        print(f"[VAD]    - 步长大小: {hop_ms}毫秒")
        print(f"[VAD]    - 最小分段时长: {min_segment_duration}秒")
        print(f"[VAD]    - 最小切割长度: {min_split_duration}秒")
        print(f"[VAD]    - 目标分段时长: {target_segment_duration}秒")
        print(f"[VAD]    - VAD模型: {vad_model_choice}")
        
        # 打印输入音频的基本信息
        print(f"[VAD] 📥 输入音频类型: {type(audio)}")
        
        try:
            # 参数验证和音频数据提取
            audio_np = None
            original_sample_rate = 16000
            
            try:
                # 尝试处理音频数据，支持多种格式
                if isinstance(audio, dict):
                    # 假设字典格式为 {'waveform': [...], 'sample_rate': ...} 或类似结构
                    if 'waveform' in audio and 'sample_rate' in audio:
                        audio_np = audio['waveform'].numpy() if isinstance(audio['waveform'], torch.Tensor) else audio['waveform']
                        original_sample_rate = audio['sample_rate']
                    else:
                        # 尝试获取第一个可能的音频数据和采样率
                        for key, value in audio.items():
                            if isinstance(value, (torch.Tensor, list, np.ndarray)):
                                audio_np = value.numpy() if isinstance(value, torch.Tensor) else value
                                # 尝试找到采样率
                                if 'rate' in key.lower():
                                    original_sample_rate = value
                                elif 'sample_rate' in audio:
                                    original_sample_rate = audio['sample_rate']
                                break
                elif hasattr(audio, '__len__') and len(audio) >= 2:
                    # 原始格式：列表/元组 (audio_data, sample_rate)
                    audio_np = audio[0].numpy() if isinstance(audio[0], torch.Tensor) else audio[0]
                    original_sample_rate = audio[1]
                else:
                    print(f"[VAD] 错误：无法识别音频格式")
                    return (audio, ([torch.tensor([]) if audio_np is None else torch.tensor(audio_np)], original_sample_rate), 1)
                
                if audio_np is None:
                    print(f"[VAD] 错误：无法从音频输入中提取波形数据")
                    return (audio, ([torch.tensor([])], original_sample_rate), 1)
                    
                print(f"[VAD] 音频格式: 采样率={original_sample_rate}Hz, 长度={len(audio_np)}样本")
                print(f"[VAD] 音频时长: {len(audio_np)/original_sample_rate:.2f}秒")
            except Exception as e:
                print(f"[VAD] 处理音频格式时出错: {str(e)}")
                # 使用默认值返回
                return (audio, ([torch.tensor([])], 16000), 1)
            
            # 验证音频长度，对于极短音频直接返回
            print(f"[VAD] 📏 音频长度验证: {len(audio_np)}样本, 时长={len(audio_np)/original_sample_rate:.3f}秒")
            if len(audio_np) <= 1:
                print(f"[VAD] ⚠️ 警告：音频长度过短({len(audio_np)}样本)，无法进行有效分段，直接返回")
                return (([torch.tensor(audio_np)], original_sample_rate), 1)
            elif len(audio_np) < int(min_segment_duration * original_sample_rate):
                print(f"[VAD] ⚠️ 警告：音频长度小于最小分段时长，可能无法有效分段")
            
            # 如果采样率不是16kHz，重采样到16kHz用于VAD检测
            try:
                if original_sample_rate != self.sample_rate:
                    print(f"[VAD] 重采样音频从 {original_sample_rate}Hz 到 {self.sample_rate}Hz")
                    audio_16k = self._resample_audio(audio_np, original_sample_rate, self.sample_rate)
                else:
                    audio_16k = audio_np.copy()
                    print(f"[VAD] 采样率已为 {self.sample_rate}Hz，无需重采样")
            except Exception as e:
                print(f"[VAD] 音频重采样失败: {str(e)}")
                # 使用原始音频继续处理，而不是直接返回
                audio_16k = audio_np.copy()
            
            # 使用VAD检测静音段
            print(f"[VAD] 🔊 开始VAD语音检测，阈值={vad_threshold}")
            vad_start_time = time.time()
            speech_probs = self._detect_speech(audio_16k, vad_threshold, window_ms, hop_ms)
            vad_duration = time.time() - vad_start_time
            
            # 语音概率统计
            if len(speech_probs) > 0:
                speech_ratio = np.mean(speech_probs > vad_threshold) * 100
                avg_prob = np.mean(speech_probs)
                max_prob = np.max(speech_probs)
                min_prob = np.min(speech_probs)
                print(f"[VAD] ✅ VAD检测完成")
                print(f"[VAD]    - 生成 {len(speech_probs)} 个语音概率点")
                print(f"[VAD]    - 语音比例: {speech_ratio:.1f}%")
                print(f"[VAD]    - 概率统计: 平均={avg_prob:.3f}, 最大={max_prob:.3f}, 最小={min_prob:.3f}")
                print(f"[VAD]    - 检测耗时: {vad_duration:.3f}秒")
            else:
                print(f"[VAD] ⚠️ VAD检测生成空结果，可能音频过短")
            
            # 识别静音段并在中部切割
            print(f"[VAD] ✂️ 开始静音段识别和中部切割")
            split_start_time = time.time()
            segments = self._split_at_silence_middle(audio_16k, speech_probs, min_silence_duration, 
                                                    min_segment_duration, min_split_duration, window_ms, hop_ms)
            split_duration = time.time() - split_start_time
            print(f"[VAD] ✅ 静音中部切割完成，耗时: {split_duration:.3f}秒")
            
            # 按目标时长合并相邻分段
            print(f"[VAD] 开始按目标时长({target_segment_duration}秒)合并分段")
            merged_segments = self._merge_segments_by_duration(segments, target_segment_duration)
            print(f"[VAD] 合并完成，合并后段数: {len(merged_segments)}(原始段数: {len(segments)})")
            
            # 打印合并后的分段信息
            for i, seg in enumerate(merged_segments):
                print(f"[VAD] 合并后分段 {i+1}: {len(seg)/self.sample_rate:.2f}秒 ({len(seg)}样本)")
            
            # 确保所有分段总时长等于原音频
            self._validate_total_duration(merged_segments, len(audio_16k), self.sample_rate)
            
            # 如果需要，将分段重采样回原始采样率
            final_segments = []
            try:
                for i, segment in enumerate(merged_segments):
                    if original_sample_rate != self.sample_rate:
                        resampled = self._resample_audio(segment, self.sample_rate, original_sample_rate)
                        final_segments.append(torch.tensor(resampled))
                        print(f"[VAD] 重采样分段 {i+1} 回原始采样率 {original_sample_rate}Hz")
                    else:
                        final_segments.append(torch.tensor(segment))
            except Exception as e:
                print(f"[VAD] 分段重采样失败: {str(e)}")
                # 使用已提取的音频数据和采样率
                return (audio, ([torch.tensor(audio_np) if audio_np is not None else torch.tensor([])], original_sample_rate), 1)
            
            print(f"[VAD] 音频分段处理成功完成，返回 {len(final_segments)} 个分段")
            # 返回结果：分段的语音(AUDIO格式)、数量和分段列表(AUDIO_LIST格式)
            # 将final_segments转换为张量并创建正确的字典格式
            segments_tensor = torch.stack(final_segments) if final_segments else torch.empty(0)
            audio_dict = {"waveform": segments_tensor, "sample_rate": original_sample_rate}
            return (audio_dict, len(final_segments), final_segments)
            
        except Exception as e:
            print(f"[VAD] ❌ 分段处理失败: {str(e)}")
            import traceback
            print(f"[VAD] 📋 错误堆栈:")
            traceback.print_exc()
            print(f"[VAD] 🔄 启用安全模式，返回原始音频作为单个分段")
            # 失败时返回安全的默认值
            safe_segments = [torch.tensor(audio_np) if audio_np is not None else torch.tensor([])]
            # 创建正确的字典格式
            segments_tensor = torch.stack(safe_segments) if safe_segments else torch.empty(0)
            audio_dict = {"waveform": segments_tensor, "sample_rate": original_sample_rate}
            return (audio_dict, 1, safe_segments)
            
        finally:
            total_duration = time.time() - start_time
            print(f"[VAD] ⏱️ 音频分段处理总耗时: {total_duration:.3f}秒")
    
    def _detect_speech(self, audio, threshold, window_ms, hop_ms):
        """
        使用VAD模型检测语音概率
        """
        try:
            if self.vad_model is None:
                print("[VAD] 📊 未加载VAD模型，使用振幅包络检测作为备选")
                return self._amplitude_envelope_detection(audio, threshold, window_ms, hop_ms)
            
            # 将音频转换为torch张量
            audio_tensor = torch.tensor(audio).unsqueeze(0)
            
            # 计算窗口和步长的样本数
            window_size = int(window_ms * self.sample_rate / 1000)
            hop_size = int(hop_ms * self.sample_rate / 1000)
            
            # 检测语音概率
            speech_probs = []
            for i in range(0, len(audio) - window_size, hop_size):
                segment = audio_tensor[:, i:i+window_size]
                with torch.no_grad():
                    prob = self.vad_model(segment, self.sample_rate).item()
                speech_probs.append(prob)
            
            return np.array(speech_probs)
        except Exception as e:
            print(f"[VAD] 语音检测失败: {str(e)}")
            return self._amplitude_envelope_detection(audio, threshold, window_ms, hop_ms)
    
    def _amplitude_envelope_detection(self, audio, threshold, window_ms, hop_ms):
        """
        备用方法：使用振幅包络检测语音
        """
        window_size = int(window_ms * self.sample_rate / 1000)
        hop_size = int(hop_ms * self.sample_rate / 1000)
        
        # 计算振幅包络
        amplitude_envelope = []
        for i in range(0, len(audio) - window_size, hop_size):
            segment = audio[i:i+window_size]
            envelope = np.max(np.abs(segment))
            amplitude_envelope.append(envelope)
        
        # 归一化
        amplitude_envelope = np.array(amplitude_envelope)
        # 处理空数组的情况
        if len(amplitude_envelope) == 0:
            return np.array([])
        max_amplitude = np.max(amplitude_envelope) if np.max(amplitude_envelope) > 0 else 1
        normalized_envelope = amplitude_envelope / max_amplitude
        
        # 转换为类似VAD概率的值
        return normalized_envelope
    
    def _split_at_silence_middle(self, audio, speech_probs, min_silence_duration, 
                               min_segment_duration, min_split_duration, window_ms, hop_ms):
        """
        在静音段的中部进行切割
        """
        try:
            # 计算所有需要的参数
            window_size = int(window_ms * self.sample_rate / 1000)
            hop_size = int(hop_ms * self.sample_rate / 1000)
            min_silence_samples = int(min_silence_duration * self.sample_rate)
            min_segment_samples = int(min_segment_duration * self.sample_rate)
            min_split_samples = int(min_split_duration * self.sample_rate)
            
            print(f"[VAD] 📐 静音切割参数计算:")
            print(f"[VAD]    - 窗口大小: {window_size}样本 ({window_ms}毫秒)")
            print(f"[VAD]    - 步长大小: {hop_size}样本 ({hop_ms}毫秒)")
            print(f"[VAD]    - 最小静音长度: {min_silence_samples}样本 ({min_silence_duration}秒)")
            print(f"[VAD]    - 最小分段长度: {min_segment_samples}样本 ({min_segment_duration}秒)")
            print(f"[VAD]    - 最小切割长度: {min_split_samples}样本 ({min_split_duration}秒)")
            
            # 检查语音概率数组是否为空
            if len(speech_probs) == 0:
                print(f"[VAD] ⚠️ 语音概率数组为空，无法进行分段")
                return [audio]
            
            # 识别静音段
            print(f"[VAD] 🔍 开始静音段识别...")
            is_speech = speech_probs > 0.5  # 0.5是VAD判决阈值
            speech_count = np.sum(is_speech)
            silence_count = len(is_speech) - speech_count
            print(f"[VAD]    - 语音帧数: {speech_count}, 静音帧数: {silence_count}")
            
            # 转换为原始音频采样点对应的静音标记
            print(f"[VAD] 🔄 转换语音标记到原始音频采样点...")
            sample_speech_flags = np.zeros(len(audio), dtype=bool)
            for i, flag in enumerate(is_speech):
                try:
                    start_idx = i * hop_size
                    end_idx = min(start_idx + window_size, len(audio))
                    sample_speech_flags[start_idx:end_idx] = flag
                except Exception as e:
                    print(f"[VAD] ❌ 处理音频帧 {i} 时出错: {str(e)}")
            
            # 找出静音段的起始和结束位置
            print(f"[VAD] 🎯 查找符合条件的静音段...")
            silence_segments = []
            in_silence = False
            silence_start = 0
            total_silence_duration = 0
            
            for i, is_speaking in enumerate(sample_speech_flags):
                if not is_speaking and not in_silence:
                    in_silence = True
                    silence_start = i
                elif is_speaking and in_silence:
                    in_silence = False
                    silence_end = i
                    silence_length = silence_end - silence_start
                    total_silence_duration += silence_length
                    
                    # 检查静音段长度
                    if silence_length >= min_silence_samples:
                        # 确保静音段前后有足够长的语音
                        prev_speech_end = silence_start
                        next_speech_start = silence_end
                        
                        # 检查前面的语音段
                        j = silence_start - 1
                        while j >= 0 and sample_speech_flags[j]:
                            j -= 1
                        prev_speech_start = j + 1
                        
                        # 计算前面语音段的长度
                        prev_speech_length = prev_speech_end - prev_speech_start
                        
                        # 检查后面的语音段
                        j = silence_end
                        while j < len(sample_speech_flags) and sample_speech_flags[j]:
                            j += 1
                        next_speech_end = j
                        next_speech_length = next_speech_end - next_speech_start
                        
                        # 只在前后都有足够长语音段的情况下添加此静音段作为切割点
                        if prev_speech_length >= min_split_samples and next_speech_length >= min_split_samples:
                            silence_segments.append({
                                'start': silence_start,
                                'end': silence_end,
                                'middle': (silence_start + silence_end) // 2,
                                'length': silence_length
                            })
                            print(f"[VAD]    - 找到有效静音段: 位置={silence_start}-{silence_end}, 长度={silence_length/self.sample_rate:.3f}秒")
                        else:
                            print(f"[VAD]    - 跳过静音段: 前后语音段不足 ({prev_speech_length/self.sample_rate:.3f}秒/{next_speech_length/self.sample_rate:.3f}秒)")
                    else:
                        print(f"[VAD]    - 跳过短静音段: 长度={silence_length/self.sample_rate:.3f}秒 < {min_silence_duration}秒")
            
            # 打印静音统计
            total_silence_seconds = total_silence_duration / self.sample_rate
            audio_duration = len(audio) / self.sample_rate
            silence_ratio = (total_silence_duration / len(audio) * 100) if len(audio) > 0 else 0
            print(f"[VAD] 📊 静音统计:")
            print(f"[VAD]    - 音频总时长: {audio_duration:.3f}秒")
            print(f"[VAD]    - 静音总时长: {total_silence_seconds:.3f}秒 ({silence_ratio:.1f}%)")
            print(f"[VAD]    - 有效切割静音段: {len(silence_segments)}个")
            
            print(f"[VAD] 找到 {len(silence_segments)} 个符合条件的静音段")
            
            # 根据静音段切割音频
            segments = []
            last_end = 0
            
            # 添加第一个语音段
            if silence_segments:
                first_silence = silence_segments[0]
                first_segment = audio[last_end:first_silence['middle']]
                if len(first_segment) >= min_segment_samples:
                    segments.append(first_segment)
                    last_end = first_silence['middle']
                    print(f"[VAD] 添加第一个语音段: {len(first_segment)/self.sample_rate:.2f}秒")
            
            # 添加中间的语音段
            for i in range(1, len(silence_segments)):
                try:
                    prev_silence = silence_segments[i-1]
                    curr_silence = silence_segments[i]
                    segment = audio[last_end:curr_silence['middle']]
                    if len(segment) >= min_segment_samples:
                        segments.append(segment)
                        last_end = curr_silence['middle']
                        print(f"[VAD] 添加中间语音段 {i}: {len(segment)/self.sample_rate:.2f}秒")
                except Exception as e:
                    print(f"[VAD] 处理中间分段时出错: {str(e)}")
            
            # 添加最后一个语音段
            last_segment = audio[last_end:]
            if len(last_segment) >= min_segment_samples:
                segments.append(last_segment)
                print(f"[VAD] 添加最后一个语音段: {len(last_segment)/self.sample_rate:.2f}秒")
            
            # 如果没有找到合适的分段，返回整个音频
            if not segments:
                print(f"[VAD] 未找到合适的分段，返回完整音频")
                segments = [audio]
            
            print(f"[VAD] 静音中部切割完成，共生成 {len(segments)} 个分段")
            return segments
        except Exception as e:
            print(f"[VAD] 静音中部切割过程中出错: {str(e)}")
            # 出错时返回原始音频作为单个分段
            return [audio]
    
    def _merge_segments_by_duration(self, segments, target_duration):
        """
        增强版分段合并逻辑 - 智能合并分段以提高长音频分段质量
        考虑分段边界、静音位置和时长一致性，确保合并后的分段更加自然
        """
        print(f"[VAD] 🔄 开始智能分段合并处理...")
        merge_start_time = time.time()
        
        if not segments or len(segments) <= 1:
            print(f"[VAD]    - 分段数量不足({len(segments)}个)，无需合并")
            return segments
        
        # 计算目标时长的样本数
        target_samples = int(target_duration * self.sample_rate)
        # 设置更合理的误差范围，根据目标时长动态调整
        min_target_samples = max(0, target_samples - int(0.8 * self.sample_rate))  # 最小目标时长，放宽到-0.8秒
        max_target_samples = target_samples + int(1.2 * self.sample_rate)  # 最大目标时长，允许到+1.2秒
        
        print(f"[VAD] 📏 合并参数:")
        print(f"[VAD]    - 目标时长: {target_duration}秒 ({target_samples}样本)")
        print(f"[VAD]    - 误差范围: {min_target_samples/self.sample_rate:.1f}秒 到 {max_target_samples/self.sample_rate:.1f}秒")
        
        # 预处理：计算所有分段的长度和累计长度
        segment_lengths = [len(seg) for seg in segments]
        total_length = sum(segment_lengths)
        avg_segment_length = total_length / len(segments)
        
        print(f"[VAD] 📊 分段合并前统计:")
        print(f"[VAD]    - 分段总数: {len(segments)}个")
        print(f"[VAD]    - 总时长: {total_length/self.sample_rate:.2f}秒")
        print(f"[VAD]    - 平均长度: {avg_segment_length/self.sample_rate:.2f}秒")
        print(f"[VAD]    - 长度分布: 最小={min(segment_lengths)/self.sample_rate:.2f}秒, 最大={max(segment_lengths)/self.sample_rate:.2f}秒")
        
        merged = []
        current_segment = segments[0].copy()
        current_length = segment_lengths[0]
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            next_length = segment_lengths[i]
            combined_length = current_length + next_length
            
            # 增强策略1：优先保持适度长度的分段，避免过短或过长
            if combined_length > 2 * target_samples:
                # 如果合并后太长，保存当前分段，开始新的
                merged.append(current_segment)
                current_segment = next_segment.copy()
                current_length = next_length
            # 增强策略2：如果当前段已经在理想范围内，保存当前分段
            elif min_target_samples <= current_length <= max_target_samples:
                merged.append(current_segment)
                current_segment = next_segment.copy()
                current_length = next_length
            # 增强策略3：如果当前段太短，即使合并后超过目标范围也合并
            elif current_length < min_target_samples * 0.7:  # 更严格的短分段判断
                current_segment = np.concatenate([current_segment, next_segment])
                current_length = combined_length
            # 增强策略4：平衡策略，尝试接近目标时长
            elif combined_length <= max_target_samples:
                # 如果合并后在最大范围内，合并
                current_segment = np.concatenate([current_segment, next_segment])
                current_length = combined_length
            else:
                # 否则保存当前分段
                merged.append(current_segment)
                current_segment = next_segment.copy()
                current_length = next_length
        
        # 添加最后一个分段
        merged.append(current_segment)
        
        # 后处理优化1：修复所有过短的分段
        print(f"[VAD] 🔧 开始后处理优化...")
        short_segment_count = sum(1 for seg in merged if len(seg) < min_target_samples * 0.7)
        print(f"[VAD]    - 发现过短分段数: {short_segment_count}个 (<{min_target_samples * 0.7 / self.sample_rate:.1f}秒)")
        
        i = 0
        merge_count = 0
        while i < len(merged) - 1:
            curr_len = len(merged[i])
            next_len = len(merged[i+1])
            
            # 如果当前分段太短，尝试与下一个合并
            if curr_len < min_target_samples * 0.7:
                print(f"[VAD]    - 合并过短分段 {i}: {curr_len/self.sample_rate:.2f}秒 + {i+1}: {next_len/self.sample_rate:.2f}秒")
                merged[i] = np.concatenate([merged[i], merged[i+1]])
                merged.pop(i+1)
                merge_count += 1
            # 如果下一个分段太短，尝试与当前合并
            elif next_len < min_target_samples * 0.7:
                print(f"[VAD]    - 合并过短分段 {i+1}: {next_len/self.sample_rate:.2f}秒 到 {i}: {curr_len/self.sample_rate:.2f}秒")
                merged[i] = np.concatenate([merged[i], merged[i+1]])
                merged.pop(i+1)
                merge_count += 1
            else:
                i += 1
        
        print(f"[VAD]    - 优化合并完成，合并次数: {merge_count}")
        
        # 后处理优化2：检查最后一个分段
        if len(merged) > 1:
            last_len = len(merged[-1])
            # 如果最后一个分段太短，与前一个合并
            if last_len < min_target_samples * 0.7:
                merged[-2] = np.concatenate([merged[-2], merged[-1]])
                merged.pop()
        
        # 统计合并后的分段信息
        merged_lengths = [len(seg) for seg in merged]
        merged_total = sum(merged_lengths)
        merged_avg = merged_total / len(merged) if merged else 0
        
        print(f"[VAD] 分段合并后：总数={len(merged)}, 总长={merged_total/self.sample_rate:.2f}秒, 平均={merged_avg/self.sample_rate:.2f}秒")
        for i, seg_len in enumerate(merged_lengths):
            print(f"[VAD] 合并后分段 {i+1}: {seg_len/self.sample_rate:.2f}秒 ({seg_len}样本)")
        
        return merged
    
    def _validate_total_duration(self, segments, original_length, sample_rate):
        """
        验证所有分段总时长是否等于原音频时长
        """
        try:
            total_samples = sum(len(segment) for segment in segments)
            difference = abs(total_samples - original_length)
            difference_seconds = difference / sample_rate
            
            # 计算百分比差异
            if original_length > 0:
                percentage_diff = (difference / original_length) * 100
            else:
                percentage_diff = 0
            
            print(f"[VAD] 总时长验证: 原始={original_length}样本, 分段总和={total_samples}样本")
            print(f"[VAD] 差异={difference_seconds:.3f}秒 ({percentage_diff:.2f}%)")
            
            # 根据差异大小给出不同级别的警告
            if difference_seconds > 0.1:
                print(f"[VAD] 警告：分段总时长与原音频有较大差异 ({difference_seconds:.3f}秒)")
            elif difference_seconds > 0.01:
                print(f"[VAD] 注意：分段总时长与原音频有轻微差异 ({difference_seconds:.3f}秒)")
            else:
                print(f"[VAD] 分段总时长验证通过，差异小于0.01秒")
                
            # 检查是否有分段丢失
            if total_samples < original_length:
                print(f"[VAD] 警告：分段总时长小于原音频，可能有音频数据丢失")
        except Exception as e:
            print(f"[VAD] 总时长验证失败: {str(e)}")
    
    def _resample_audio(self, audio, original_rate, target_rate):
        """
        重采样音频 - 增强版，更好地处理极短音频
        """
        resample_start_time = time.time()
        try:
            print(f"[VAD] 🔄 重采样处理: 从 {original_rate}Hz 到 {target_rate}Hz")
            print(f"[VAD]    - 原始长度: {len(audio)}样本")
            print(f"[VAD]    - 原始时长: {len(audio)/original_rate:.3f}秒")
            
            # 对于极短音频的特殊处理
            if len(audio) <= 1:
                # 确保至少有2个采样点以避免重采样错误
                min_samples = max(2, int(target_rate / original_rate * len(audio)))
                print(f"[VAD] 🛑 处理极短音频({len(audio)}样本)，确保至少{min_samples}个采样点")
                result = np.zeros(min_samples) if len(audio) == 0 else np.full(min_samples, audio[0])
                print(f"[VAD]    - 生成固定值音频，长度={len(result)}样本")
                return result
            
            # 使用scipy进行重采样
            duration = len(audio) / original_rate
            num_samples = max(2, int(duration * target_rate))  # 确保至少有2个采样点
            print(f"[VAD]    - 目标采样数: {num_samples} (基于时长: {duration:.3f}秒)")
            
            resampled = signal.resample(audio, num_samples)
            
            # 确保重采样后的音频范围合理
            max_abs_val = np.max(np.abs(resampled))
            print(f"[VAD]    - 重采样后振幅范围: [-{max_abs_val:.3f}, {max_abs_val:.3f}]")
            
            if max_abs_val > 1.0:
                print(f"[VAD] ⚠️ 警告：重采样后音频振幅超过1.0，进行归一化")
                if max_abs_val > 0:
                    resampled = resampled / max_abs_val
                    print(f"[VAD]    - 归一化完成，新振幅范围: [-1.0, 1.0]")
            
            resample_duration = time.time() - resample_start_time
            print(f"[VAD] ✅ 重采样完成")
            print(f"[VAD]    - 新长度: {len(resampled)}样本")
            print(f"[VAD]    - 新时长: {len(resampled)/target_rate:.3f}秒")
            print(f"[VAD]    - 处理耗时: {resample_duration:.3f}秒")
            return resampled
        except Exception as e:
            print(f"[VAD] ❌ 音频重采样失败: {str(e)}")
            import traceback
            print(f"[VAD] 📋 重采样错误详情:")
            traceback.print_exc()
            
            # 出错时生成合理的默认值，避免空音频
            default_samples = max(2, int(target_rate / original_rate * len(audio)))
            print(f"[VAD] 🔄 生成默认音频替代: {default_samples}样本")
            result = np.zeros(default_samples) if len(audio) == 0 else np.full(default_samples, np.mean(audio))
            return result

class WhisperTranscribeNode:
    """
    音频转文字节点 - 使用Whisper模型将音频转换为文字
    支持多语言识别、自动语言检测、长音频分段处理
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "language": ([
                    "zh", "en", "ja", "ko", "fr", "de", "es", "ru", "pt"
                ], {
                    "default": "zh",
                    "tooltip": "选择音频语言，中文环境推荐直接选zh"
                }),

            },
            "optional": {
                "enable_word_timestamps": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "是否启用词级时间戳（需要large模型，对中文效果不佳）"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.1,
                    "tooltip": "中文推荐0.0-0.2，值越小越确定但可能错过细节"
                }),
                "beam_size": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "tooltip": "中文推荐5-10，束搜索宽度越大越精准但越慢"
                }),
                "best_of": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 10, 
                    "step": 1,
                    "tooltip": "中文推荐3-5，候选结果数量"
                }),
                "condition_on_previous_text": ("BOOLEAN", {
                    "default": True, 
                    "tooltip": "是否基于前文上下文，中文对话推荐开启"
                }),            
                "unload_model_after_transcribe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "转录完成后是否卸载模型释放内存（建议大模型或内存不足时开启）"
                }),
                "initial_prompt": ("STRING", {
                    "default": "以下是中文语音转录：", 
                    "multiline": True,
                    "tooltip": "初始提示词，可加入中文标点、常见词汇等"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription",)
    FUNCTION = "transcribe"
    CATEGORY = "lianlaoshi/audio"

    def transcribe(self, audio, language="zh", enable_word_timestamps=False,
                   temperature=0.0, beam_size=5, best_of=5, condition_on_previous_text=True,
                   initial_prompt="以下是中文语音转录：", unload_model_after_transcribe=False):
        """
        音频转录主函数 - 针对中文环境优化
        """
        print(f"[WhisperTranscribeNode] 🎤 开始音频转录 - 使用最优模型: large-v3, 语言: {language}")
        print(f"[WhisperTranscribeNode] 🔧 中文优化参数 - 温度: {temperature}, 束搜索: {beam_size}, 候选: {best_of}")
        
        try:
            # 检查音频输入
            if not audio or "waveform" not in audio:
                print(f"[WhisperTranscribeNode] ❌ 错误: 无效的音频输入")
                return ("",)
            
            # 获取音频数据
            waveform = audio["waveform"]
            sample_rate = int(audio.get("sample_rate", 16000))
            
            print(f"[WhisperTranscribeNode] 📊 输入音频信息 - shape: {waveform.shape}, sample_rate: {sample_rate}")
            
            # 处理音频维度
            if waveform.dim() == 3:
                waveform = waveform[0]  # 去掉batch维度
            if waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)  # 多通道转单通道
            
            waveform = waveform.squeeze()
            print(f"[WhisperTranscribeNode] 🔧 处理后的音频shape: {waveform.shape}")
            
            # 重采样到16kHz（Whisper要求）
            if sample_rate != 16000:
                print(f"[WhisperTranscribeNode] 🔄 重采样: {sample_rate}Hz -> 16000Hz")
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            # 设置设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[WhisperTranscribeNode] 💻 使用设备: {device}")
            
            # 加载Whisper模型 - 固定使用large-v3，中文最优选择
            model_name = "openai/whisper-large-v3"
            print(f"[WhisperTranscribeNode] 📦 加载最优中文模型: {model_name}")
            
            # 使用相对路径获取ComfyUI的models目录
            import os
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models", "whisper")
            print(f"[WhisperTranscribeNode] 📁 模型目录: {models_dir}")
            
            # 检查本地模型是否存在
            local_model_path = os.path.join(models_dir, "whisper-large-v3")
            if os.path.exists(local_model_path):
                print(f"[WhisperTranscribeNode] ✅ 发现本地模型，从本地加载: {local_model_path}")
                processor = WhisperProcessor.from_pretrained(local_model_path)
                model = WhisperForConditionalGeneration.from_pretrained(local_model_path).to(device).eval()
            else:
                print(f"[WhisperTranscribeNode] 🌐 本地模型不存在，从Hugging Face下载并缓存到本地")
                print(f"[WhisperTranscribeNode] ⏱️ 首次下载可能需要几分钟，请耐心等待...")
                print(f"[WhisperTranscribeNode] 💡 下载进度可在ComfyUI控制台或终端查看")
                print(f"[WhisperTranscribeNode] 🔧 支持断点续传，意外中断可重新继续下载")
                # 确保模型目录存在
                os.makedirs(models_dir, exist_ok=True)
                
                # 从Hugging Face下载并保存到本地
                print(f"[WhisperTranscribeNode] 📥 开始下载处理器...")
                processor = WhisperProcessor.from_pretrained(model_name, resume_download=True)
                print(f"[WhisperTranscribeNode] ✅ 处理器下载完成")
                
                print(f"[WhisperTranscribeNode] 📥 开始下载模型（约3GB，可能需要几分钟）...")
                print(f"[WhisperTranscribeNode] 🔄 下载过程中请勿关闭程序...")
                print(f"[WhisperTranscribeNode] 💡 提示：large-v3模型文件较大，网速慢的用户请耐心等待")
                print(f"[WhisperTranscribeNode] 📊 如果下载中断，重新运行会继续下载")
                
                # 记录开始时间
                import time
                start_time = time.time()
                
                try:
                    model = WhisperForConditionalGeneration.from_pretrained(model_name, resume_download=True)
                    
                    # 计算下载用时
                    download_time = time.time() - start_time
                    download_speed = 3.0 / (download_time / 60)  # 估算下载速度（GB/分钟）
                    print(f"[WhisperTranscribeNode] ✅ 模型下载完成（用时: {download_time:.1f}秒，约{download_speed:.1f}GB/分钟）")
                    
                    # 保存到本地
                    print(f"[WhisperTranscribeNode] 💾 正在保存模型到本地...")
                    save_start = time.time()
                    processor.save_pretrained(local_model_path)
                    model.save_pretrained(local_model_path)
                    save_time = time.time() - save_start
                    
                    print(f"[WhisperTranscribeNode] ✅ 模型已保存到本地: {local_model_path}")
                    print(f"[WhisperTranscribeNode] 🎉 下次使用将直接从本地加载，速度更快！")
                    print(f"[WhisperTranscribeNode] 📊 总用时: {download_time + save_time:.1f}秒")
                    model = model.to(device).eval()
                    
                except Exception as download_error:
                    print(f"[WhisperTranscribeNode] ❌ 下载失败: {download_error}")
                    print(f"[WhisperTranscribeNode] 💡 请检查网络连接，或手动下载后放到: {local_model_path}")
                    print(f"[WhisperTranscribeNode] 🔗 手动下载地址: https://huggingface.co/openai/whisper-large-v3")
                    raise download_error
            
            # 准备生成参数 - 中文优化配置
            generate_kwargs = {
                "temperature": temperature,
                "num_beams": beam_size,
                "num_return_sequences": 1,
                "do_sample": temperature > 0,
                "early_stopping": True,
            }
            
            # 设置语言相关参数 - 修复语言强制问题
            if language != "auto":
                # 设置强制语言 - 使用更直接的方式
                print(f"[WhisperTranscribeNode] 🌍 强制语言: {language}")
                # 使用language参数直接传递给generate方法
                generate_kwargs["language"] = language
            
            # 中文特殊优化
            if language == "zh":
                generate_kwargs.update({
                    "repetition_penalty": 1.1,  # 减少重复
                    "length_penalty": 1.0,      # 平衡长度
                    "no_repeat_ngram_size": 3, # 避免3-gram重复
                })
                print(f"[WhisperTranscribeNode] 🇨🇳 应用中文优化参数")
            
            if enable_word_timestamps:
                generate_kwargs["return_timestamps"] = "word"
                print(f"[WhisperTranscribeNode] ⏱️ 启用词级时间戳（large-v3支持）")
            
            # 设置初始提示词
            if initial_prompt and initial_prompt.strip():
                print(f"[WhisperTranscribeNode] 📝 使用初始提示词: {initial_prompt[:30]}...")
                print(f"[WhisperTranscribeNode] 💡 初始提示词将在转录后处理中应用: {initial_prompt[:50]}...")
                # 注意：由于Whisper内部prompt_ids处理复杂，我们暂时不直接设置prompt_ids
                # 而是在转录结果后处理中应用初始提示词的逻辑
            
            # 处理音频
            total_samples = waveform.size(0)
            audio_duration = total_samples / 16000
            print(f"[WhisperTranscribeNode] ⏱️ 音频总时长: {audio_duration:.2f}秒")
            
            # 准备输入
            inputs = processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding="longest",
                truncation=False
            )
            input_features = inputs["input_features"].to(device)
            print(f"[WhisperTranscribeNode] 📊 原始mel特征形状: {input_features.shape}")
            
            # 确保mel特征长度为3000
            if input_features.shape[-1] < 3000:
                # 填充到3000
                pad_length = 3000 - input_features.shape[-1]
                input_features = torch.nn.functional.pad(input_features, (0, pad_length))
                print(f"[WhisperTranscribeNode] 📏 mel特征填充到3000: {input_features.shape}")
            elif input_features.shape[-1] > 3000:
                # 截断到3000
                input_features = input_features[..., :3000]
                print(f"[WhisperTranscribeNode] 📏 mel特征截断到3000: {input_features.shape}")
            
            # 生成转录
            with torch.no_grad():
                print(f"[WhisperTranscribeNode] 🎯 使用指定语言生成: {language}")
                generated_ids = model.generate(input_features, **generate_kwargs)
            
            # 解码
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            full_transcription = text.strip()
            
            # 应用初始提示词的影响（后处理）
            if initial_prompt and initial_prompt.strip():
                # 如果初始提示词包含特定的格式要求，尝试在结果中体现
                if "以下是中文语音转录" in initial_prompt or "中文" in initial_prompt:
                    print(f"[WhisperTranscribeNode] 📝 应用中文初始提示词影响")
                
                # 如果初始提示词包含特定的格式要求
                if "标点" in initial_prompt or "标点符号" in initial_prompt:
                    print(f"[WhisperTranscribeNode] 📝 应用标点符号提示")
                
                print(f"[WhisperTranscribeNode] ✅ 初始提示词影响已应用")
            
            # 中文后处理优化
            if language == "zh":
                # 调用专门的中文后处理方法
                full_transcription = self._post_process_chinese(full_transcription)
                
                print(f"[WhisperTranscribeNode] 🇨🇳 应用中文后处理优化")
            
            print(f"[WhisperTranscribeNode] ✅ 转录完成")
            print(f"[WhisperTranscribeNode] 📝 转录结果: {full_transcription[:100]}...")
            
            # 清理资源，减少Windows异步I/O错误
            if hasattr(model, 'cpu'):
                model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 根据用户选择卸载模型
            if unload_model_after_transcribe:
                print(f"[WhisperTranscribeNode] 🧹 开始卸载模型以释放内存...")
                try:
                    # 释放模型和处理器资源
                    del model
                    del processor
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"[WhisperTranscribeNode] ✅ 模型已成功卸载，内存已释放")
                except Exception as e:
                    print(f"[WhisperTranscribeNode] ⚠️ 模型卸载时发生错误: {e}")
                    # 即使出错也继续返回结果
            else:
                print(f"[WhisperTranscribeNode] 💡 未开启模型卸载，模型将保持在内存中供下次使用")
                print(f"[WhisperTranscribeNode] 💡 提示：如果内存不足，可以在节点设置中开启'转录完成后卸载模型'")
            
            return (full_transcription,)
            
        except Exception as e:
            print(f"[WhisperTranscribeNode] ❌ 转录失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 添加更友好的错误信息
            error_msg = str(e)
            if "ProactorBasePipeTransport" in error_msg:
                error_msg = "程序运行环境异常，请重启ComfyUI后重试"
            elif "connection_lost" in error_msg:
                error_msg = "连接中断，请检查网络状态并重启ComfyUI"
            elif "socket" in error_msg.lower():
                error_msg = "网络连接异常，请检查网络设置"
            
            return (f"转录失败: {error_msg}",)

    def _post_process_chinese(self, text):
        """中文后处理优化 - 全面处理标点符号和繁简转换，支持上下文理解"""
        import re
        
        # 1. 英文标点转中文标点（改进版，处理引号配对问题）
        # 先处理左右引号
        text = re.sub(r'"([^"]+)"', r'“\1”', text)  # 双引号配对
        text = re.sub(r"'([^']+)'", r'‘\1’', text)  # 单引号配对
        
        # 处理其他标点
        punctuation_map = {
            ',': '，',  # 逗号
            '.': '。',  # 句号
            '?': '？',  # 问号
            '!': '！',  # 感叹号
            ':': '：',  # 冒号
            ';': '；',  # 分号
            '(': '（',  # 括号（左）
            ')': '）',  # 括号（右）
        }
        
        for en_punc, zh_punc in punctuation_map.items():
            text = text.replace(en_punc, zh_punc)
        
        # 2. 增强的繁体转简体字典
        traditional_to_simplified = {
            # 基础词汇
            '妳': '你', '牠': '它', '祂': '它',
            '因爲': '因为', '爲什麼': '为什么', '這樣': '这样',
            '什麼': '什么', '怎麼': '怎么', '那麼': '那么',
            '嗎': '吗', '呢': '呢', '吧': '吧',
            '對': '对', '說': '说', '時': '时',
            '個': '个', '來': '来', '會': '会',
            '這': '这', '那': '那', '裡': '里', '裡面': '里面',
            '們': '们', '他': '他', '她': '她', '它': '它',
            '沒': '没', '有': '有', '是': '是', '在': '在',
            '的': '的', '和': '和', '與': '与', '及': '及',
            '將': '将', '為': '为', '會': '会', '要': '要',
            '可以': '可以', '可能': '可能', '必須': '必须',
            '應該': '应该', '能夠': '能够', '需要': '需要',
            # 常用动词
            '說': '说', '做': '做', '看': '看', '聽': '听',
            '寫': '写', '讀': '读', '學習': '学习', '工作': '工作',
            '認識': '认识', '了解': '了解', '知道': '知道',
            # 常用名词
            '時間': '时间', '地方': '地方', '事情': '事情',
            '問題': '问题', '方法': '方法', '原因': '原因',
            '結果': '结果', '例子': '例子', '意見': '意见',
            # 其他常用词
            '已經': '已经', '現在': '现在', '將來': '将来',
            '以前': '以前', '之后': '之后', '重要': '重要',
            '特別': '特别', '非常': '非常', '很多': '很多',
            '一些': '一些', '所有': '所有', '自己': '自己',
        }
        
        # 执行繁简转换
        for traditional, simplified in traditional_to_simplified.items():
            text = text.replace(traditional, simplified)
        
        # 3. 智能句子分割和标点添加
        # 步骤1: 先处理已有标点，确保正确格式
        text = re.sub(r'([。！？])\s*', r'\1', text)  # 移除标点后的空格
        
        # 步骤2: 增强的句子分割 - 考虑更多句子边界模式
        # 使用正则表达式分割句子，同时保留分隔符
        def split_and_keep_delimiters(text, delimiters):
            # 创建一个包含所有分隔符的正则表达式
            pattern = '(' + '|'.join(re.escape(delim) for delim in delimiters) + ')'  # noqa
            # 分割字符串
            parts = re.split(pattern, text)
            # 组合句子和分隔符
            sentences = []
            current_sentence = ''
            for part in parts:
                if part in delimiters:
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip() + part)
                        current_sentence = ''
                else:
                    current_sentence += part
            # 处理最后一个句子（如果没有分隔符）
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            return sentences
        
        # 定义分隔符列表
        delimiters = ['。', '！', '？']
        sentences = split_and_keep_delimiters(text, delimiters)
        
        # 步骤3: 智能标点处理（单个句子级别）
        processed_sentences = []
        for sentence in sentences:
            # 去除首尾空白
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查句子是否已经有合适的结尾标点
            if any(sentence.endswith(punc) for punc in ['。', '！', '？']):
                # 句子已有结尾标点，只添加内部标点
                sentence = self._add_internal_punctuation(sentence)
            else:
                # 根据句子特征添加合适的结尾标点
                if any(sentence.endswith(ending) for ending in ['吗', '呢', '么', '吧']):
                    sentence += '？'
                elif any(sentence.endswith(ending) for ending in ['啊', '呀', '哇', '啦']):
                    # 检查句子长度和内容，判断是感叹还是陈述
                    if len(sentence) > 5 or any(word in sentence for word in ['真的', '非常', '太', '好', '棒']):
                        sentence += '！'
                    else:
                        sentence += '。'
                else:
                    # 检查是否是命令或请求
                    if any(word in sentence[:3] for word in ['请', '帮我', '麻烦']):
                        sentence += '。'
                    # 检查是否是感叹句
                    elif any(word in sentence for word in ['！', '啊', '呀', '哇']):
                        sentence = sentence.replace('！', '') + '！'
                    # 默认为陈述句
                    else:
                        sentence += '。'
                
                # 添加内部标点
                sentence = self._add_internal_punctuation(sentence)
            
            processed_sentences.append(sentence)
        
        # 步骤4: 添加上下文理解，优化句子之间的标点和关系
        # 检测文本类型（对话、叙述等）
        text_type = self._detect_text_type(processed_sentences)
        
        # 基于上下文优化标点和句子关系
        processed_sentences = self._optimize_with_context(processed_sentences, text_type)
        
        # 重新组合句子
        text = ' '.join(processed_sentences)
        
        # 4. 添加适当的空格
        # 数字和单位之间添加空格
        text = re.sub(r'(\d+)([年月日时分秒])', r'\1 \2', text)
        # 中文和英文/数字之间添加空格
        text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z0-9])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fa5])', r'\1 \2', text)
        
        # 5. 修正数字和单位格式
        # 修正电话号码格式
        text = re.sub(r'(\d{3,4})\s*-?\s*(\d{3,4})\s*-?\s*(\d{3,4})', r'\1-\2-\3', text)
        # 修正日期格式
        text = re.sub(r'(\d{4})[年/-]*(\d{1,2})[月/-]*(\d{1,2})[日]?', r'\1年\2月\3日', text)
        # 修正时间格式（分步处理）
        # 先处理有秒的情况
        text = re.sub(r'(\d{1,2})[:：]\s*(\d{1,2})[:：]\s*(\d{1,2})', r'\1时\2分\3秒', text)
        # 再处理只有时分的情况
        text = re.sub(r'(\d{1,2})[:：]\s*(\d{1,2})', r'\1时\2分', text)
        
        # 6. 处理连续的标点符号和空格
        # 合并连续的句号
        text = re.sub(r'。{2,}', '。', text)
        # 合并连续的感叹号
        text = re.sub(r'！{2,}', '！', text)
        # 合并连续的问号
        text = re.sub(r'？{2,}', '？', text)
        # 处理逗号重复
        text = re.sub(r'，{2,}', '，', text)
        # 合并连续空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def _detect_text_type(self, sentences):
        """检测文本类型，判断是对话、叙述还是其他类型"""
        # 对话模式检测关键词
        dialogue_indicators = ['你', '我', '他', '她', '它', '你们', '我们', '他们', '吗', '呢', '吧']
        # 叙述模式检测关键词
        narrative_indicators = ['今天', '昨天', '明天', '然后', '接着', '后来', '之前', '之后', '首先', '其次']
        
        dialogue_count = 0
        narrative_count = 0
        
        # 计算对话和叙述特征的出现次数
        for sentence in sentences:
            # 检查是否有明显的对话特征
            for indicator in dialogue_indicators:
                if indicator in sentence:
                    dialogue_count += 1
                    break
            
            # 检查是否有明显的叙述特征
            for indicator in narrative_indicators:
                if indicator in sentence:
                    narrative_count += 1
                    break
        
        # 确定文本类型
        if dialogue_count >= len(sentences) * 0.5:
            return "dialogue"  # 对话类型
        elif narrative_count >= len(sentences) * 0.4:
            return "narrative"  # 叙述类型
        else:
            return "general"  # 普通类型
            
    def _optimize_with_context(self, sentences, text_type):
        """基于上下文优化句子之间的标点和关系"""
        if not sentences or len(sentences) <= 1:
            return sentences
            
        optimized_sentences = []
        
        # 1. 对话模式处理
        if text_type == "dialogue":
            # 对话模式下，优化问答关系，确保问题后面跟问号，回答后面根据情况选择标点
            for i, sentence in enumerate(sentences):
                # 如果当前句子以问号结尾，而下一句不以问号或感叹号结尾，考虑是否需要调整为回答
                if i < len(sentences) - 1:
                    if sentence.endswith('？') and not sentences[i+1].endswith(('？', '！')):
                        # 问题后面的句子可能是回答，可以考虑添加引号
                        # 这里简化处理，主要确保标点正确
                        pass
                    
                # 对于疑问句，确保标点正确
                if any(word in sentence for word in ['谁', '什么', '哪里', '何时', '为什么', '如何', '吗', '呢', '吧']) and not sentence.endswith(('？', '！')):
                    # 移除原有句号，添加问号
                    if sentence.endswith('。'):
                        sentence = sentence[:-1] + '？'
                    
                optimized_sentences.append(sentence)
        
        # 2. 叙述模式处理
        elif text_type == "narrative":
            # 叙述模式下，优化时间顺序和因果关系
            for i, sentence in enumerate(sentences):
                # 检查是否存在因果关系关键词
                causal_indicators = ['因为', '所以', '因此', '于是', '导致', '结果']
                for indicator in causal_indicators:
                    if indicator in sentence and not any(sentence.endswith(punc) for punc in ['！', '？']):
                        # 因果关系句子通常用句号结尾
                        if not sentence.endswith('。'):
                            if sentence.endswith(('，', '；')):
                                sentence = sentence[:-1] + '。'
                            else:
                                sentence = sentence + '。'
                
                # 处理时间顺序连接词
                temporal_indicators = ['然后', '接着', '后来', '之后']
                if i > 0 and any(indicator in sentence for indicator in temporal_indicators):
                    # 确保前一句以句号结尾，而不是问号或感叹号
                    prev_sentence = optimized_sentences[-1]
                    if prev_sentence.endswith(('？', '！')):
                        # 如果前一句是疑问句或感叹句，则保持不变
                        pass
                    elif not prev_sentence.endswith('。'):
                        # 否则确保前一句以句号结尾
                        optimized_sentences[-1] = prev_sentence[:-1] + '。' if prev_sentence.endswith(('，', '；')) else prev_sentence + '。'
                
                optimized_sentences.append(sentence)
        
        # 3. 通用模式处理
        else:
            # 处理句子之间的逻辑关系
            for i, sentence in enumerate(sentences):
                # 检查是否有连接词需要特殊处理
                conjunction_sentence = False
                
                # 转折关系
                contrast_words = ['但是', '不过', '然而', '可是', '却']
                for word in contrast_words:
                    if word in sentence and i > 0:
                        conjunction_sentence = True
                        # 确保前一句以句号结尾
                        prev_sentence = optimized_sentences[-1]
                        if not prev_sentence.endswith('。'):
                            optimized_sentences[-1] = prev_sentence[:-1] + '。' if prev_sentence.endswith(('，', '；')) else prev_sentence + '。'
                        break
                
                # 并列关系
                if not conjunction_sentence:
                    parallel_words = ['同时', '另外', '此外', '还有']
                    for word in parallel_words:
                        if word in sentence and i > 0:
                            conjunction_sentence = True
                            # 确保前一句以句号结尾
                            prev_sentence = optimized_sentences[-1]
                            if not prev_sentence.endswith('。'):
                                optimized_sentences[-1] = prev_sentence[:-1] + '。' if prev_sentence.endswith(('，', '；')) else prev_sentence + '。'
                            break
                
                optimized_sentences.append(sentence)
        
        # 4. 确保段落逻辑一致性
        # 检查相邻句子是否存在矛盾的标点使用
        for i in range(len(optimized_sentences) - 1):
            # 如果当前句是感叹句，而下一句很短且不是感叹句，可能需要调整
            if optimized_sentences[i].endswith('！') and len(optimized_sentences[i+1]) < 10 and not optimized_sentences[i+1].endswith('！'):
                # 可能是连续的感叹，考虑将下一句也改为感叹句
                if not optimized_sentences[i+1].endswith('？'):  # 避免将问句改为感叹句
                    if optimized_sentences[i+1].endswith('。'):
                        optimized_sentences[i+1] = optimized_sentences[i+1][:-1] + '！'
        
        return optimized_sentences
        
    def _add_internal_punctuation(self, sentence):
        """为句子添加内部标点符号（如逗号）"""
        import re
        
        # 保留结尾标点
        ending_punc = ''
        if any(sentence.endswith(punc) for punc in ['。', '！', '？']):
            ending_punc = sentence[-1]
            sentence = sentence[:-1]
        
        # 1. 在连接词后添加逗号 - 扩展连接词库
        connectors = {
            '转折': ['但是', '不过', '然而', '可是', '却', '只是', '反而', '倒是'],
            '因果': ['因为', '所以', '因此', '因而', '于是', '致使', '导致', '结果'],
            '并列': ['同时', '另外', '此外', '还有', '同样', '并且', '而且'],
            '时间': ['然后', '接着', '之后', '后来', '此前', '首先', '其次', '再次'],
            '选择': ['或者', '否则', '要么', '与其', '不如', '宁可']
        }
        
        # 展开所有连接词
        all_connectors = [conn for conn_list in connectors.values() for conn in conn_list]
        
        for connector in all_connectors:
            # 确保只匹配完整的词
            pattern = r'(?<!\w)' + re.escape(connector) + r'(?!\w)'  # noqa
            sentence = re.sub(pattern, connector + '，', sentence)
        
        # 2. 在时间词后添加逗号 - 扩展时间词库
        time_words = [
            # 日期
            '今天', '明天', '昨天', '前天', '后天', '大后天', '大大后天',
            '本周', '上周', '下周', '本周一', '本周二', '本周三', '本周四', '本周五', '本周六', '本周日',
            '上周一', '上周二', '上周三', '上周四', '上周五', '上周六', '上周日',
            '下周一', '下周二', '下周三', '下周四', '下周五', '下周六', '下周日',
            # 时间段
            '上午', '下午', '晚上', '早上', '中午', '凌晨', '夜晚',
            '早晨', '黄昏', '傍晚', '深夜', '午后',
            # 时间副词
            '现在', '后来', '之前', '之后', '目前', '从前', '将来', '未来',
            '立刻', '马上', '顿时', '瞬间', '忽然', '突然', '不久', '稍后'
        ]
        
        for time_word in time_words:
            pattern = r'(?<!\w)' + re.escape(time_word) + r'(?!\w)'  # noqa
            sentence = re.sub(pattern, time_word + '，', sentence)
        
        # 3. 在地点词后添加逗号（改进规则）
        place_suffixes = ['市', '区', '县', '镇', '乡', '村', '路', '街', '巷', '道', '桥', '省', '州', '府', '厅']
        for suffix in place_suffixes:
            # 改进地点模式匹配，避免在不需要的地方添加逗号
            pattern = r'(?<=\w)' + re.escape(suffix) + r'(?=[\u4e00-\u9fa5][^\u4e00-\u9fa5]|\w+)'  # noqa
            sentence = re.sub(pattern, suffix + '，', sentence)
        
        # 4. 处理冒号情况（如列举）
        if '例如' in sentence or '比如' in sentence:
            sentence = sentence.replace('例如', '例如：').replace('比如', '比如：')
        
        # 5. 新增：添加数字后的量词标点规则
        # 处理数字+量词模式
        sentence = re.sub(r'(\d+)([个只条辆本件台])(?=[\u4e00-\u9fa5])', r'\1\2，', sentence)
        
        # 6. 新增：添加感叹词后的标点
        exclamation_words = ['啊', '呀', '哦', '哇', '哈', '嘿', '喂', '嗯', '哎', '哎唷']
        for word in exclamation_words:
            # 确保感叹词后添加适当标点
            pattern = r'(?<!\w)' + re.escape(word) + r'(?!\w)'  # noqa
            sentence = re.sub(pattern, word + '，', sentence)
        
        # 7. 修正多余的逗号
        # 避免多个连续逗号
        sentence = re.sub(r'，{2,}', '，', sentence)
        # 避免开头有逗号
        if sentence.startswith('，'):
            sentence = sentence[1:]
        # 避免结尾有逗号（除非是句子中间部分）
        if sentence.endswith('，'):
            sentence = sentence[:-1]
        
        # 恢复结尾标点
        return sentence + ending_punc
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            # 只进行基本资源清理，避免干扰ComfyUI的事件循环
            # 移除异步任务取消，防止asyncio.exceptions.CancelledError
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # 忽略清理过程中的错误

# 节点映射
NODE_CLASS_MAPPINGS = {
    "WhisperVADSplitNode": WhisperVADSplitNode,
    "WhisperTranscribeNode": WhisperTranscribeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WhisperVADSplitNode": "lian 智能音频分段(VAD)",
    "WhisperTranscribeNode": "lian 音频转文字(Whisper)",
}
