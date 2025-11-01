import os
import subprocess
import numpy as np
import tempfile
import json
from typing import List, Dict, Any
import torch
import folder_paths
from comfy.utils import ProgressBar

def tensor_to_int(tensor, bits):
    tensor = tensor.cpu().numpy() * (2**bits-1) + 0.5
    return np.clip(tensor, 0, (2**bits-1))

def tensor_to_shorts(tensor):
    return tensor_to_int(tensor, 16).astype(np.uint16)

def tensor_to_bytes(tensor):
    return tensor_to_int(tensor, 8).astype(np.uint8)

def ffmpeg_process(args, video_format, video_metadata, file_path, env, total_frames=None):
    import json
    
    res = None
    frame_data = yield
    total_frames_output = 0
    
    # 处理元数据保存（如果需要）
    if video_format.get('save_metadata', 'False') != 'False':
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        metadata = json.dumps(video_metadata)
        metadata_path = os.path.join(folder_paths.get_temp_directory(), "metadata.txt")
        # 转义元数据中的特殊字符
        metadata = metadata.replace("\\","\\\\")
        metadata = metadata.replace(";","\\;")
        metadata = metadata.replace("#","\\#")
        metadata = metadata.replace("=","\\=")
        metadata = metadata.replace("\n","\\\n")
        metadata = "comment=" + metadata
        with open(metadata_path, "w") as f:
            f.write(";FFMETADATA1\n")
            f.write(metadata)
        m_args = args[:1] + ["-i", metadata_path] + args[1:] + ["-metadata", "creation_time=now"]
        
        with subprocess.Popen(m_args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                err = proc.stderr.read()
                # 检查输出文件是否存在
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise Exception("An error occurred in the ffmpeg subprocess:\n" + err.decode('utf-8', errors='ignore'))
    
    # 如果元数据处理失败或不需要元数据，使用标准处理
    if res != b'':
        with subprocess.Popen(args + [file_path], stderr=subprocess.PIPE,
                              stdin=subprocess.PIPE, env=env) as proc:
            try:
                while frame_data is not None:
                    proc.stdin.write(frame_data)
                    frame_data = yield
                    total_frames_output += 1
                proc.stdin.flush()
                proc.stdin.close()
                res = proc.stderr.read()
            except BrokenPipeError as e:
                res = proc.stderr.read()
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise Exception("An error occurred in the ffmpeg subprocess:\n" + res.decode('utf-8', errors='ignore'))
    
    yield total_frames_output
    if len(res) > 0:
        print(res.decode('utf-8', errors='ignore'), end="")

class VideoCombineNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 16.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "filename": ("STRING", {"default": "video_output"}),
                "pix_fmt": (["yuv420p", "yuv420p10le", "yuv422p", "yuv444p", "rgb24", "rgba"], {"default": "yuv420p"}),
                "crf": ("INT", {"default": 19, "min": 0, "max": 51, "step": 1}),
                "save_metadata": ("BOOLEAN", {"default": True}),
                "trim_to_audio": ("BOOLEAN", {"default": False}),
                "last_frames_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "pingpong": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "filename", "last_frames_images")
    OUTPUT_NODE = True
    CATEGORY = "lianlaoshi/video"
    FUNCTION = "combine_video"

    def combine_video(
        self,
        images,
        frame_rate: float,
        filename: str,
        pix_fmt: str = "yuv420p",
        crf: int = 19,
        save_metadata: bool = True,
        trim_to_audio: bool = False,
        last_frames_count: int = 0,
        audio=None,
        pingpong: bool = False,
        **kwargs
    ):
        # 处理输入数据
        batched_output = images

        # 处理pingpong效果
        if pingpong:
            batched_output = torch.cat([batched_output, batched_output[1:-1].flip(0)], dim=0)

        # 获取总帧数用于进度条
        total_frames = len(batched_output)

        # 获取输出目录
        output_dir = folder_paths.get_output_directory()
        
        # 生成文件名 - 直接使用mp4扩展名
        counter = 1
        extension = "mp4"
        while True:
            full_filename = f"{filename}_{counter:05d}.{extension}"
            file_path = os.path.join(output_dir, full_filename)
            if not os.path.exists(file_path):
                break
            counter += 1

        # 准备视频元数据
        dimensions = f"{batched_output.shape[2]}x{batched_output.shape[1]}"
        
        # 创建帧生成器，转换为字节数据（匹配VHS实现）
        def frame_generator():
            for i, image in enumerate(batched_output):
                # 转换tensor到numpy数组
                img_array = 255. * image.cpu().numpy()
                img = np.clip(img_array, 0, 255).astype(np.uint8)
                yield img.tobytes()  # 直接返回字节数据

        video_metadata = {
            "frame_rate": frame_rate,
            "filename": filename,
        }

        # 构建FFmpeg命令，参考VHS原始实现
        args = [
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
            "-color_trc", "iec61966-2-1",
            "-s", dimensions, "-r", str(frame_rate), "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", pix_fmt,
            "-crf", str(crf),
            "-preset", "medium"
        ]
        
        # 添加音频处理（暂时禁用）
        args.extend(["-an"])

        # 设置环境变量
        env = os.environ.copy()
        
        # 构建video_format用于ffmpeg_process
        video_format = {
            "extension": extension,
            "save_metadata": str(save_metadata)
        }
        

        
        # 处理音频合并（参考VHS原始实现）
        final_file_path = file_path
        final_filename = full_filename
        
        if audio is not None:
            # 检查音频数据是否有效
            a_waveform = None
            try:
                a_waveform = audio['waveform']
            except:
                pass
            
            if a_waveform is not None:
                # 先生成无音频视频
                output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env, total_frames)
                output_process.send(None)  # 启动生成器
                
                # 创建进度条
                pbar = ProgressBar(total_frames)
                
                # 发送帧数据
                for frame_bytes in frame_generator():
                    pbar.update(1)
                    output_process.send(frame_bytes)
                
                # 完成处理
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                
                # 然后合并音频（参考VHS原始实现）
                try:
                    # 为音频文件单独检查文件名，确保不会覆盖现有文件
                    audio_counter = 1
                    while True:
                        output_file_with_audio = f"{filename}_{audio_counter:05d}-audio.mp4"
                        output_file_with_audio_path = os.path.join(output_dir, output_file_with_audio)
                        if not os.path.exists(output_file_with_audio_path):
                            break
                        audio_counter += 1
                    
                    # 设置音频编码参数
                    audio_pass = ["-c:a", "aac"]  # 使用AAC音频编码
                    
                    # 获取音频参数
                    channels = audio['waveform'].size(1)
                    sample_rate = audio['sample_rate']
                    min_audio_dur = total_frames / frame_rate + 1
                    
                    # 设置音频填充参数
                    if trim_to_audio:
                        apad = []
                    else:
                        apad = ["-af", f"apad=whole_dur={min_audio_dur}"]
                    
                    # 查找FFmpeg路径
                    import subprocess
                    import shutil
                    
                    ffmpeg_path = shutil.which("ffmpeg")
                    if not ffmpeg_path:
                        # 尝试常见路径
                        possible_paths = [
                            "ffmpeg",
                            "ffmpeg.exe",
                            os.path.join(os.path.dirname(sys.executable), "ffmpeg.exe"),
                            os.path.join(os.path.dirname(sys.executable), "Scripts", "ffmpeg.exe")
                        ]
                        for path in possible_paths:
                            if shutil.which(path):
                                ffmpeg_path = path
                                break
                        else:
                            raise Exception("FFmpeg not found in system PATH")
                    
                    # 构建FFmpeg命令（参考VHS实现）
                    mux_args = [ffmpeg_path, "-v", "error", "-y", "-i", file_path,
                               "-ar", str(sample_rate), "-ac", str(channels),
                               "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                               + audio_pass + apad + ["-shortest", output_file_with_audio_path]
                    
                    # 准备音频数据
                    audio_data = audio['waveform'].squeeze(0).transpose(0,1).numpy().tobytes()
                    
                    # 执行FFmpeg命令
                    res = subprocess.run(mux_args, input=audio_data, 
                                       capture_output=True, check=True)
                    
                    if res.stderr:
                        print(res.stderr.decode('utf-8'), end="")
                    
                    final_file_path = output_file_with_audio_path
                    final_filename = output_file_with_audio
                    
                    # 删除无音频版本的文件（实现VHS_KeepIntermediate=False的效果）
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as remove_error:
                        pass
                    
                except Exception as e:
                    # 不生成无音频版本，直接抛出异常
                    raise Exception(f"音频处理失败，无法生成带音频的视频: {str(e)}")
            else:
                # 没有有效音频数据，生成无音频视频
                output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env, total_frames)
                output_process.send(None)  # 启动生成器
                
                # 创建进度条
                pbar = ProgressBar(total_frames)
                
                # 发送帧数据
                for frame_bytes in frame_generator():
                    pbar.update(1)
                    output_process.send(frame_bytes)
                
                # 完成处理
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
        else:
            # 没有音频输入，生成无音频视频
            output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env, total_frames)
            output_process.send(None)  # 启动生成器
            
            # 创建进度条
            pbar = ProgressBar(total_frames)
            
            # 发送帧数据
            for frame_bytes in frame_generator():
                pbar.update(1)
                output_process.send(frame_bytes)
            
            # 关闭管道并等待完成
            try:
                total_frames_output = output_process.send(None)
                output_process.send(None)
            except StopIteration:
                pass
        
        # 获取最后几帧图像
        if last_frames_count > 0 and len(batched_output) > 0:
            start_idx = max(0, len(batched_output) - last_frames_count)
            last_frames_images = batched_output[start_idx:]
        else:
            # 如果不需要最后几帧或没有图像，返回空的tensor
            if len(batched_output) > 0:
                last_frames_images = torch.empty(0, *batched_output.shape[1:])
            else:
                # 创建一个默认形状的空tensor
                last_frames_images = torch.empty(0, 3, 512, 512)  # 默认形状

        return (final_file_path, final_filename, last_frames_images)
    

class VideoMergeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_paths": ("STRING", {
                    "multiline": True,
                    "tooltip": "视频文件路径，支持多行输入或逗号分隔的多个路径"
                }),
                "output_filename": ("STRING", {
                    "default": "merged_video.mp4",
                    "tooltip": "输出合成视频的文件名"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "可选的背景音频，如果提供则替换视频音频，否则保留原音频"
                }),
                "video_paths_list": ("*", {
                    "tooltip": "可选的视频路径列表输入，如果提供则优先使用此输入"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    CATEGORY = "lianlaoshi/video"
    FUNCTION = "merge_videos"

    def __init__(self):
        self.ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        """查找FFmpeg可执行文件"""
        # 检查常见路径
        possible_paths = [
            "ffmpeg",  # 系统PATH中
            "ffmpeg.exe",  # Windows
            r"C:\ffmpeg\bin\ffmpeg.exe",  # 常见Windows安装路径
            "/usr/bin/ffmpeg",  # Linux
            "/usr/local/bin/ffmpeg",  # macOS
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, "-version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")

    def _validate_video_files(self, video_paths: List[str]) -> List[str]:
        """验证视频文件是否存在且有效"""
        if not video_paths:
            raise ValueError("视频路径列表为空，无法合并")
        
        # 首先检查所有文件是否存在，如果有任何文件不存在就立即失败
        missing_files = []
        for path in video_paths:
            if not os.path.exists(path):
                missing_files.append(path)
                continue
            
            if not os.path.isfile(path):
                missing_files.append(path)
                continue
            
            # 检查文件大小
            file_size = os.path.getsize(path)
            if file_size == 0:
                missing_files.append(path)
                continue
        
        # 如果有任何文件缺失，抛出详细的错误信息
        if missing_files:
            error_msg = f"发现 {len(missing_files)} 个无效或缺失的视频文件:\n"
            for file in missing_files:
                error_msg += f"  - {file}\n"
            error_msg += "\n请确保所有视频文件都存在且不为空。"
            raise FileNotFoundError(error_msg)
        
        return video_paths

    def _get_audio_duration(self, audio_data) -> float:
        """获取音频数据的时长（秒）"""
        try:
            if audio_data is None:
                return 0.0
            
            # 处理LazyAudioMap格式
            if hasattr(audio_data, '__class__') and 'LazyAudioMap' in str(audio_data.__class__):
                try:
                    # 尝试获取waveform和sample_rate属性
                    if hasattr(audio_data, 'waveform') and hasattr(audio_data, 'sample_rate'):
                        waveform = audio_data.waveform
                        sample_rate = audio_data.sample_rate
                    # 尝试字典式访问
                    elif hasattr(audio_data, '__getitem__'):
                        waveform = audio_data['waveform']
                        sample_rate = audio_data['sample_rate']
                    # 尝试调用get_audio或load方法
                    elif hasattr(audio_data, 'get_audio'):
                        result = audio_data.get_audio()
                        waveform = result['waveform'] if isinstance(result, dict) else result[0]
                        sample_rate = result['sample_rate'] if isinstance(result, dict) else result[1]
                    elif hasattr(audio_data, 'load'):
                        result = audio_data.load()
                        waveform = result['waveform'] if isinstance(result, dict) else result[0]
                        sample_rate = result['sample_rate'] if isinstance(result, dict) else result[1]
                    else:
                        return 0.0
                    
                    # 处理提取的音频数据
                    audio_data = {"waveform": waveform, "sample_rate": sample_rate}
                except Exception as e:
                    return 0.0
            
            # 处理字典格式 {"waveform": tensor, "sample_rate": int}
            if isinstance(audio_data, dict):
                waveform = audio_data.get("waveform")
                sample_rate = audio_data.get("sample_rate", 44100)
                
                if waveform is None:
                    return 0.0
                    
                # 统一处理waveform维度
                if hasattr(waveform, 'shape'):
                    # 处理不同维度的音频数据
                    if len(waveform.shape) == 1:
                        # 1D: [samples] -> 转换为 [1, samples]
                        samples = waveform.shape[0]
                    elif len(waveform.shape) == 2:
                        # 2D: 可能是 [samples, channels] 或 [channels, samples]
                        # 通常channels数量较少，samples数量较多
                        if waveform.shape[0] <= waveform.shape[1]:
                            # [channels, samples] 格式
                            samples = waveform.shape[1]
                        else:
                            # [samples, channels] 格式，需要转置
                            samples = waveform.shape[0]
                    elif len(waveform.shape) == 3:
                        # 3D: [batch, channels, samples] 或 [batch, samples, channels]
                        if waveform.shape[1] <= waveform.shape[2]:
                            # [batch, channels, samples] 格式
                            samples = waveform.shape[2]
                        else:
                            # [batch, samples, channels] 格式
                            samples = waveform.shape[1]
                    else:
                        # 更高维度，取最后一个维度作为samples
                        samples = waveform.shape[-1]
                else:
                    # 如果没有shape属性，尝试获取长度
                    samples = len(waveform) if hasattr(waveform, '__len__') else 0
                    
                return samples / sample_rate if sample_rate > 0 else 0.0
            
            # 处理元组格式 (waveform, sample_rate)
            elif isinstance(audio_data, (tuple, list)) and len(audio_data) >= 2:
                waveform, sample_rate = audio_data[0], audio_data[1]
                
                # 统一处理waveform维度
                if hasattr(waveform, 'shape'):
                    # 处理不同维度的音频数据
                    if len(waveform.shape) == 1:
                        # 1D: [samples] -> 转换为 [1, samples]
                        samples = waveform.shape[0]
                    elif len(waveform.shape) == 2:
                        # 2D: 可能是 [samples, channels] 或 [channels, samples]
                        # 通常channels数量较少，samples数量较多
                        if waveform.shape[0] <= waveform.shape[1]:
                            # [channels, samples] 格式
                            samples = waveform.shape[1]
                        else:
                            # [samples, channels] 格式，需要转置
                            samples = waveform.shape[0]
                    elif len(waveform.shape) == 3:
                        # 3D: [batch, channels, samples] 或 [batch, samples, channels]
                        if waveform.shape[1] <= waveform.shape[2]:
                            # [batch, channels, samples] 格式
                            samples = waveform.shape[2]
                        else:
                            # [batch, samples, channels] 格式
                            samples = waveform.shape[1]
                    else:
                        # 更高维度，取最后一个维度作为samples
                        samples = waveform.shape[-1]
                else:
                    # 如果没有shape属性，尝试获取长度
                    samples = len(waveform) if hasattr(waveform, '__len__') else 0
                
                return samples / sample_rate if sample_rate > 0 else 0.0
            
            # 处理直接的tensor格式
            elif hasattr(audio_data, 'shape'):
                # 假设采样率为44100，这是一个默认值
                sample_rate = 44100
                
                # 统一处理waveform维度
                if len(audio_data.shape) == 1:
                    # 1D: [samples] -> 转换为 [1, samples]
                    samples = audio_data.shape[0]
                elif len(audio_data.shape) == 2:
                    # 2D: 可能是 [samples, channels] 或 [channels, samples]
                    # 通常channels数量较少，samples数量较多
                    if audio_data.shape[0] <= audio_data.shape[1]:
                        # [channels, samples] 格式
                        samples = audio_data.shape[1]
                    else:
                        # [samples, channels] 格式，需要转置
                        samples = audio_data.shape[0]
                elif len(audio_data.shape) == 3:
                    # 3D: [batch, channels, samples] 或 [batch, samples, channels]
                    if audio_data.shape[1] <= audio_data.shape[2]:
                        # [batch, channels, samples] 格式
                        samples = audio_data.shape[2]
                    else:
                        # [batch, samples, channels] 格式
                        samples = audio_data.shape[1]
                else:
                    # 更高维度，取最后一个维度作为samples
                    samples = audio_data.shape[-1]
                
                return samples / sample_rate
            
            else:
                return 0.0
                
        except Exception as e:
            return 0.0

    def _save_audio_to_temp(self, audio_data) -> str:
        """保存音频数据到临时文件，使用scipy.io.wavfile确保兼容性"""
        if audio_data is None:
            return ""
        
        try:
            import tempfile
            import scipy.io.wavfile
            
            waveform = None
            sample_rate = 44100
            
            # 处理LazyAudioMap格式
            if hasattr(audio_data, '__class__') and 'LazyAudioMap' in str(audio_data.__class__):
                try:
                    # 尝试获取waveform和sample_rate属性
                    if hasattr(audio_data, 'waveform') and hasattr(audio_data, 'sample_rate'):
                        waveform = audio_data.waveform
                        sample_rate = audio_data.sample_rate
                    # 尝试字典式访问
                    elif hasattr(audio_data, '__getitem__'):
                        try:
                            waveform = audio_data['waveform']
                            sample_rate = audio_data['sample_rate']
                        except:
                            waveform = None
                    # 尝试调用get_audio或load方法
                    elif hasattr(audio_data, 'get_audio'):
                        result = audio_data.get_audio()
                        waveform = result['waveform'] if isinstance(result, dict) else result[0]
                        sample_rate = result['sample_rate'] if isinstance(result, dict) else result[1]
                    elif hasattr(audio_data, 'load'):
                        result = audio_data.load()
                        waveform = result['waveform'] if isinstance(result, dict) else result[0]
                        sample_rate = result['sample_rate'] if isinstance(result, dict) else result[1]
                    
                    # 如果上述方法都失败，尝试直接转换
                    if waveform is None:
                        return ""
                    
                    # 处理提取的音频数据
                    audio_data = {"waveform": waveform, "sample_rate": sample_rate}
                except Exception as e:
                    return ""
            
            # 处理不同的音频数据格式
            if isinstance(audio_data, dict):
                # 标准格式：{"waveform": tensor, "sample_rate": int}
                waveform = audio_data.get("waveform")
                sample_rate = audio_data.get("sample_rate", 44100)
            elif isinstance(audio_data, (list, tuple)) and len(audio_data) >= 2:
                # 元组/列表格式：(waveform, sample_rate)
                waveform = audio_data[0]
                sample_rate = audio_data[1] if len(audio_data) > 1 else 44100
            elif hasattr(audio_data, 'shape'):
                # 直接是tensor格式
                waveform = audio_data
                sample_rate = 44100
            else:
                return ""
            
            if waveform is None:
                return ""
            
            # 转换为numpy数组
            if hasattr(waveform, 'cpu'):
                waveform = waveform.cpu()
            if hasattr(waveform, 'numpy'):
                waveform = waveform.numpy()
            
            # 处理不同维度的音频数据
            original_shape = waveform.shape
            
            # 统一处理为2D格式 [channels, samples]
            if len(waveform.shape) == 1:
                # 1D: [samples] -> [1, samples]
                waveform = waveform.reshape(1, -1)
            elif len(waveform.shape) == 2:
                # 2D: 检查是否需要转置
                if waveform.shape[0] > waveform.shape[1]:
                    # 可能是 [samples, channels] -> [channels, samples]
                    waveform = waveform.T
            elif len(waveform.shape) == 3:
                # 3D: [batch, channels, samples] -> [channels, samples]
                if waveform.shape[0] == 1:
                    waveform = waveform.squeeze(0)
                else:
                    # 取第一个batch
                    waveform = waveform[0]
            elif len(waveform.shape) > 3:
                # 高维数据：尝试压缩到2D
                # 保留最后两个维度，压缩其他维度
                new_shape = (-1, waveform.shape[-1])
                waveform = waveform.reshape(new_shape)
            
            # 确保是2D格式 [channels, samples]
            if len(waveform.shape) != 2:
                return ""
            
            channels, samples = waveform.shape
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 准备音频数据用于scipy保存
            # scipy.io.wavfile.write期望的格式是 [samples, channels] 或 [samples] (单声道)
            if channels == 1:
                # 单声道：[samples]
                audio_data_for_save = waveform[0]
            else:
                # 多声道：[samples, channels]
                audio_data_for_save = waveform.T
            
            # 转换为16位整数格式
            if audio_data_for_save.dtype != np.int16:
                # 假设输入是浮点数在[-1, 1]范围内
                if audio_data_for_save.max() <= 1.0 and audio_data_for_save.min() >= -1.0:
                    audio_data_for_save = (audio_data_for_save * 32767).astype(np.int16)
                else:
                    # 如果不在[-1,1]范围，进行归一化
                    max_val = max(abs(audio_data_for_save.max()), abs(audio_data_for_save.min()))
                    if max_val > 0:
                        audio_data_for_save = (audio_data_for_save / max_val * 32767).astype(np.int16)
                    else:
                        audio_data_for_save = audio_data_for_save.astype(np.int16)
            
            # 使用scipy保存音频文件
            scipy.io.wavfile.write(temp_path, sample_rate, audio_data_for_save)
            
            return temp_path
            
        except ImportError:
            return ""
        except Exception as e:
            return ""

    def merge_videos(
        self,
        video_paths,
        output_filename: str,
        audio=None,
        video_paths_list=None,
        **kwargs
    ) -> Dict[str, Any]:
        
        try:
            # 处理音频输入
            temp_audio_path = ""
            if audio is not None:
                try:
                    temp_audio_path = self._save_audio_to_temp(audio)
                    if temp_audio_path:
                        audio_duration = self._get_audio_duration(audio)
                except Exception as e:
                    temp_audio_path = ""
            
            # 优先使用列表输入，如果提供的话
            if video_paths_list is not None:
                if hasattr(video_paths_list, '__iter__') and not isinstance(video_paths_list, str):
                    # 处理可迭代对象（列表、元组、集合等），但排除字符串
                    path_list = [str(path) for path in video_paths_list]
                    valid_paths = self._validate_video_files(path_list)
                else:
                    # 如果不是可迭代对象，转换为列表
                    valid_paths = self._validate_video_files([str(video_paths_list)])
            elif isinstance(video_paths, str):
                # 如果是字符串，尝试按行分割
                path_list = [path.strip() for path in video_paths.split('\n') if path.strip()]
                if not path_list:
                    # 如果分割后为空，尝试按逗号分割
                    path_list = [path.strip() for path in video_paths.split(',') if path.strip()]
                if not path_list:
                    # 如果还是为空，将整个字符串作为单个路径
                    path_list = [video_paths.strip()]
                
                valid_paths = self._validate_video_files(path_list)
            elif hasattr(video_paths, '__iter__'):
                # 处理可迭代对象（列表、元组、集合等字符串序列）
                try:
                    path_list = [str(path) for path in video_paths]
                    valid_paths = self._validate_video_files(path_list)
                except Exception as e:
                    raise ValueError(f"Failed to process iterable video_paths: {e}")
            else:
                raise ValueError(f"video_paths must be a string or iterable, got {type(video_paths)}")
            
            if len(valid_paths) < 2:
                raise ValueError(f"At least 2 valid video files are required for merging, but only found {len(valid_paths)} valid files")
            
            # 确定输出目录和文件路径
            output_dir = folder_paths.get_output_directory()
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成唯一文件名
            base_name, ext = os.path.splitext(output_filename)
            if not ext:
                ext = '.mp4'
            
            counter = 1
            final_filename = f"{base_name}_{counter:05}{ext}"
            output_path = os.path.join(output_dir, final_filename)
            
            while os.path.exists(output_path):
                counter += 1
                final_filename = f"{base_name}_{counter:05}{ext}"
                output_path = os.path.join(output_dir, final_filename)
            
            # 执行视频合并
            total_duration = self._execute_merge(
                valid_paths, output_path, temp_audio_path
            )
            
            # 清理临时音频文件
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e:
                    pass
            
            # 验证输出文件
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("Merged video file was not created or is empty")
            
            absolute_output_path = os.path.abspath(output_path)
            
            return {
                "ui": {},
                "result": (absolute_output_path,)
            }
            
        except Exception as e:
            # 确保在异常情况下也清理临时音频文件
            if 'temp_audio_path' in locals() and temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass  # 忽略清理时的错误
            
            raise RuntimeError(f"Video merge failed: {str(e)}")

    def _execute_merge(self, video_paths: List[str], output_path: str, 
                      temp_audio_path: str) -> float:
        """执行视频合并"""
        return self._merge_concat(video_paths, output_path, temp_audio_path)

    def _merge_concat(self, video_paths: List[str], output_path: str, 
                     temp_audio_path: str) -> float:
        """使用concat方法无损合并视频（支持背景音频添加）"""
        
        # 验证输入参数
        if not video_paths:
            raise ValueError("No video paths provided for merging")
        
        import tempfile
        
        # 创建临时文件列表（使用UTF-8编码确保路径正确处理）
        concat_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                concat_file = f.name
                for video_path in video_paths:
                    # 使用绝对路径并正确转义
                    abs_path = os.path.abspath(video_path)
                    f.write(f"file '{abs_path}'\n")
            
            # 第一步：无损合并视频（copy流，避免重编码）
            cmd_video = [
                self.ffmpeg_path, "-y", 
                "-f", "concat", 
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",                    # 完全无损复制
                "-avoid_negative_ts", "make_zero",  # 避免时间戳问题
                output_path
            ]
            
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"视频合并失败:\n{result.stderr}")
            
            # 第二步：如果提供了音频，合并音视频
            if temp_audio_path and os.path.exists(temp_audio_path):
                # 获取视频时长，确保音频不超过视频长度
                video_duration = self._get_video_duration(output_path)
                audio_duration = self._get_audio_duration_from_file(temp_audio_path)
                
                # 如果音频长度超过视频长度，裁剪音频
                final_audio_path = temp_audio_path
                if audio_duration > video_duration:
                    final_audio_path = self._trim_audio_to_duration(temp_audio_path, video_duration)
                
                # 创建带音频的最终文件
                temp_output = output_path.replace(".mp4", "_with_audio.mp4")
                cmd_audio = [
                    self.ffmpeg_path, "-y",
                    "-i", output_path,           # 已合并的视频
                    "-i", final_audio_path,      # 音频文件
                    "-map", "0:v",               # 映射视频流
                    "-map", "1:a",               # 映射音频流
                    "-c:v", "copy",              # 视频流保持不变
                    "-c:a", "aac",               # 音频编码为AAC
                    "-b:a", "128k",              # 音频比特率
                    "-shortest",                 # 以最短的为准
                    temp_output
                ]
                
                result = subprocess.run(cmd_audio, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"音频合并失败:\n{result.stderr}")
                
                # 替换原视频文件
                if os.path.exists(temp_output):
                    os.replace(temp_output, output_path)
            
            # 获取最终视频时长
            return self._get_video_duration(output_path)
            
        except Exception as e:
            raise
        finally:
            # 清理临时文件
            if concat_file and os.path.exists(concat_file):
                try:
                    os.unlink(concat_file)
                except Exception:
                    pass

    def _get_audio_duration_from_file(self, audio_path: str) -> float:
        """从音频文件获取时长"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 从stderr中解析时长信息
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    # 格式: Duration: 00:01:23.45, start: 0.000000, bitrate: 1234 kb/s
                    duration_str = line.split('Duration: ')[1].split(',')[0]
                    # 解析 HH:MM:SS.ss 格式
                    time_parts = duration_str.split(':')
                    if len(time_parts) == 3:
                        hours = float(time_parts[0])
                        minutes = float(time_parts[1])
                        seconds = float(time_parts[2])
                        return hours * 3600 + minutes * 60 + seconds
            
            return 0.0
            
        except Exception as e:
            return 0.0

    def _trim_audio_to_duration(self, audio_path: str, target_duration: float) -> str:
        """裁剪音频到指定时长"""
        try:
            import tempfile
            
            # 创建临时输出文件
            temp_dir = tempfile.gettempdir()
            trimmed_audio_path = os.path.join(temp_dir, f"trimmed_audio_{os.getpid()}.wav")
            
            cmd = [
                self.ffmpeg_path,
                "-i", audio_path,
                "-t", str(target_duration),  # 裁剪到指定时长
                "-c:a", "pcm_s16le",  # 使用PCM编码确保兼容性
                "-y", trimmed_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return audio_path  # 返回原音频路径
            
            return trimmed_audio_path
            
        except Exception as e:
            return audio_path  # 返回原音频路径

    def _get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-f", "null",
                "-"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 从stderr中解析时长信息
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    # 格式: Duration: 00:01:23.45, start: 0.000000, bitrate: 1234 kb/s
                    duration_str = line.split('Duration: ')[1].split(',')[0]
                    # 解析 HH:MM:SS.ss 格式
                    time_parts = duration_str.split(':')
                    if len(time_parts) == 3:
                        hours = float(time_parts[0])
                        minutes = float(time_parts[1])
                        seconds = float(time_parts[2])
                        return hours * 3600 + minutes * 60 + seconds
            
            return 0.0
            
        except Exception as e:
            return 0.0

# 节点映射
NODE_CLASS_MAPPINGS = {
    "VideoCombineNode": VideoCombineNode,
    "VideoMergeNode": VideoMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCombineNode": "lian Video Combine 🎬",
    "VideoMergeNode": "lian Video Merge 🎞️",
}