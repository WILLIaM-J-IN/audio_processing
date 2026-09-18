# real_time_recorder.py
import pyaudio
import wave
import threading
import time
import os
import tempfile
from datetime import datetime
import numpy as np

# 设置HF镜像（避免下载问题）
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入现有模块的功能
from emotion import predict_emotion_file


class RealTimeRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = pyaudio.paInt16
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.current_emotion = "等待录音..."
        self.current_confidence = 0.0
        self.last_saved_file = None
        self.record_thread = None
        self.ui_update_callback = None  # 新增：UI更新回调函数

    def set_ui_update_callback(self, callback):
        """设置UI更新回调函数"""
        self.ui_update_callback = callback

    def start_recording(self):
        """开始录音（无限时长，直到手动停止）"""
        if self.is_recording:
            return False

        self.frames = []
        self.is_recording = True
        self.last_saved_file = None
        self.current_emotion = "录音中..."
        self.current_confidence = 0.0

        # 立即通知UI更新
        self._notify_ui_update()

        try:
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            # 在新线程中录音（无限循环）
            self.record_thread = threading.Thread(
                target=self._record_audio
            )
            self.record_thread.daemon = True
            self.record_thread.start()

            return True

        except Exception as e:
            print(f"录音启动失败: {e}")
            self.is_recording = False
            self.current_emotion = f"录音失败: {e}"
            self._notify_ui_update()
            return False

    def _record_audio(self):
        """实际录音过程（无限循环直到停止）"""
        print("开始录音（点击停止按钮结束录音）...")

        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"录音错误: {e}")
                break

        # 录音停止后自动保存和分析
        if self.frames:
            self._analyze_and_save_recording()

    def stop_recording(self):
        """停止录音"""
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

            self.current_emotion = "分析中..."
            self.current_confidence = 0.0
            self._notify_ui_update()  # 立即更新UI

            print("录音停止")

            # 等待录音线程结束
            if self.record_thread and self.record_thread.is_alive():
                self.record_thread.join(timeout=2.0)

    def _analyze_and_save_recording(self):
        """分析录音的情感并保存文件"""
        try:
            # 创建emotion文件夹（如果不存在）
            emotion_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion")
            os.makedirs(emotion_dir, exist_ok=True)

            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_filename = f"realtime_{timestamp}.wav"
            saved_filepath = os.path.join(emotion_dir, saved_filename)

            # 保存到emotion文件夹
            self._save_wav_file(saved_filepath)
            self.last_saved_file = saved_filename

            print(f"录音已保存: {saved_filepath}")
            print(f"录音时长: {len(self.frames) * self.chunk_size / self.sample_rate:.2f}秒")

            # 使用现有的情感识别功能分析
            emotion, confidence = predict_emotion_file(saved_filepath)

            self.current_emotion = emotion
            self.current_confidence = confidence

            print(f"分析结果: {emotion} (置信度: {confidence:.4f})")

            # 分析完成后立即通知UI更新
            self._notify_ui_update()

        except Exception as e:
            print(f"情感分析失败: {e}")
            self.current_emotion = f"分析失败: {e}"
            self.current_confidence = 0.0
            self._notify_ui_update()

    def _save_wav_file(self, filename):
        """保存为WAV文件"""
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def _notify_ui_update(self):
        """通知UI更新显示"""
        if self.ui_update_callback:
            try:
                self.ui_update_callback(self.current_emotion, self.current_confidence)
            except Exception as e:
                print(f"UI更新回调失败: {e}")

    def get_current_result(self):
        """获取当前分析结果"""
        return self.current_emotion, self.current_confidence

    def get_last_saved_file(self):
        """获取最后保存的文件名"""
        return self.last_saved_file

    def get_recording_status(self):
        """获取录音状态"""
        return self.is_recording

    def cleanup(self):
        """清理资源"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


# 全局实例
_recorder_instance = None


def get_recorder():
    """获取录音器实例（单例模式）"""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = RealTimeRecorder()
    return _recorder_instance


def set_ui_update_callback(callback):
    """设置UI更新回调"""
    recorder = get_recorder()
    recorder.set_ui_update_callback(callback)


def start_realtime_emotion():
    """开始实时情感分析录音（无限时长）"""
    recorder = get_recorder()
    return recorder.start_recording()


def stop_realtime_emotion():
    """停止录音"""
    recorder = get_recorder()
    recorder.stop_recording()


def get_realtime_result():
    """获取实时分析结果"""
    recorder = get_recorder()
    return recorder.get_current_result()


def get_last_saved_filename():
    """获取最后保存的文件名"""
    recorder = get_recorder()
    return recorder.get_last_saved_file()


def get_recording_status():
    """获取录音状态"""
    recorder = get_recorder()
    return recorder.get_recording_status()


def cleanup_recorder():
    """清理录音器"""
    global _recorder_instance
    if _recorder_instance:
        _recorder_instance.cleanup()
        _recorder_instance = None