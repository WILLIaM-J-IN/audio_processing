import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np


def herta_ultimate_cleaner():
    # 路径配置
    input_dir = r'E:\PythonProject\PythonProject_csh\source_for_enhance'
    output_dir = r'E:\PythonProject\PythonProject_csh\source_enhanced'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    if not files:
        print("[-] 文件夹里空空如也，你在耍我？")
        return

    print(f"[*] 启动终极降噪引擎，准备清理 {len(files)} 个文件...")

    for filename in files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # 1. 读取音频
            y, sr = librosa.load(input_path, sr=None)

            # 2. 强力降噪 (Spectral Gating)
            # stationary=False 表示处理非平稳噪声（比如突然的背景声）
            # prop_decrease=1.0 表示 100% 压制识别到的噪声
            # n_std_thresh_stationary=1.5 降低门限，让降噪更灵敏
            reduced_noise = nr.reduce_noise(
                y=y,
                sr=sr,
                stationary=False,
                prop_decrease=1.0,
                n_fft=2048,
                time_mask_smooth_ms=64
            )

            # 3. 人声激进增强 (Speech Boosting)
            # 通过 FFT 提取频谱，对人声核心频段 (500Hz - 4000Hz) 进行非线性增益
            stft = librosa.stft(reduced_noise)
            freqs = librosa.fft_frequencies(sr=sr)

            # 专门针对人声频段的提升掩码
            speech_mask = np.logical_and(freqs >= 500, freqs <= 4000)
            stft[speech_mask, :] *= 1.5  # 1.5倍人声频段增益

            # 4. 重构并去除多余的静音噪音
            y_enhanced = librosa.istft(stft)

            # 5. 动态压缩与标准化
            # 防止降噪后声音太小，将其拉伸到 0dB 附近
            y_final = librosa.util.normalize(y_enhanced)

            # 6. 保存
            sf.write(output_path, y_final, sr)
            print(f"[√] 强效处理完毕: {filename}")

        except Exception as e:
            print(f"[×] 出错啦: {e}")

    print("\n[!] 全部搞定。如果这还不行，那就是你录音设备的问题，或者你的噪声已经把人声盖得一点不剩了。")


if __name__ == "__main__":
    herta_ultimate_cleaner()