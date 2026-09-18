import os
import sys
import glob
import whisper
# 自动把当前 conda 的 ffmpeg 路径加入 PATH
ffmpeg_dir = os.path.join(sys.prefix, "Library", "bin")
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

print("已加入 ffmpeg 路径：", ffmpeg_dir)



def convert_audio_to_text():
    # 显示当前脚本真实路径
    print("当前脚本路径：", os.path.abspath(__file__))

    # 显示当前工作目录
    print("当前工作目录：", os.getcwd())

    # 1. 加载 Whisper 模型
    print("加载模型中...")
    model = whisper.load_model("small")

    # 2. 找到 wait_transfer 文件夹
    base_dir = os.path.dirname(os.path.abspath(__file__))
    audio_dir = os.path.join(base_dir, "wait_transfer")

    print("目标音频目录：", audio_dir)

    if not os.path.exists(audio_dir):
        print("没找到 wait_transfer 文件夹")
        return

    # 3. 搜索音频文件
    audio_files = glob.glob(os.path.join(audio_dir, "*.mp3")) \
                 + glob.glob(os.path.join(audio_dir, "*.wav")) \
                 + glob.glob(os.path.join(audio_dir, "*.m4a")) \
                 + glob.glob(os.path.join(audio_dir, "*.flac"))

    print("找到的音频文件：", audio_files)

    if not audio_files:
        print("没找到任何音频文件")
        return

    # 4. 执行识别
    for audio_path in audio_files:
        print(f"正在识别：{audio_path}")

        result = model.transcribe(audio_path, language="zh")
        text = result["text"].strip()

        # 输出保存的文件名
        txt_path = audio_path + ".txt"
        print("保存文本到：", txt_path)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

    print("完成")

if __name__ == "__main__":
    convert_audio_to_text()