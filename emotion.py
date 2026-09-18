# emotion.py

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import librosa
import librosa.display
from pathlib import Path

# 在文件开头添加这行，设置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    HubertPreTrainedModel,
    HubertModel,
)

# ==== 模型配置 ====
MODEL_NAME = "xmj2002/hubert-base-ch-speech-emotion-recognition"  # 训练好的中文情绪模型
SAMPLE_RATE = 16000
DURATION = 6  # 统一到 6 秒

# id -> 英文标签
ID2LABEL_EN = {
    0: "angry",
    1: "fear",
    2: "happy",
    3: "neutral",
    4: "sad",
    5: "surprise",
}

# id -> 中文标签（你可以按自己需要改）
ID2LABEL_ZH = {
    0: "生气",
    1: "害怕",
    2: "高兴",
    3: "中性",
    4: "伤心",
    5: "惊讶",
}


# ==== 按模型卡定义分类头 ====
class HubertClassificationHead(nn.Module):
    def __init__(self, config):
        super(HubertClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class HubertForSpeechClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super(HubertForSpeechClassification, self).__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]              # [batch, time, hidden]
        x = torch.mean(hidden_states, dim=1)    # -> [batch, hidden]
        x = self.classifier(x)                 # -> [batch, num_class]
        return x


# ==== 全局缓存 ====
_processor = None
_model = None
_device = None


def plot_emotion_analysis(wav_path, label, score, all_scores, save_path=None):
    """
    绘制音频频谱图和情感得分分布图 (无需 platform 库的字体方案)
    """
    # --- 改进的字体搜索逻辑 ---
    # 定义常见的中文字体名称（Windows, macOS, Linux 常用名）
    chinese_font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STHeiti', 'WenQuanYi Micro Hei']

    # 获取系统所有可用字体名
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 自动匹配第一个存在的字体
    for font in chinese_font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    else:
        # 如果都没找到，尝试使用系统默认的无衬线字体
        plt.rcParams['font.sans-serif'] = ['sans-serif']

    # 修复负号显示问题（必须，否则分贝值为负时会显示方块）
    plt.rcParams['axes.unicode_minus'] = False
    # -----------------------

    # 1. 加载音频 (假设 SAMPLE_RATE 已经在外部定义)
    # 建议直接读取原生采样率，避免重采样导致的信息损失
    y, sr = librosa.load(wav_path, sr=None)

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"文件: {Path(wav_path).name} | 判定结果: {label} ({score:.2%})", fontsize=16)

    # 2. 绘制梅尔频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax1)
    ax1.set_title("梅尔频谱图 (模型分析的基础)")
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')

    # 3. 绘制情感概率柱状图
    emotions = list(all_scores.keys())
    probs = list(all_scores.values())
    colors = ['orange' if e == label else 'skyblue' for e in emotions]

    bars = ax2.barh(emotions, probs, color=colors)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("概率 (置信度)")
    ax2.set_title("各情感维度得分明细")

    # 在柱状图上标注数值
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        print(f"分析图表已保存至: {save_path}")

    plt.show()

def _get_device(prefer_gpu=False):
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_model(prefer_gpu=False):
    global _processor, _model, _device

    if _model is not None:
        return _processor, _model, _device

    device = _get_device(prefer_gpu)
    print("使用设备:", device)
    print("正在加载模型:", MODEL_NAME, "（第一次会从 HuggingFace 下载）...")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertForSpeechClassification.from_pretrained(MODEL_NAME, config=config)

    model.to(device)
    model.eval()

    _processor = processor
    _model = model
    _device = device

    print("模型加载完成。")
    return processor, model, device


# ===== 单文件预测 =====
def predict_emotion_file(wav_path, prefer_gpu=False):
    """
    识别单个 wav 文件情绪，返回 (中文情绪, 置信度, 完整得分字典)
    """
    processor, model, device = _load_model(prefer_gpu)

    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError("找不到音频文件: %s" % wav_path)

    # 1. 读音频
    speech, sr = librosa.load(path=str(wav_path), sr=SAMPLE_RATE, mono=True)

    # 2. 统一长度到 DURATION 秒
    max_len = SAMPLE_RATE * DURATION
    if len(speech) > max_len:
        speech = speech[:max_len]
    else:
        pad_len = max_len - len(speech)
        if pad_len > 0:
            speech = F.pad(torch.tensor(speech), (0, pad_len)).numpy()

    # 3. 特征提取
    inputs = processor(
        speech,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    input_values = inputs.input_values.to(device)

    # 4. 模型推理
    with torch.no_grad():
        logits = model(input_values)
        # 使用 Softmax 将原始输出转为 0~1 的概率
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_id = int(torch.argmax(logits, dim=1).cpu().numpy()[0])

    # ==== 核心增强部分 ====
    # 将 ID2LABEL_ZH 中的每一个标签与对应的概率匹配起来
    # 结果类似: {'生气': 0.05, '害怕': 0.01, '高兴': 0.8, ...}
    all_scores = {ID2LABEL_ZH[i]: float(probs[i]) for i in range(len(ID2LABEL_ZH))}

    zh_label = ID2LABEL_ZH.get(pred_id, "未知")
    score = float(probs[pred_id])

    # 注意：这里现在返回三个值了！
    return zh_label, score, all_scores


# ===== 文件夹预测 =====
def recognize_emotions_in_folder(folder_name="emotion", prefer_gpu=False, show_plot=True):
    """
    识别当前脚本同级目录下 folder_name 中所有 wav 的情绪。

    返回:
        { 文件名: (中文情绪, 置信度) }
    """
    base_dir = Path(__file__).resolve().parent
    emotion_dir = base_dir / folder_name

    if not emotion_dir.is_dir():
        raise FileNotFoundError(
            "找不到情感音频文件夹: %s\n请在与 emotion.py 同级目录下创建 '%s' 并放入 .wav 文件。" %
            (emotion_dir, folder_name)
        )

    wav_files = sorted(
        [p for p in emotion_dir.iterdir() if p.suffix.lower() == ".wav"]
    )
    if not wav_files:
        print("文件夹 %s 中没有 .wav 文件。" % emotion_dir)
        return {}

    # 确保模型加载
    _load_model(prefer_gpu)

    results = {}

    print("在 %s 中找到 %d 个 wav，开始识别情绪..." % (emotion_dir, len(wav_files)))
    for wav_path in wav_files:
        try:
            label, score, all_scores = predict_emotion_file(wav_path, prefer_gpu)
            results[wav_path.name] = (label, score)
            print("音频: %s -> 情绪: %s（得分: %.4f）" % (wav_path.name, label, score))
            if show_plot:
                # 调用我上一步给你的那个绘图函数
                plot_emotion_analysis(wav_path, label, score, all_scores)
        except Exception as e:
            print("处理 %s 时出错: %s" % (wav_path.name, e))

    print("全部文件情感识别完成。")
    return results


# 直接运行 emotion.py 做一个简单测试
if __name__ == "__main__":
    recognize_emotions_in_folder()

