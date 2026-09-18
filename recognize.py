# -*- coding: utf-8 -*-
"""
说话人识别模块：从 speakers/ 和 speakers_wait/ 中做声纹识别

在你的主程序中，只需要：
    from speaker_recognizer import recognize_wait_dir
    recognize_wait_dir()

即可在控制台打印出识别结果。
"""

import os
from typing import Dict, Tuple, Optional
# 在文件开头添加这行，设置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy  # 关键：避免 Windows symlink 问题

# ================== 可配置参数（你可以按需改） ==================
DEFAULT_SPEAKERS_DIR = "speakers"        # 已注册说话人模板目录
DEFAULT_WAIT_DIR = "speakers_wait"       # 待识别音频目录
DEFAULT_THRESHOLD = 0.6                  # 相似度阈值
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

# ================== 全局变量（懒加载） ==================
_verification: Optional[SpeakerRecognition] = None
_enrolled_speakers: Optional[Dict[str, str]] = None


def _load_enrolled_speakers(speakers_dir: str) -> Dict[str, str]:
    """
    从 speakers_dir 中加载已注册说话人模板。
    约定：文件名(不含后缀) = 说话人名字。
    返回: {speaker_name: file_path}
    """
    speakers: Dict[str, str] = {}

    if not os.path.isdir(speakers_dir):
        print(f"[错误] 未找到说话人模板目录: {speakers_dir}")
        return speakers

    for fname in os.listdir(speakers_dir):
        if fname.lower().endswith(AUDIO_EXTS):
            spk_name = os.path.splitext(fname)[0]
            fpath = os.path.join(speakers_dir, fname)
            speakers[spk_name] = fpath

    print(f"[信息] 已加载 {len(speakers)} 个注册说话人: {list(speakers.keys())}")
    return speakers


def _init_model_and_speakers(speakers_dir: str):
    """
    懒加载：第一次调用时才加载模型和说话人模板
    """
    global _verification, _enrolled_speakers

    if _verification is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[信息] 使用设备: {device}")
        print("[信息] 正在加载 ECAPA 说话人识别模型（speechbrain/spkrec-ecapa-voxceleb）...")

        _verification = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            local_strategy=LocalStrategy.COPY,  # 避免 Win 下的 symlink 权限问题
        )

    if _enrolled_speakers is None:
        _enrolled_speakers = _load_enrolled_speakers(speakers_dir)


def _recognize_one_file(
    test_file: str,
    threshold: float,
) -> Tuple[Optional[str], float]:
    """
    对单个 test_file 进行说话人识别：
      - 与 _enrolled_speakers 中每个说话人模板比较
      - 取最高分作为最终结果
    返回:
      (最佳说话人名字 或 None, 最佳得分)
    """
    if _verification is None or _enrolled_speakers is None:
        raise RuntimeError("模型或说话人模板尚未初始化，请先调用 _init_model_and_speakers。")

    if not os.path.isfile(test_file):
        print(f"[警告] {test_file} 不是有效文件，跳过。")
        return None, 0.0

    best_name = None
    best_score = -1e9

    for spk_name, enrol_file in _enrolled_speakers.items():
        score, prediction = _verification.verify_files(enrol_file, test_file)
        score_val = float(score)
        if score_val > best_score:
            best_score = score_val
            best_name = spk_name

    if best_score < threshold:
        return None, best_score
    return best_name, best_score


# ================== 对外暴露的函数 ==================

def recognize_wait_dir(
    speakers_dir: str = DEFAULT_SPEAKERS_DIR,
    wait_dir: str = DEFAULT_WAIT_DIR,
    threshold: float = DEFAULT_THRESHOLD,
):
    """
    可在主程序中直接调用的函数。
    功能：识别 wait_dir 目录下所有音频文件，并用 print 打印出识别结果。

    参数:
        speakers_dir: 已注册说话人目录（文件名=说话人名字）
        wait_dir:     待识别音频所在目录
        threshold:    相似度阈值
    """
    # 初始化模型和说话人模板（只会在第一次调用时真正加载）
    _init_model_and_speakers(speakers_dir)

    if not _enrolled_speakers:
        print("[错误] 没有任何注册说话人，识别终止。")
        return

    if not os.path.isdir(wait_dir):
        print(f"[错误] 未找到待识别音频目录: {wait_dir}")
        return

    print(f"[信息] 开始识别目录 {wait_dir} 下的音频文件...\n")

    has_file = False
    for fname in os.listdir(wait_dir):
        if not fname.lower().endswith(AUDIO_EXTS):
            continue

        has_file = True
        test_path = os.path.join(wait_dir, fname)
        base_name = os.path.basename(test_path)

        spk_name, score = _recognize_one_file(
            test_file=test_path,
            threshold=threshold,
        )

        # 这里就是打印“被识别到的音频名 + 说话人名字”
        if spk_name is None:
            print(f"{base_name} -> 未知说话人 (最高相似度 {score:.3f})")
        else:
            print(f"{base_name} -> 识别为: {spk_name} (相似度 {score:.3f})")

    if not has_file:
        print(f"[警告] 目录 {wait_dir} 下没有找到任何音频文件。")



if __name__ == "__main__":
    recognize_wait_dir()
