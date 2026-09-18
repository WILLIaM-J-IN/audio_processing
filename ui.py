# ui.py
# 图形界面：三个按钮，对应 说话人识别 / 增强并播放 / 情感识别

import os

###
from real_time_recorder import start_realtime_emotion, stop_realtime_emotion, get_realtime_result, cleanup_recorder, get_last_saved_filename, get_recording_status, get_recorder
from real_time_recorder import set_ui_update_callback
###
import platform
import subprocess
import io
import contextlib
import tkinter as tk
from tkinter import messagebox, scrolledtext

from recognize import recognize_wait_dir
from enhannce import enhance_folder  # 用于先做增强
from emotion import recognize_emotions_in_folder


def run_ui():
    # 主窗口
    root = tk.Tk()
    root.title("数字音效处理系统_小组8")
    root.geometry("700x450")

    # 整体背景颜色（淡蓝，不是纯白）
    root.configure(bg="#e6f0ff")

    # 顶部标题栏
    header = tk.Frame(root, bg="#4f81bd", height=60)
    header.pack(fill="x", side="top")

    title_label = tk.Label(
        header,
        text="语音处理一体工具",
        bg="#4f81bd",
        fg="white",
        font=("Microsoft YaHei", 16, "bold"),
        pady=10,
    )
    title_label.pack(side="left", padx=20)

    subtitle_label = tk.Label(
        header,
        text="说话人识别 · 音频增强 · 情感识别",
        bg="#4f81bd",
        fg="white",
        font=("Microsoft YaHei", 10),
    )
    subtitle_label.pack(side="right", padx=20)

    # 中间主区域
    main_frame = tk.Frame(root, bg="#e6f0ff")
    main_frame.pack(fill="both", expand=True, padx=15, pady=15)

    # 左侧按钮区域（卡片式白底）
    button_card = tk.Frame(main_frame, bg="#ffffff", bd=0, relief="ridge")
    button_card.pack(side="left", fill="y", ipadx=10, ipady=10)

    card_title = tk.Label(
        button_card,
        text="功能面板",
        bg="#ffffff",
        fg="#333333",
        font=("Microsoft YaHei", 12, "bold"),
    )
    card_title.pack(pady=(5, 10))

    # 按钮样式参数
    btn_kwargs = {
        "width": 18,
        "height": 2,
        "font": ("Microsoft YaHei", 10),
        "bg": "#4f81bd",
        "fg": "white",
        "activebackground": "#385d8a",
        "activeforeground": "white",
        "relief": "flat",
        "cursor": "hand2",
    }

    # 右侧日志输出区域（卡片式）
    log_card = tk.Frame(main_frame, bg="#ffffff", bd=0, relief="ridge")
    log_card.pack(side="right", fill="both", expand=True, padx=(15, 0))

    log_title = tk.Label(
        log_card,
        text="运行日志",
        bg="#ffffff",
        fg="#333333",
        font=("Microsoft YaHei", 12, "bold"),
    )
    log_title.pack(pady=(5, 5))

    log_text = scrolledtext.ScrolledText(
        log_card,
        wrap="word",
        font=("Consolas", 15),
        bg="#f7f9ff",
        fg="#333333",
    )
    log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def append_log(msg: str):
        log_text.insert("end", msg + "\n")
        log_text.see("end")  # 自动滚到底部
        log_text.update_idletasks()

    def open_file_with_default_app(path: str):
        """用系统默认程序打开文件（相当于双击）。"""
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(path)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Linux / 其他
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            append_log(f"【错误】打开文件失败：{path}，原因：{e}")

    # === 功能按钮对应的回调 ===

    def on_recognize_speaker():
        """
        说话人识别：
        - 调用 recognize_wait_dir()
        - 把原本打印到控制台的内容，捕获后写到右侧“运行日志”里
        """
        append_log("【说话人识别】开始处理 speakers_wait 目录中的音频...")

        buf = io.StringIO()
        try:
            # 把 recognize_wait_dir 的 print 输出重定向到 buf
            with contextlib.redirect_stdout(buf):
                recognize_wait_dir()

            output = buf.getvalue().strip()
            if output:
                append_log("【说话人识别】详细结果：")
                for line in output.splitlines():
                    append_log("  " + line)
            else:
                append_log("【说话人识别】没有任何输出。")

            append_log("【说话人识别】处理完成。")
        except Exception as e:
            messagebox.showerror("错误", f"说话人识别失败：{e}")
            append_log(f"【错误】说话人识别失败：{e}")

    def on_enhance_audio():
        """
        完整功能：
        1. 先调用 enhance_folder() 对 enhance_source 中的音频进行增强，
           输出到根目录下 enhance 文件夹（文件名为 原文件名_enhanced.wav）。
        2. 再打开 enhance 目录下的所有文件进行播放。
        """
        append_log("【音频增强】开始增强 enhance_source 目录中的音频...")

        # 第一步：调用增强
        try:
            enhance_folder()  # 使用 enhannce.py 中的默认参数
            append_log("【音频增强】增强完成，准备打开增强后的文件进行播放。")
        except Exception as e:
            messagebox.showerror("错误", f"音频增强失败：{e}")
            append_log(f"【错误】音频增强失败：{e}")
            return

        # 第二步：打开 enhance 目录下所有文件进行播放
        base_dir = os.path.dirname(os.path.abspath(__file__))
        enhance_dir = os.path.join(base_dir, "enhance")

        if not os.path.isdir(enhance_dir):
            messagebox.showerror("错误", f"未找到增强结果目录：{enhance_dir}")
            append_log(f"【音频增强】目录不存在：{enhance_dir}")
            return

        files = [
            f for f in os.listdir(enhance_dir)
            if os.path.isfile(os.path.join(enhance_dir, f))
        ]

        if not files:
            messagebox.showinfo("提示", "enhance 目录下没有任何文件。")
            append_log("【音频增强】enhance 目录为空，没有可播放的文件。")
            return

        append_log(f"【音频增强】即将打开 enhance 目录下的 {len(files)} 个文件：")
        for fname in files:
            full_path = os.path.join(enhance_dir, fname)
            append_log(f"  打开：{fname}")
            open_file_with_default_app(full_path)

        append_log("【音频增强】所有文件已发送给系统默认程序打开。")

    def on_emotion_recognize():
        append_log("【情感识别】开始识别 emotion 目录中的音频情绪...")
        try:
            results = recognize_emotions_in_folder()
            if not results:
                append_log("【情感识别】未在 emotion 目录中找到 wav 文件。")
                return
            append_log("【情感识别】结果汇总：")
            for fname, (label, score) in results.items():
                append_log(f"  {fname} -> {label} ({score:.4f})")
            append_log("【情感识别】全部完成。")
        except Exception as e:
            messagebox.showerror("错误", f"情感识别失败：{e}")
            append_log(f"【错误】情感识别失败：{e}")

 #################
    def on_realtime_emotion():
        """开始实时情感分析"""
        append_log("【实时情感分析】开始录音...")
        append_log("【提示】点击'停止录音'按钮结束录音")
        # 先清除上次保存的文件记录
        get_recorder().last_saved_file = None
        if start_realtime_emotion():
            append_log("【实时情感分析】录音进行中，请说话...")
            # 更新按钮状态
            btn4.config(state="disabled", bg="#cccccc")  # 禁用开始按钮
            btn5.config(state="normal", bg="#4f81bd")  # 启用停止按钮
        else:
            append_log("【实时情感分析】录音启动失败")

    def on_stop_recording():
        """停止录音"""
        stop_realtime_emotion()
        append_log("【实时情感分析】录音已停止")

        # 恢复按钮状态
        btn4.config(state="normal", bg="#4f81bd")  # 启用开始按钮
        btn5.config(state="disabled", bg="#cccccc")  # 禁用停止按钮

        # 检查是否保存了文件
        root.after(1000, check_recording_saved)

    def check_recording_saved():
        """检查录音是否保存成功"""
        saved_file = get_last_saved_filename()
        if saved_file:
            append_log(f"【实时情感分析】录音已保存: {saved_file}")
            append_log("【提示】该文件已存入emotion文件夹，可用于后续情感识别")
        else:
            append_log("【实时情感分析】录音保存失败")
 #######################

    # === 三个按钮 ===现在是5个了
    btn1 = tk.Button(
        button_card,
        text="1. 说话人识别",
        command=on_recognize_speaker,
        **btn_kwargs,
    )
    btn1.pack(pady=5)

    btn2 = tk.Button(
        button_card,
        text="2. 增强并播放",
        command=on_enhance_audio,
        **btn_kwargs,
    )
    btn2.pack(pady=5)

    btn3 = tk.Button(
        button_card,
        text="3. 情感识别",
        command=on_emotion_recognize,
        **btn_kwargs,
    )
    btn3.pack(pady=5)
############################
    btn4 = tk.Button(
        button_card,
        text="4. 实时情感分析(录音)",
        command=on_realtime_emotion,
        **btn_kwargs,
    )
    btn4.pack(pady=5)

    # 添加停止按钮
    btn5_kwargs = btn_kwargs.copy()
    btn5_kwargs["bg"] = "#cccccc"  # 灰色表示禁用

    btn5 = tk.Button(
        button_card,
        text="5. 停止录音",
        command=on_stop_recording,
        **btn5_kwargs,
    )
    btn5.pack(pady=5)
    btn5.config(state="disabled")  # 初始状态为禁用

    # 添加实时结果显示区域
    realtime_frame = tk.Frame(button_card, bg="#ffffff")
    realtime_frame.pack(pady=10, fill="x")

    realtime_label = tk.Label(
        realtime_frame,
        text="实时结果:",
        bg="#ffffff",
        fg="#333333",
        font=("Microsoft YaHei", 10, "bold"),
    )
    realtime_label.pack(anchor="w")

    recording_status_var = tk.StringVar(value="录音状态: 未开始")
    emotion_var = tk.StringVar(value="等待录音...")
    confidence_var = tk.StringVar(value="置信度: 0.0000")
    recording_status_display = tk.Label(
        realtime_frame,
        textvariable=recording_status_var,
        bg="#ffffff",
        fg="#ff6b6b",  # 红色表示状态
        font=("Microsoft YaHei", 9, "bold"),
    )
    recording_status_display.pack(anchor="w", pady=(5, 0))

    emotion_display = tk.Label(
        realtime_frame,
        textvariable=emotion_var,
        bg="#ffffff",
        fg="#4f81bd",
        font=("Microsoft YaHei", 12, "bold"),
    )
    emotion_display.pack(anchor="w", pady=(5, 0))

    confidence_display = tk.Label(
        realtime_frame,
        textvariable=confidence_var,
        bg="#ffffff",
        fg="#666666",
        font=("Microsoft YaHei", 9),
    )
    confidence_display.pack(anchor="w")

    # === 定义UI更新函数 ===
    def update_ui_display(emotion, confidence):
        """直接从录音器回调更新UI显示"""
        emotion_var.set(emotion)
        confidence_var.set(f"置信度: {confidence:.4f}")
        # 强制立即更新UI
        root.update_idletasks()

    # 设置UI更新回调
    set_ui_update_callback(update_ui_display)

    # 实时更新函数
    def update_realtime_display():
        emotion, confidence = get_realtime_result()
        emotion_var.set(emotion)
        confidence_var.set(f"置信度: {confidence:.4f}")

        # 更新录音状态
        if get_recording_status():
            recording_status_var.set("录音状态: ● 录音中...")
            recording_status_display.config(fg="#ff6b6b")  # 红色
        else:
            recording_status_var.set("录音状态: 未开始")
            recording_status_display.config(fg="#666666")  # 灰色

        root.after(1000, update_realtime_display)  # 每秒更新一次

    root.after(1000, update_realtime_display)

    # 窗口关闭时清理资源
    def on_closing():
        cleanup_recorder()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
###########

    # 底部小提示
    footer = tk.Label(
        root,
        text="提示：请提前准备好相应目录下的音频文件（speakers / speakers_wait / enhance_source / enhance / emotion）。",
        bg="#e6f0ff",
        fg="#555555",
        font=("Microsoft YaHei", 9),
    )
    footer.pack(side="bottom", pady=(0, 8))

    root.mainloop()


if __name__ == "__main__":
    run_ui()
