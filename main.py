# main.py

from ui import run_ui

if __name__ == "__main__":
    # 启动图形界面
    run_ui()
'''
from recognize import recognize_wait_dir
from enhannce import enhance_folder
from emotion import recognize_emotions_in_folder  # 注意函数名

if __name__ == "__main__":

    recognize_wait_dir()
    enhance_folder()

    results = recognize_emotions_in_folder()  # 默认扫描 ./emotion

    print("\n汇总结果：")
    for fname, (label, score) in results.items():

        print(f"{fname} -> {label} ({score:.4f})")
'''

