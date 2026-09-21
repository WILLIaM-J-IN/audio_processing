# Image Emotion Recognition and Enhancement

This project is a visual processing tool that integrates image recognition, quality enhancement, and emotion analysis. It provides an intuitive Graphical User Interface (GUI) alongside an automated script mode for batch processing folders.

## Features

*   **GUI Mode:** Provides an interactive interface for user-friendly operation.
*   **Image Recognition Pipeline:** Automatically processes image data within specified waiting directories.
*   **Automated Image Enhancement:** Features batch enhancement capabilities to improve image quality across entire folders.
*   **Emotion Analysis:** Automatically scans the target emotion folder (default is `./emotion`), identifies emotions, and outputs the predicted label alongside its confidence score.

## Getting Started

### 1. Launching the GUI (Default Mode)

By default, the system launches with the graphical interface. Run the `main.py` file in your terminal:

```bash
python main.py
```

Upon execution, the program calls the `run_ui()` method from the `ui` module to start the interface.

### 2. Batch Processing Mode (Script Mode)

If you prefer to process images and recognize emotions automatically in the background, you can switch to the script mode by editing `main.py`.

Simply comment out the `run_ui()` function under `if __name__ == "__main__":` and uncomment the multi-line block below it. Once enabled, the program will execute the following pipeline:

1.  **Recognition:** Calls `recognize_wait_dir()` to process files in the waiting queue.
2.  **Enhancement:** Calls `enhance_folder()` to improve image quality in the target folder.
3.  **Emotion Summary:** Calls `recognize_emotions_in_folder()` to analyze the `./emotion` directory and prints the predicted emotion label and score for each file to the console.

## Core Dependencies

Ensure the following Python modules are present in your project directory:

*   `ui.py`: Contains the main interface function `run_ui`.
*   `recognize.py`: Contains the `recognize_wait_dir` function.
*   `enhannce.py`: Contains the `enhance_folder` function (note the spelling of the filename).
*   `emotion.py`: Contains the core emotion analysis function `recognize_emotions_in_folder`.
