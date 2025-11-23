
# SmartTaps: Voice-Controlled Interaction for Smartphones

SmartTaps enables users to control mobile devices through natural voice commands, converting speech into precise tap actions on the screen. The system combines Whisper (speech-to-text), a vision-language model (now OWL-ViT), and automated UI execution on an emulated Android device. It is designed for hands-free, accessible interaction and understands the visual content of the interface, allowing commands like "Tap the blue clock" without app-specific APIs.

The current prototype uses OWL-ViT for improved bounding box prediction and UI element localization, replacing the previous CLIP-based approach. The system is experimental and best used on Android emulators.


## Installation and Setup Instructions

This project is a prototype designed to run on a computer, using an emulator (Android Studio with Android Debug Bridge) to simulate a mobile device.

### Prerequisites

You will need the following software installed on your system:

1. Python 3.11+
2. Android Studio Emulator: This is required to simulate a working mobile device where the tap commands will be executed.
3. Android Debug Bridge: command-line tool to execute actions on emulated device (typically comes with Android Studio)

### 1. Clone the Repository
```
git clone https://github.com/michaelcalderin/smart-taps.git
cd smart-taps
```

### 2. Install Required Python Libraries

All necessary libraries, including PyTorch and Hugging Face models (Whisper, CLIP/OWL-ViT), are required for the system architecture.
```
pip install -r requirements.txt
```

### 3. Configure Android Studio Emulator

* Launch Android Studio.
* Create a new Virtual Device (AVD). A 1080x1920 phone profile is recommended.
* Ensure the emulator is running before starting the SmartTaps script.


## Running Instructions

The core execution logic is contained within a Python script that orchestrates the entire process, from voice input to tap execution.

### 1. Prepare the Environment

Make sure your Android Studio Emulator is actively running and visible on your screen.

### 2. Run the Main Script

Execute the primary Python script:
```
python ui/app.py
```

### 3. Interact with the System

1. The Python script will use your device's microphone as input.
2. Speak a command (e.g., "Tap the blue clock," "Open Spotify," or "Click the search icon").
3. The system will process the command and automatically execute the corresponding tap on the emulated mobile device, allowing you to visualize the result.

## Dataset Information

The SmartTaps system relies on two primary datasets for its multimodal pipeline:

### 1. Audio-to-Text Evaluation Dataset

| Feature | Details |
|---------|---------|
| Source | [LJSpeech-1.1](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset) |
| Type | Audio and Text (transcriptions) |
| Size | 13.1k audio samples, ~3.56 GB |
| Purpose | Used as a benchmark to assess the transcription quality of the pre-trained Whisper audio-to-text model. |

### 2. Screen Annotation Dataset (for Vision-Language Training)

| Feature | Details |
|---------|---------|
| Source | [Google Research Screen Annotation Dataset (derived from the RICO dataset)](https://github.com/google-research-datasets/screen_annotation) |
| Type | Images with UI element annotations (bounding box, labels) |
| Size | 22,417 annotated Android phone screens |
| Purpose | Paired with synthetic text commands (generated using an LLM) to create training data <br>(Text Command, Screenshot) → Coordinates to Tap. |

## Replicating the Project (Optional)

### 1. Prepare the Dataset

Run ```notebooks/dataset_creation.ipynb``` to generate a clean dataset from Google’s screen annotations. This notebook:
- Parses raw string annotations from Google’s screen dataset
- Extracts UI elements and bounding boxes
- Saves a structured CSV for training/validation/testing in ```data/screen_annotation/extracted_annotations.csv```

### 2. Train the Vision-Language Model

Use ```notebooks/v1_training.ipynb``` to fine-tune CLIP for bounding box prediction. Default hyperparameters:
- Learning rate (regression head): 1e-4
- Learning rate (unfrozen CLIP layers): 1e-5
- Batch size: 32
- Epochs: 5
- Loss function: Complete IoU (CIoU)
- Optimizer: AdamW

OR

Use ```notebooks/training.ipynb``` to fine-tune OWL-ViT for bounding box prediction. Default hyperparameters:
- Learning rate (detection head): 3e-4
- Learning rate (unfrozen vision encoder layers): 8e-6
- Learning rate (unfrozen text encoder layers): 2e-6
- Batch size: 8
- Epochs: 1
- Loss functions: Focal (classification) and Smooth L1 (regression)
    - Refer to [OWL-ViT Paper](https://arxiv.org/abs/2205.06230) for more information about classification and regression in detection head
- Optimizer: AdamW

After training, a model checkpoint will be saved automatically as ```src/best_bounding_box_model.pth```.

### 3. Launch the SmartTaps Interface

Start your Android emulator through Android Studio.
Then, in your terminal, run:
```
python src/app.py
```
This script:
- Allows the recording of a 3-second audio clip.
- Converts speech into text using Whisper.
- Captures a screenshot from the emulator using ADB.
- Predicts the bounding box of the target UI element using the trained vision-language model.
- Executes a tap at the center of the predicted bounding box.

Predicted taps and bounding boxes are logged in the terminal and visualized for debugging in ```ui/screenshots```.


## Updated Performance Results

| Model    | Epochs | Samples | Train IoU | Val IoU | Test IoU | Test Loss |
|----------|--------|---------|-----------|---------|----------|-----------|
| CLIP     | 5      | 10,000  | 0.052     | 0.057   | 0.057    | 1.21      |
| OWL-ViT  | 1      | 10,000  | N/A       | 0.1935  | 0.1910   | 0.0020    |

- CLIP produced poor bounding box predictions and low IoU scores.
- OWL-ViT, even with limited training, shows substantial improvement in localizing UI elements.
- Note: loss used is different for CLIP vs. OWL-ViT (comparison is limited)


## Known Issues and Warnings

- Detection accuracy is limited by training time and dataset diversity.
- Some UI elements may not be recognized, especially outside the training distribution.
- Only Android emulators are supported; iOS is not available.
- The system is experimental so predictions may be unreliable for unfamiliar screens.
- No continuous listening; manual recording is required.
- Image compression can reduce UI clarity (less of a problem for OWL-ViT).
- Overfitting to recurring icons has been observed.
- Command ambiguity when multiple elements match a description.


## Author Name and Contact

| Role    | Details                                 |
|---------|-----------------------------------------|
| Author  | Michael Calderin                        |
| Email   | michaelcalderin@ufl.edu                 |
| Institution | Department of Engineering Education, University of Florida |
| Location | Gainesville, FL, USA                   |
