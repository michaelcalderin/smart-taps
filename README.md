# SmartTaps: Voice-Controlled Interaction for Smartphones

SmartTaps is a new software program that enables users to control their mobile devices entirely through voice commands. It converts spoken instructions into precise tap actions on the screen by combining a speech-to-text model, multimodal vision-language understanding, and automated UI execution.

This project provides a hands-free, accessible alternative to traditional touch-based controls. It is "screen-aware," meaning it understands the visual content of the interface, allowing for natural language instructions like "Click the blue button," without relying on rigid, pre-defined commands or app-specific APIs.

## Installation and Setup Instructions

This project is a prototype designed to run on a computer, using an emulator (Android Studio with Android Debug Bridge) to simulate a mobile device.

### Prerequisites

You will need the following software installed on your system:

1. Python 3.11+
2. Android Studio Emulator: This is required to simulate a working mobile device where the tap commands will be executed.
3. Android Debug Bridge: command-line tool to execute actions on emulated device (typically comes with Android Studio)

### 1. Clone the Repository
```
git clone https://github.com/<YourUsername>/smart-taps.git
cd smart-taps
```

### 2. Install Required Python Libraries

All necessary libraries, including PyTorch and Hugging Face models (Whisper, CLIP), are required for the system architecture.
```
pip install -r requirements.txt
```

### 3. Configure Android Studio Emulator

* Launch Android Studio.
* Create a new Virtual Device (AVD). A standard phone profile is recommended.
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

1. The Python script will run in the background, monitoring your microphone for input.
2. Speak a command (e.g., "Open Spotify," "Click the search icon," or "Tap the settings button").
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

Use ```notebooks/training.ipynb``` to fine-tune CLIP for bounding box prediction. Default hyperparameters:
- Learning rate (regression head): 1e-4
- Learning rate (unfrozen CLIP layers): 1e-5
- Batch size: 32
- Epochs: 5
- Loss function: Complete IoU (CIoU)
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

## Current Results
| Metric | Training | Validation | Testing |
| ------ | -------- | ---------- | ------- |
| Loss   | 1.16     | 1.21       | 1.21    |
| IoU    | 0.052    | 0.057      | 0.057   |

## Known Issues
- Low IoU scores due to limited training data and small input size (224x224).
- Image compression significantly reduces UI clarity.
- Overfitting to recurring icons.
- Command ambiguity when multiple elements match a description.
- Manual recording via keyboard input (not yet continuous listening).
- Android-only support; iOS data unavailable.

## Author Name and Contact

| Role | Details |
|------|---------|
| Author | Michael Calderin |
| Institution | Department of Engineering Education, University of Florida |
| Contact | michaelcalderin@ufl.edu |
| Affiliation | Gainesville, FL, USA |
