\# SmartTaps: Voice-Controlled Interaction for Smartphones



SmartTaps is a new software program that enables users to control their mobile devices entirely through voice commands. It converts spoken instructions into precise tap actions on the screen by combining a speech-to-text model, multimodal vision-language understanding, and automated UI execution.



This project provides a hands-free, accessible alternative to traditional touch-based controls. It is "screen-aware," meaning it understands the visual content of the interface, allowing for natural language instructions like "Click the blue button," without relying on rigid, pre-defined commands or app-specific APIs.



\## Installation and Setup Instructions



This project is a prototype designed to run on a computer, using an emulator to simulate a mobile device.



\### Prerequisites



You will need the following software installed on your system:



1\. Python 3.11+

2\. Android Studio Emulator: This is required to simulate a working mobile device where the tap commands will be executed.

3\. Appium: The Python library used to interact with and execute taps on the emulator.



\### 1. Clone the Repository

```bash

git clone https://github.com/<YourUsername>/smart-taps.git

cd smart-taps

```



\### 2. Install Required Python Libraries



All necessary libraries, including PyTorch, Hugging Face models (Whisper, CLIP), and Appium, are required for the system architecture.

```bash

pip install -r requirements.txt

```



\### 3. Configure Android Studio Emulator



\* Launch Android Studio.

\* Create a new Virtual Device (AVD). A standard phone profile (e.g., Pixel 4) is recommended.

\* Ensure the emulator is running before starting the SmartTaps script.



\## Running Instructions



The core execution logic is contained within a Python script that orchestrates the entire process, from voice input to tap execution.



\### 1. Prepare the Environment



Make sure your Android Studio Emulator is actively running and visible on your screen.



\### 2. Run the Main Script



Execute the primary Python script (e.g., `main.py`):

```bash

python main.py

```



\### 3. Interact with the System



1\. The Python script will run in the background, monitoring your microphone for input.

2\. Speak a command (e.g., "Open Spotify," "Click the search icon," or "Tap the settings button").

3\. The system will process the command and automatically execute the corresponding tap on the emulated mobile device, allowing you to visualize the result.



\## Dataset Information



The SmartTaps system relies on two primary datasets for its multimodal pipeline:



\### 1. Audio-to-Text Evaluation Dataset



| Feature | Details |

|---------|---------|

| Source | \[LJSpeech-1.1](https://www.kaggle.com/datasets/mathurinache/the-lj-speech-dataset) |

| Type | Audio and Text (transcriptions) |

| Size | 13.1k audio samples, ~3.56 GB |

| Purpose | Used as a benchmark to assess the transcription quality of the pre-trained Whisper audio-to-text model. |



\### 2. Screen Annotation Dataset (for Vision-Language Training)



| Feature | Details |

|---------|---------|

| Source | \[Google Research Screen Annotation Dataset (derived from the RICO dataset)](https://github.com/google-research-datasets/screen\_annotation) |

| Type | Images with UI element annotations (bounding box, labels) |

| Size | 22,417 annotated Android phone screens |

| Purpose | Paired with synthetic text commands (generated using an LLM) to create training data <br>(Text Command, Screenshot) → Coordinates to Tap. |



\## Author Name and Contact



| Role | Details |

|------|---------|

| Author | Michael Calderin |

| Institution | Department of Engineering Education, University of Florida |

| Contact | michaelcalderin@ufl.edu |

| Affiliation | Gainesville, FL, USA |

