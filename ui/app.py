import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import subprocess
from datetime import datetime
import whisper
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def convert_bbox_to_device_coords(bbox_norm, device_w, device_h):
    """
    Convert bounding box from normalized [0,1] space to device pixel coordinates.
    OWL-ViT processor resizes images to 768x768 with non-uniform scaling (squashing),
    so we directly scale normalized coordinates to device dimensions.
    
    Args:
        bbox_norm: (x1, y1, x2, y2) in normalized [0,1] space
        device_w, device_h: device width and height in pixels
    
    Returns:
        (x1, y1, x2, y2) in device pixel coordinates
    """
    x1 = bbox_norm[0] * device_w
    y1 = bbox_norm[1] * device_h
    x2 = bbox_norm[2] * device_w
    y2 = bbox_norm[3] * device_h
    
    # Clamp to device bounds
    x1 = max(0, min(device_w, x1))
    y1 = max(0, min(device_h, y1))
    x2 = max(0, min(device_w, x2))
    y2 = max(0, min(device_h, y2))
    
    return (x1, y1, x2, y2)


def record_audio(duration=3, sample_rate=16000):
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording (16kHz is optimal for Whisper)
    
    Returns:
        Audio data as numpy array
    """

    print(f"Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * sample_rate), 
                   samplerate=sample_rate, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    print("Recording complete!")
    return audio.flatten()


def transcribe_audio(audio_data, whisper_model):
    """
    Transcribe audio to text using Whisper.
    
    Args:
        audio_data: Audio data as numpy array
        whisper_model: Loaded Whisper model
    
    Returns:
        Transcribed text string
    """

    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_data, fp16=False)
    transcribed_text = result["text"].strip()
    print(f"Transcribed: '{transcribed_text}'")
    
    return transcribed_text


def load_whisper_model(model_size="base"):
    """
    Load Whisper model.
    
    Args:
        model_size: tiny, base, etc.
    
    Returns:
        Loaded Whisper model
    """

    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print("Whisper model loaded!")
    return model


class ADBDevice:
    """Helper class for ADB operations."""
    
    def __init__(self, device_id=None):
        """
        Initialize ADB device. Uses first device if device_id is not given.
        """

        self.device_id = device_id
        self.device_prefix = ["-s", device_id] if device_id else []
        
        # Check if adb is available
        try:
            result = subprocess.run(["adb", "version"], 
                                  capture_output=True, text=True, check=True)
            print(f"ADB version: {result.stdout.split()[4]}")
        except Exception:
            raise RuntimeError("ADB not found.")
        
        # Check for connected devices
        self.check_device()
    
    def check_device(self):
        """Check if device is connected."""

        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True)
        
        devices = [line.split()[0] for line in result.stdout.split('\n')[1:] 
                  if line.strip() and 'device' in line]
        
        if not devices:
            raise RuntimeError("No Android devices connected.")
        
        if self.device_id:
            if self.device_id not in devices:
                raise RuntimeError(f"Device {self.device_id} not found. Available: {devices}")
            print(f"Connected to device: {self.device_id}")
        else:
            self.device_id = devices[0]
            self.device_prefix = ["-s", self.device_id]
            print(f"Using device: {self.device_id}")
            if len(devices) > 1:
                print(f"Note: Multiple devices found. Using first one. Available: {devices}")
    
    def get_screen_size(self):
        """Get device screen resolution."""

        result = subprocess.run(
            ["adb"] + self.device_prefix + ["shell", "wm", "size"],
            capture_output=True, text=True, check=True
        )
        # Output format: "Physical size: 1080x1920"
        size_line = result.stdout.strip()
        resolution = size_line.split(": ")[1]
        width, height = map(int, resolution.split("x"))
        return width, height
    
    def capture_screenshot(self, save_path):
        """
        Capture screenshot from Android device.
        
        Args:
            save_path: Local path to save screenshot
        
        Returns:
            PIL Image object
        """

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Capture screenshot on device
        subprocess.run(
            ["adb"] + self.device_prefix + ["shell", "screencap", "-p", "/sdcard/screen.png"],
            check=True
        )
        
        # Pull screenshot to local
        subprocess.run(
            ["adb"] + self.device_prefix + ["pull", "/sdcard/screen.png", save_path],
            capture_output=True, check=True
        )
        
        # Clean up device
        subprocess.run(
            ["adb"] + self.device_prefix + ["shell", "rm", "/sdcard/screen.png"],
            capture_output=True, check=True
        )
        
        # Load and return image
        image = Image.open(save_path)
        return image
    
    def tap(self, x, y):
        """
        Perform tap at specified coordinates.
        
        Args:
            x, y: Coordinates to tap (in device pixels) ... (0, 0) is top-left of screen
        """

        subprocess.run(
            ["adb"] + self.device_prefix + ["shell", "input", "tap", str(int(x)), str(int(y))],
            check=True
        )

def load_model(model_path, device):
    """Load the trained OWL-ViT model."""
    
    model_name = "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    return model, processor


def predict_and_tap(text_query, model, processor, adb_device, screenshot_dir="../ui/screenshots"):
    """
    Main function: capture screenshot, predict bounding box, and tap.
    
    Args:
        text_query: Text description of what to click
        model: Trained OWL-ViT model
        processor: OWL-ViT processor
        adb_device: ADBDevice instance
        screenshot_dir: Directory to save screenshots
    """
    print(f"\n{'='*60}")
    print(f"Query: '{text_query}'")
    print(f"{'='*60}")
    
    # Capture screenshot from Android device
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
    
    print("Capturing screenshot from Android device...")
    screenshot = adb_device.capture_screenshot(screenshot_path)
    device_w, device_h = screenshot.size
    print(f"Screenshot saved: {screenshot_path}")
    print(f"Device resolution: {device_w}x{device_h}")
    
    # Run model prediction (OWL-ViT processor handles resizing to 768x768)
    print("\nRunning model prediction...")
    device = next(model.parameters()).device
    
    # Ensure screenshot is in RGB mode (processor requirement)
    if screenshot.mode != 'RGB':
        screenshot = screenshot.convert('RGB')
    
    # Process image and text (processor will resize to 768x768 with non-uniform scaling)
    inputs = processor(
        text=[text_query],
        images=[screenshot],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=16
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Outputs
    logits = outputs.logits[0, :, 0]
    pred_boxes = outputs.pred_boxes[0]
    
    # Get the patch with highest score for tapping
    best_patch_idx = logits.argmax()
    best_box = pred_boxes[best_patch_idx].cpu().numpy()  # [cx, cy, w, h] normalized [0,1]
    best_score = torch.sigmoid(logits[best_patch_idx]).item()
    
    # Get top 3 predictions for visualization
    top3_indices = torch.topk(logits, k=min(3, len(logits))).indices
    top3_boxes = pred_boxes[top3_indices].cpu().numpy()
    top3_scores = torch.sigmoid(logits[top3_indices]).cpu().numpy()
    
    print(f"Best prediction score: {best_score:.4f}")
    print(f"Predicted bbox (center + size, normalized): [cx={best_box[0]:.4f}, cy={best_box[1]:.4f}, w={best_box[2]:.4f}, h={best_box[3]:.4f}]")
    
    # Convert from center+size to corner format for visualization
    cx, cy, w, h = best_box
    x1_norm = cx - w / 2
    y1_norm = cy - h / 2
    x2_norm = cx + w / 2
    y2_norm = cy + h / 2
    
    bbox_norm = np.array([x1_norm, y1_norm, x2_norm, y2_norm])
    print(f"Predicted bbox (corners, normalized): [{x1_norm:.4f}, {y1_norm:.4f}, {x2_norm:.4f}, {y2_norm:.4f}]")
    
    # Convert to device coordinates (simple scaling since processor squashes to 768x768)
    x1, y1, x2, y2 = convert_bbox_to_device_coords(bbox_norm, device_w, device_h)
    print(f"Converted bbox (device coords): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Calculate center point and tap
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    print(f"\nTap location: ({center_x:.1f}, {center_y:.1f})")
    
    # Perform the tap via ADB
    print("Tapping on Android device...")
    adb_device.tap(center_x, center_y)
    print("Tap executed!")
    
    # Create visualization figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    
    # Show device screenshot with prediction
    ax.imshow(screenshot)
    
    # Draw all top 3 predictions
    colors = ['red', 'orange', 'yellow']
    linestyles = ['--', '-.', ':']
    linewidths = [4, 3, 2]
    
    for rank, (box, score) in enumerate(zip(top3_boxes, top3_scores)):
        # Convert from center+size to corner format
        cx, cy, w, h = box
        x1_norm = cx - w / 2
        y1_norm = cy - h / 2
        x2_norm = cx + w / 2
        y2_norm = cy + h / 2
        
        # Convert to device coordinates
        bbox_norm = np.array([x1_norm, y1_norm, x2_norm, y2_norm])
        x1_pred, y1_pred, x2_pred, y2_pred = convert_bbox_to_device_coords(bbox_norm, device_w, device_h)
        
        # Draw bounding box
        pred_w = x2_pred - x1_pred
        pred_h = y2_pred - y1_pred
        
        rect = patches.Rectangle(
            (x1_pred, y1_pred), pred_w, pred_h,
            linewidth=linewidths[rank], edgecolor=colors[rank], 
            facecolor='none', linestyle=linestyles[rank],
            label=f'#{rank+1}: {score:.3f}'
        )
        ax.add_patch(rect)
        
        # Draw center point for top-1 prediction (the one we're tapping)
        if rank == 0:
            center_x_pred = (x1_pred + x2_pred) / 2
            center_y_pred = (y1_pred + y2_pred) / 2
            ax.plot(center_x_pred, center_y_pred, 'ro', markersize=12, 
                   markeredgecolor='white', markeredgewidth=2)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add title with details
    ax.set_title(
        f'Query: "{text_query}" (Score: {best_score:.2f})\n'
        f'Device: {device_w}x{device_h} | Tap: ({center_x:.1f}, {center_y:.1f})\n'
        f'Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]',
        fontsize=12, fontweight='bold', pad=15
    )
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save comparison image
    comparison_path = os.path.join(screenshot_dir, f"comparison_{timestamp}.png")
    plt.savefig(comparison_path)
    plt.close()
    print(f"Comparison image saved: {comparison_path}")
    
    return (center_x, center_y), (x1, y1, x2, y2)


def main():

    # Intro comment
    print("="*60)
    print("ANDROID ADB VOICE-CONTROLLED TAP AUTOMATION")
    print("="*60)
    
    # Setup torch
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using torch device: {torch_device}")
    
    # Load Whisper model
    print("\nLoading Whisper model...")
    whisper_model = load_whisper_model(model_size="base")
    
    # Initialize ADB (auto-detect device)
    print("\nInitializing ADB connection...")
    try:
        adb_device = ADBDevice()
        device_w, device_h = adb_device.get_screen_size()
        print(f"Device screen size: {device_w}x{device_h}")
    except Exception as e:
        print(f"Error connecting to device: {e}")
        return
    
    # Load OWL-ViT model
    model_path = "../src/best_bounding_box_model.pth"
    print(f"\nLoading OWL-ViT model from: {model_path}")
    
    try:
        model, processor = load_model(model_path, torch_device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("\n" + "="*60)
    print("Ready! Press Enter to record voice command (3 seconds)")
    print("Type 'quit' to exit, or just press Enter to record")
    print("="*60)
    
    while True:
        user_input = input("\nPress Enter to record (or type 'quit'): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        try:
            # Record audio
            audio_data = record_audio(duration=3, sample_rate=16000)
            
            # Transcribe to text
            text_query = transcribe_audio(audio_data, whisper_model)
            
            if not text_query:
                print("No speech detected. Please try again.")
                continue
            
            # Execute the tap
            predict_and_tap(text_query, model, processor, adb_device)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()