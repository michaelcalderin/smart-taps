import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import subprocess
from datetime import datetime
import whisper
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Model architecture
class BoundingBoxModel(nn.Module):
    def __init__(self,
                 clip_model_name="openai/clip-vit-base-patch32",
                 freeze_clip=True,
                 unfreeze_last_n_vision_layers=0,
                 unfreeze_last_n_text_layers=0,
                 device="cuda",
                 max_box_width=1,
                 max_box_height=1):
        super().__init__()
        self.device = device
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        self.freeze_clip = freeze_clip
        self.unfreeze_last_n_vision_layers = unfreeze_last_n_vision_layers
        self.unfreeze_last_n_text_layers = unfreeze_last_n_text_layers

        self.max_box_width = max_box_width
        self.max_box_height = max_box_height

        embed_dim = self.clip.config.projection_dim

        # Freeze CLIP
        if self.freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            
            # Unfreeze last N vision layers if specified
            if self.unfreeze_last_n_vision_layers > 0:
                vision_layers = self.clip.vision_model.encoder.layers
                for layer in vision_layers[-self.unfreeze_last_n_vision_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Unfroze last {self.unfreeze_last_n_vision_layers} vision layers")
            
            # Unfreeze last N text layers if specified
            if self.unfreeze_last_n_text_layers > 0:
                text_layers = self.clip.text_model.encoder.layers
                for layer in text_layers[-self.unfreeze_last_n_text_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"Unfroze last {self.unfreeze_last_n_text_layers} text layers")

        # Fusion and prediction head
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Bounding box regression head
        self.bbox_head = nn.Linear(256, 4)

    def forward(self, images, texts, targets=None, visualize=False, batch_idx=0, loss_value=None):
        # Encode text
        text_inputs = self.processor(
            text=texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.set_grad_enabled(not self.freeze_clip or self.unfreeze_last_n_text_layers > 0):
            text_embeds = self.clip.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Encode images (already 224x224 with padding)
        image_inputs = self.processor(
            images=images, 
            return_tensors="pt"
        ).to(self.device)
        
        with torch.set_grad_enabled(not self.freeze_clip or self.unfreeze_last_n_vision_layers > 0):
            image_embeds = self.clip.get_image_features(**image_inputs)
        image_embeds = F.normalize(image_embeds, dim=-1)

        # Combine image and text embeddings
        combined = torch.cat([image_embeds, text_embeds], dim=-1)
        features = self.fusion(combined)

        # Predict bounding box in center + size format
        bbox_params = self.bbox_head(features)
        
        # Convert to (x1, y1, x2, y2) format with sigmoid for [0,1] range
        cx = torch.sigmoid(bbox_params[:, 0])
        cy = torch.sigmoid(bbox_params[:, 1])
        w = torch.sigmoid(bbox_params[:, 2]) * self.max_box_width # Model tends to predict big boxes and doesn't learn so this will restrict size
        h = torch.sigmoid(bbox_params[:, 3]) * self.max_box_height
        
        # Convert center + size to corners
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Clamp to [0, 1]
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)
        
        bbox = torch.stack([x1, y1, x2, y2], dim=1)
        return bbox


def pad_image_to_square(image, target_size=224, fill_color=(0, 0, 0)):
    """
    Pad image to square while maintaining aspect ratio.
    Returns padded image and padding info for coordinate conversion.
    """

    w, h = image.size
    max_dim = max(w, h)
    scale = target_size / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    image = image.resize((new_w, new_h), Image.BILINEAR)
    padded_image = Image.new('RGB', (target_size, target_size), fill_color)
    
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    padded_image.paste(image, (paste_x, paste_y))
    
    return padded_image, (paste_x, paste_y, new_w, new_h, scale)


def convert_bbox_to_device_coords(bbox_224, padding_info, device_w, device_h, target_size=224):
    """
    Convert bounding box from 224x224 padded space back to original device coordinates.
    Works for any device dimensions, not just the original 1080x1920 training data.
    
    Args:
        bbox_224: (x1, y1, x2, y2) in 224x224 normalized [0,1] space
        padding_info: (paste_x, paste_y, new_w, new_h, scale) from pad_image_to_square
        device_w, device_h: current device width and height (can be any dimensions)
        target_size: model input size (224)
    
    Returns:
        (x1, y1, x2, y2) in device pixel coordinates
    """

    paste_x, paste_y, new_w, new_h, scale = padding_info
    
    # Convert from normalized [0,1] to absolute pixels in 224x224 space
    x1_224 = bbox_224[0] * target_size
    y1_224 = bbox_224[1] * target_size
    x2_224 = bbox_224[2] * target_size
    y2_224 = bbox_224[3] * target_size
    
    # Remove padding offset to get coordinates in the resized (but unpadded) space
    x1_unpadded = x1_224 - paste_x
    y1_unpadded = y1_224 - paste_y
    x2_unpadded = x2_224 - paste_x
    y2_unpadded = y2_224 - paste_y
    
    # Clamp to valid range in unpadded space ... ensures coords stay within actual image area (not in padding)
    x1_unpadded = max(0, min(new_w, x1_unpadded))
    y1_unpadded = max(0, min(new_h, y1_unpadded))
    x2_unpadded = max(0, min(new_w, x2_unpadded))
    y2_unpadded = max(0, min(new_h, y2_unpadded))
    
    # Unscale to current device dimensions
    x1_device = x1_unpadded / scale
    y1_device = y1_unpadded / scale
    x2_device = x2_unpadded / scale
    y2_device = y2_unpadded / scale
    
    # Final clamp to ensure coords are within device bounds
    x1_device = max(0, min(device_w, x1_device))
    y1_device = max(0, min(device_h, y1_device))
    x2_device = max(0, min(device_w, x2_device))
    y2_device = max(0, min(device_h, y2_device))
    
    return (x1_device, y1_device, x2_device, y2_device)


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
    """Load the trained model."""

    model = BoundingBoxModel(
        device=device,
        freeze_clip=True,
        unfreeze_last_n_vision_layers=4,
        unfreeze_last_n_text_layers=4,
        max_box_width=0.25,
        max_box_height=0.25
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    
    return model


def predict_and_tap(text_query, model, adb_device, screenshot_dir="../ui/screenshots"):
    """
    Main function: capture screenshot, predict bounding box, and tap.
    
    Args:
        text_query: Text description of what to click
        model: Trained BoundingBoxModel
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
    
    # Prepare image for model
    print("\nProcessing image...")
    padded_image, padding_info = pad_image_to_square(screenshot, target_size=224)
    paste_x, paste_y, new_w, new_h, scale = padding_info
    print(f"Padding info: paste=({paste_x}, {paste_y}), size=({new_w}, {new_h}), scale={scale:.4f}")
    
    # Run model prediction
    print("\nRunning model prediction...")
    with torch.no_grad():
        pred_bbox = model([padded_image], [text_query])
    
    # Get prediction in 224x224 normalized space
    bbox_224_norm = pred_bbox[0].cpu().numpy()
    print(f"Predicted bbox (224x224 normalized): [{bbox_224_norm[0]:.4f}, {bbox_224_norm[1]:.4f}, {bbox_224_norm[2]:.4f}, {bbox_224_norm[3]:.4f}]")
    
    # Convert to device coordinates
    x1, y1, x2, y2 = convert_bbox_to_device_coords(
        bbox_224_norm, padding_info, device_w, device_h
    )
    print(f"Converted bbox (device coords): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Calculate center point and tap
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    print(f"\nTap location: ({center_x:.1f}, {center_y:.1f})")
    
    # Perform the tap via ADB
    print("Tapping on Android device...")
    adb_device.tap(center_x, center_y)
    print("Tap executed!")
    
    # Create figure with padded 224x224 view and normal view of predicted bounding box
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left side: 224x224 padded image with prediction
    ax1 = axes[0]
    padded_array = torch.tensor(np.array(padded_image)).permute(2, 0, 1).float() / 255.0
    
    # Apply CLIP normalization for visualization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    normalized_array = (padded_array - mean) / std
    
    # Denormalize back for display
    display_array = normalized_array * std + mean
    display_array = torch.clamp(display_array, 0, 1).permute(1, 2, 0).numpy()
    
    ax1.imshow(display_array)
    
    # Draw prediction box on 224x224 image (convert normalized to pixels)
    pred_x1_224 = bbox_224_norm[0] * 224
    pred_y1_224 = bbox_224_norm[1] * 224
    pred_x2_224 = bbox_224_norm[2] * 224
    pred_y2_224 = bbox_224_norm[3] * 224
    pred_w_224 = pred_x2_224 - pred_x1_224
    pred_h_224 = pred_y2_224 - pred_y1_224
    
    rect_224 = patches.Rectangle(
        (pred_x1_224, pred_y1_224), pred_w_224, pred_h_224,
        linewidth=3, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax1.add_patch(rect_224)
    
    # Draw center point
    center_x_224 = (pred_x1_224 + pred_x2_224) / 2
    center_y_224 = (pred_y1_224 + pred_y2_224) / 2
    ax1.plot(center_x_224, center_y_224, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
    
    ax1.set_title(f'224x224 Padded Image\nPredicted Box: [{pred_x1_224:.1f}, {pred_y1_224:.1f}, {pred_x2_224:.1f}, {pred_y2_224:.1f}]', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Right side: Device screenshot with prediction
    ax2 = axes[1]
    ax2.imshow(screenshot)
    
    # Draw prediction box on device scale
    pred_w_device = x2 - x1
    pred_h_device = y2 - y1
    
    rect_device = patches.Rectangle(
        (x1, y1), pred_w_device, pred_h_device,
        linewidth=4, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax2.add_patch(rect_device)
    
    # Draw center point
    ax2.plot(center_x, center_y, 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
    
    ax2.set_title(f'Device Screenshot ({device_w}x{device_h})\nTap Location: ({center_x:.1f}, {center_y:.1f})', 
                  fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Add overall title
    fig.suptitle(f'Query: "{text_query}"\nBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
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
    
    # Load CLIP model
    model_path = "../src/best_bounding_box_model.pth"
    print(f"\nLoading bounding box model from: {model_path}")
    
    try:
        model = load_model(model_path, torch_device)
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
            predict_and_tap(text_query, model, adb_device)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()