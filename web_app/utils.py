"""
Utility functions for web application
"""

import json
import logging
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Model Loading Utilities
# ============================================================

def load_label_map(label_map_path):
    """
    Load label mapping from JSON file
    
    Args:
        label_map_path: Path to label_map.json
        
    Returns:
        dict: Mapping of class index to label name
    """
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        
        logger.info(f"Loaded label map with {len(label_map)} classes")
        return label_map
    
    except FileNotFoundError:
        logger.error(f"Label map not found: {label_map_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in label map: {label_map_path}")
        raise


# ============================================================
# Image Processing Utilities
# ============================================================

def decode_base64_image(base64_string):
    """
    Decode base64 image string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL.Image: Decoded image
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,...")
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        
        return image
    
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise


def encode_base64_image(image):
    """
    Encode PIL Image to base64 string
    
    Args:
        image: PIL.Image object
        
    Returns:
        str: Base64 encoded image string
    """
    try:
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        base64_string = base64.b64encode(image_data).decode('utf-8')
        
        return base64_string
    
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise


def resize_image(image, width, height):
    """
    Resize image to specified dimensions
    
    Args:
        image: PIL.Image object
        width: Target width
        height: Target height
        
    Returns:
        PIL.Image: Resized image
    """
    return image.resize((width, height), Image.Resampling.LANCZOS)


def convert_image_format(image, format_str='RGB'):
    """
    Convert image to specified format
    
    Args:
        image: PIL.Image object
        format_str: Target format ('RGB', 'RGBA', 'BGR', etc.)
        
    Returns:
        PIL.Image: Converted image
    """
    if format_str == 'BGR':
        # Convert to RGB then swap channels
        image = image.convert('RGB')
        image = Image.new('RGB', image.size)
        data = list(image.getdata())
        image.putdata([(r, g, b) for (b, g, r) in data])
    else:
        image = image.convert(format_str)
    
    return image


# ============================================================
# Array Utilities
# ============================================================

def pad_or_truncate_sequence(sequence, target_length):
    """
    Pad or truncate sequence to target length
    
    Args:
        sequence: Numpy array of shape (N, features)
        target_length: Target sequence length
        
    Returns:
        numpy.ndarray: Sequence of shape (target_length, features)
    """
    current_length = len(sequence)
    
    if current_length == target_length:
        return sequence
    
    elif current_length < target_length:
        # Pad with last frame repeated
        padding = target_length - current_length
        last_frame = sequence[-1]
        padding_frames = np.repeat(last_frame[np.newaxis, :], padding, axis=0)
        padded_sequence = np.vstack([sequence, padding_frames])
        return padded_sequence
    
    else:
        # Truncate - take evenly spaced frames
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        truncated_sequence = sequence[indices]
        return truncated_sequence


def normalize_keypoints(keypoints, image_width, image_height):
    """
    Normalize keypoints to [-1, 1] range
    
    Args:
        keypoints: Numpy array of shape (num_frames, 225)
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        numpy.ndarray: Normalized keypoints in [-1, 1] range
    """
    # Reshape to (num_frames, 75, 3)
    num_frames = len(keypoints)
    keypoints_reshaped = keypoints.reshape(num_frames, 75, 3)
    
    # Normalize x coordinates
    keypoints_reshaped[:, :, 0] = (2 * keypoints_reshaped[:, :, 0] / image_width) - 1
    
    # Normalize y coordinates
    keypoints_reshaped[:, :, 1] = (2 * keypoints_reshaped[:, :, 1] / image_height) - 1
    
    # Z coordinates already in normalized range [0, 1]
    
    # Reshape back to (num_frames, 225)
    keypoints_normalized = keypoints_reshaped.reshape(num_frames, 225)
    
    return keypoints_normalized


def extract_confidence_from_probs(probs, confidence_threshold=0.0):
    """
    Extract maximum confidence from prediction probabilities
    
    Args:
        probs: Dict or array of class probabilities
        confidence_threshold: Minimum threshold for valid prediction
        
    Returns:
        tuple: (max_prob, is_valid)
    """
    if isinstance(probs, dict):
        max_prob = max(probs.values())
    else:
        max_prob = float(np.max(probs))
    
    is_valid = max_prob >= confidence_threshold
    
    return max_prob, is_valid


# ============================================================
# File Utilities
# ============================================================

def ensure_directory_exists(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
        
    Returns:
        Path: Directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directory ensured: {directory}")
    return directory


def get_latest_checkpoint(checkpoint_dir):
    """
    Get latest checkpoint file from directory
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Path: Path to latest checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    # Find .pth files
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Get latest by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    logger.info(f"Found latest checkpoint: {latest}")
    
    return latest


def get_config_from_checkpoint(checkpoint_path):
    """
    Extract configuration metadata from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        dict: Configuration metadata
    """
    import torch
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'accuracy': checkpoint.get('accuracy', 'unknown'),
            'loss': checkpoint.get('loss', 'unknown'),
        }
        
        logger.info(f"Extracted config from checkpoint: {config}")
        return config
    
    except Exception as e:
        logger.error(f"Error extracting config from checkpoint: {str(e)}")
        return {}


# ============================================================
# Logging Utilities
# ============================================================

def setup_logging(log_file=None, log_level='INFO'):
    """
    Setup logging configuration
    
    Args:
        log_file: Optional log file path
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        ensure_directory_exists(Path(log_file).parent)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    logger.info(f"Logging configured: level={log_level}, file={log_file}")


# ============================================================
# Validation Utilities
# ============================================================

def validate_model_path(model_path):
    """
    Validate that model file exists and is readable
    
    Args:
        model_path: Path to model file
        
    Returns:
        bool: True if valid
        
    Raises:
        FileNotFoundError: If model not found
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.is_file():
        raise FileNotFoundError(f"Model path is not a file: {model_path}")
    
    logger.info(f"Model path validated: {model_path}")
    return True


def validate_label_map_path(label_map_path):
    """
    Validate that label map file exists and is readable
    
    Args:
        label_map_path: Path to label_map.json file
        
    Returns:
        bool: True if valid
        
    Raises:
        FileNotFoundError: If label map not found
    """
    label_map_path = Path(label_map_path)
    
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_path}")
    
    if not label_map_path.is_file():
        raise FileNotFoundError(f"Label map path is not a file: {label_map_path}")
    
    logger.info(f"Label map path validated: {label_map_path}")
    return True


def validate_configuration(num_frames, confidence_threshold):
    """
    Validate configuration parameters
    
    Args:
        num_frames: Number of frames
        confidence_threshold: Confidence threshold value
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError(f"Invalid num_frames: {num_frames}. Must be positive integer.")
    
    if not (0 <= confidence_threshold <= 1):
        raise ValueError(f"Invalid confidence_threshold: {confidence_threshold}. Must be in [0, 1].")
    
    logger.info(f"Configuration validated: num_frames={num_frames}, threshold={confidence_threshold}")
    return True


# ============================================================
# Statistics Utilities
# ============================================================

class PredictionStats:
    """Statistics tracker for predictions"""
    
    def __init__(self):
        self.total_predictions = 0
        self.predictions_history = []
        self.confidence_scores = []
        self.inference_times = []
    
    def add_prediction(self, label, confidence, inference_time):
        """Add prediction to statistics"""
        self.total_predictions += 1
        self.predictions_history.append(label)
        self.confidence_scores.append(confidence)
        self.inference_times.append(inference_time)
    
    def get_stats(self):
        """Get aggregated statistics"""
        if not self.confidence_scores:
            return {}
        
        return {
            'total_predictions': self.total_predictions,
            'avg_confidence': float(np.mean(self.confidence_scores)),
            'max_confidence': float(np.max(self.confidence_scores)),
            'min_confidence': float(np.min(self.confidence_scores)),
            'avg_inference_time': float(np.mean(self.inference_times)),
            'last_label': self.predictions_history[-1] if self.predictions_history else None,
            'last_confidence': self.confidence_scores[-1] if self.confidence_scores else None,
        }
    
    def reset(self):
        """Reset statistics"""
        self.total_predictions = 0
        self.predictions_history = []
        self.confidence_scores = []
        self.inference_times = []


__all__ = [
    'load_label_map',
    'decode_base64_image',
    'encode_base64_image',
    'resize_image',
    'convert_image_format',
    'pad_or_truncate_sequence',
    'normalize_keypoints',
    'extract_confidence_from_probs',
    'ensure_directory_exists',
    'get_latest_checkpoint',
    'get_config_from_checkpoint',
    'setup_logging',
    'validate_model_path',
    'validate_label_map_path',
    'validate_configuration',
    'PredictionStats',
]
