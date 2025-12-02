"""
Web Application Configuration
"""

from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class ModelConfig:
    """Model paths and parameters for Vietnamese Sign Language recognition"""
    
    # Model paths (relative to project root)
    MODELS_DIR = PROJECT_ROOT / 'models'
    CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
    
    # Use vsl_v1 model (97.77% val_acc) - best performing model for Vietnamese Sign Language
    MODEL_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'best.pth'
    LABEL_MAP_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'label_map.json'


class WebappConfig:
    """Real-time webapp specific configuration (optimized for speed)"""
    
    # Reduce sequence length for faster inference
    SEQ_LEN = 18  # Reduced from 25 for real-time performance
    
    # Use simpler sampling mode for speed
    SAMPLING_MODE = "1"  # Mode 1 is faster than mode 2
    
    # Confidence threshold
    MIN_CONFIDENCE = 0.50  # Increased from 0.35 for better accuracy

