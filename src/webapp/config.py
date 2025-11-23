"""
Web Application Configuration
"""

import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# ============================================================
# Flask Configuration
# ============================================================

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False
    
    # Socket.IO settings
    SOCKETIO_MESSAGE_QUEUE = None
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '[%(asctime)s] %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*')


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# ============================================================
# Model Configuration
# ============================================================

class ModelConfig:
    """Model paths and parameters"""
    
    # Model paths (relative to project root)
    MODELS_DIR = PROJECT_ROOT / 'models'
    CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
    
    # Use vsl_v1 model (97.77% val_acc) - best performing model for Vietnamese Sign Language
    MODEL_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'best.pth'
    LABEL_MAP_PATH = CHECKPOINTS_DIR / 'vsl_v1' / 'label_map.json'
    
    # Model parameters
    INPUT_SIZE = 225  # 75 landmarks * 3 (x, y, z)
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 3  # Number of classes (người, tôi, Việt Nam)
    
    # Inference parameters
    NUM_FRAMES = 25                    # Number of frames per sequence (VARIABLE - can be changed)
    CONFIDENCE_THRESHOLD = 0.30        # Minimum confidence for valid prediction
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480


# ============================================================
# MediaPipe Configuration
# ============================================================

class MediaPipeConfig:
    """MediaPipe Holistic settings"""
    
    STATIC_IMAGE_MODE = False
    MODEL_COMPLEXITY = 1  # 0: Light, 1: Full, 2: Heavy
    SMOOTH_LANDMARKS = True
    ENABLE_SEGMENTATION = False
    SMOOTH_SEGMENTATION = True
    REFINE_FACE_LANDMARKS = False
    
    # Pose detection thresholds
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# Server Configuration
# ============================================================

class ServerConfig:
    """Server settings"""
    
    HOST = '127.0.0.1'
    PORT = 5000
    
    # Frame capture settings
    FPS = 25
    FRAME_BUFFER_SIZE = 25
    
    # Communication settings
    SOCKET_TIMEOUT = 30
    FRAME_ENCODING = 'base64'
    
    # Performance
    MAX_WORKERS = 1  # Number of background workers
    ALLOW_UNSAFE_WERKZEUG = False


# ============================================================
# UI Configuration
# ============================================================

class UIConfig:
    """Frontend settings"""
    
    # Display
    PREDICTION_FONT_SIZE = '48px'
    CONFIDENCE_FONT_SIZE = '24px'
    HISTORY_MAX_ITEMS = 10
    
    # Animation settings
    PREDICTION_UPDATE_SPEED = 200  # milliseconds
    HISTORY_UPDATE_SPEED = 500
    
    # Statistics
    SHOW_FPS_COUNTER = True
    SHOW_LATENCY_STATS = True


# ============================================================
# Logging Configuration
# ============================================================

class LogConfig:
    """Logging settings"""
    
    LOG_DIR = PROJECT_ROOT / 'logs'
    APP_LOG_FILE = LOG_DIR / 'app.log'
    ERROR_LOG_FILE = LOG_DIR / 'error.log'
    
    LOG_FORMAT = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # File logging
    MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5


# ============================================================
# Export configuration objects
# ============================================================

def get_config(env=None):
    """Get configuration based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }
    
    return config_map.get(env, DevelopmentConfig)


# Default exports
__all__ = [
    'Config',
    'DevelopmentConfig',
    'ProductionConfig',
    'TestingConfig',
    'ModelConfig',
    'MediaPipeConfig',
    'ServerConfig',
    'UIConfig',
    'LogConfig',
    'get_config',
]
