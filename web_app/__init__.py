"""
Vietnamese Sign Language Real-time Web Application
"""

__version__ = '1.0.0'
__author__ = 'Student Name'
__description__ = 'Real-time Vietnamese sign language recognition using deep learning'

from .config import (
    Config,
    DevelopmentConfig,
    ProductionConfig,
    TestingConfig,
    ModelConfig,
    MediaPipeConfig,
    ServerConfig,
    UIConfig,
    LogConfig,
    get_config,
)

from .utils import (
    load_label_map,
    decode_base64_image,
    encode_base64_image,
    setup_logging,
    validate_model_path,
    validate_label_map_path,
    PredictionStats,
)

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
    'load_label_map',
    'decode_base64_image',
    'encode_base64_image',
    'setup_logging',
    'validate_model_path',
    'validate_label_map_path',
    'PredictionStats',
]
