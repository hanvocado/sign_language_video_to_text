#!/usr/bin/env python3
"""
Quick Start Script for Vietnamese Sign Language Web Application

This script helps you set up and run the web application with proper configuration.
"""

import os
import sys
import json
import logging
from pathlib import Path
from colorama import Fore, Style, init

# Initialize colorama for colored output on Windows
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
WEB_APP_DIR = PROJECT_ROOT / 'web_app'
MODELS_DIR = PROJECT_ROOT / 'models'
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'


def print_header():
    """Print welcome header"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}🤟 Vietnamese Sign Language Recognition - Web App Setup")
    print(f"{Fore.CYAN}{'='*60}\n")


def print_step(step_num, title):
    """Print step header"""
    print(f"\n{Fore.YELLOW}[Step {step_num}] {title}")
    print(f"{Fore.YELLOW}{'-'*60}")


def check_python_version():
    """Check Python version requirement"""
    print_step(1, "Checking Python Version")
    
    version_info = sys.version_info
    print(f"  Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 8):
        print(f"  {Fore.RED}❌ Python 3.8+ required")
        return False
    
    print(f"  {Fore.GREEN}✓ Python version OK")
    return True


def check_model_files():
    """Check if model files exist"""
    print_step(2, "Checking Model Files")
    
    model_path = CHECKPOINTS_DIR / 'best.pth'
    label_map_path = CHECKPOINTS_DIR / 'label_map.json'
    
    # Check model file
    if not model_path.exists():
        print(f"  {Fore.YELLOW}⚠ Model file not found: {model_path}")
        print(f"  {Fore.YELLOW}  → You need to train the model first")
        print(f"  {Fore.YELLOW}  → Run: python -m src.model.train --data_dir data/splits")
        return False
    else:
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  {Fore.GREEN}✓ Model file: {model_path} ({file_size_mb:.1f}MB)")
    
    # Check label map file
    if not label_map_path.exists():
        print(f"  {Fore.YELLOW}⚠ Label map not found: {label_map_path}")
        return False
    else:
        try:
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            num_classes = len(label_map)
            print(f"  {Fore.GREEN}✓ Label map: {label_map_path} ({num_classes} classes)")
            print(f"    Classes: {', '.join(label_map.values())}")
        except Exception as e:
            print(f"  {Fore.RED}✗ Error reading label map: {str(e)}")
            return False
    
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print_step(3, "Checking Dependencies")
    
    required_packages = [
        'flask',
        'flask_socketio',
        'torch',
        'cv2',  # opencv-python
        'mediapipe',
        'numpy',
        'PIL',  # pillow
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {Fore.GREEN}✓ {package}")
        except ImportError:
            print(f"  {Fore.RED}✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n  {Fore.YELLOW}Install missing packages:")
        print(f"  {Fore.YELLOW}pip install -r requirements.txt")
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    print_step(4, "Creating Directories")
    
    directories = [
        WEB_APP_DIR,
        WEB_APP_DIR / 'static',
        WEB_APP_DIR / 'templates',
        PROJECT_ROOT / 'logs',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  {Fore.GREEN}✓ {directory}")


def validate_server_config():
    """Validate server configuration"""
    print_step(5, "Validating Server Configuration")
    
    server_path = WEB_APP_DIR / 'server.py'
    
    if not server_path.exists():
        print(f"  {Fore.RED}✗ server.py not found")
        return False
    
    # Read and check configuration
    with open(server_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'NUM_FRAMES': 'NUM_FRAMES = ' in content,
        'CONFIDENCE_THRESHOLD': 'CONFIDENCE_THRESHOLD = ' in content,
        'MediaPipe': 'mediapipe' in content,
        'PyTorch': 'torch' in content,
        'Socket.IO': 'socketio' in content,
    }
    
    for check_name, result in checks.items():
        status = f"{Fore.GREEN}✓" if result else f"{Fore.RED}✗"
        print(f"  {status} {check_name}")
    
    return all(checks.values())


def validate_client_config():
    """Validate client configuration"""
    print_step(6, "Validating Client Configuration")
    
    client_path = WEB_APP_DIR / 'static' / 'app.js'
    
    if not client_path.exists():
        print(f"  {Fore.RED}✗ app.js not found")
        return False
    
    with open(client_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'WebSocket': 'Socket.IO' in content or 'socket' in content,
        'Frame Buffer': 'frameBuffer' in content,
        'NUM_FRAMES': 'NUM_FRAMES' in content,
        'Prediction Handling': 'response_back' in content,
    }
    
    for check_name, result in checks.items():
        status = f"{Fore.GREEN}✓" if result else f"{Fore.RED}✗"
        print(f"  {status} {check_name}")
    
    return all(checks.values())


def show_configuration_summary():
    """Display current configuration"""
    print_step(7, "Configuration Summary")
    
    config_info = {
        'Project Root': PROJECT_ROOT,
        'Web App Dir': WEB_APP_DIR,
        'Models Dir': MODELS_DIR,
        'Model Path': CHECKPOINTS_DIR / 'best.pth',
        'Label Map': CHECKPOINTS_DIR / 'label_map.json',
    }
    
    for key, value in config_info.items():
        exists = f"{Fore.GREEN}[exists]" if Path(value).exists() else f"{Fore.YELLOW}[missing]"
        print(f"  {key}:")
        print(f"    {value} {exists}")
    
    # Read current settings from server.py
    print(f"\n  {Fore.CYAN}Server Configuration:")
    try:
        server_path = WEB_APP_DIR / 'server.py'
        with open(server_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'NUM_FRAMES' in line and '=' in line and not line.strip().startswith('#'):
                    print(f"    {line.strip()}")
                elif 'CONFIDENCE_THRESHOLD' in line and '=' in line and not line.strip().startswith('#'):
                    print(f"    {line.strip()}")
    except Exception as e:
        print(f"    {Fore.YELLOW}Could not read config: {str(e)}")


def show_next_steps():
    """Display next steps"""
    print_step(8, "Next Steps")
    
    print(f"\n  {Fore.CYAN}1. Start the server:")
    print(f"     {Fore.WHITE}cd {PROJECT_ROOT}")
    print(f"     {Fore.WHITE}python web_app/server.py")
    
    print(f"\n  {Fore.CYAN}2. Open in browser:")
    print(f"     {Fore.WHITE}http://127.0.0.1:5000")
    
    print(f"\n  {Fore.CYAN}3. Verify connection:")
    print(f"     {Fore.WHITE}• Wait for 'Connected to server' message")
    print(f"     {Fore.WHITE}• Allow camera access in browser")
    print(f"     {Fore.WHITE}• Check WebSocket status (should show green)")
    
    print(f"\n  {Fore.CYAN}4. Test predictions:")
    print(f"     {Fore.WHITE}• Show sign language gesture to camera")
    print(f"     {Fore.WHITE}• Wait for predictions to appear")
    print(f"     {Fore.WHITE}• Check confidence threshold and adjust if needed")
    
    print(f"\n  {Fore.CYAN}5. Adjust parameters:")
    print(f"     {Fore.WHITE}• Change 'Number of Frames' (5-100)")
    print(f"     {Fore.WHITE}• Adjust 'Confidence Threshold' slider")
    print(f"     {Fore.WHITE}• Settings update in real-time")


def run_health_check():
    """Run comprehensive health check"""
    print_step(9, "Running Health Check")
    
    checks = [
        ("Python Version", check_python_version()),
        ("Model Files", check_model_files()),
        ("Dependencies", check_dependencies()),
        ("Server Config", validate_server_config()),
        ("Client Config", validate_client_config()),
    ]
    
    all_passed = all(result for _, result in checks)
    
    print(f"\n  {Fore.CYAN}Health Check Summary:")
    for check_name, result in checks:
        status = f"{Fore.GREEN}PASS" if result else f"{Fore.RED}FAIL"
        print(f"    {status}: {check_name}")
    
    if all_passed:
        print(f"\n  {Fore.GREEN}✓ All checks passed! Ready to run.")
    else:
        print(f"\n  {Fore.YELLOW}⚠ Some checks failed. Please fix issues before running.")
    
    return all_passed


def main():
    """Main execution"""
    print_header()
    
    try:
        # Run all checks
        create_directories()
        
        if not run_health_check():
            print(f"\n{Fore.RED}Setup incomplete. Please fix the issues above.")
            return 1
        
        # Show configuration
        show_configuration_summary()
        
        # Show next steps
        show_next_steps()
        
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}Setup completed successfully! 🎉")
        print(f"{Fore.GREEN}{'='*60}\n")
        
        return 0
    
    except Exception as e:
        print(f"\n{Fore.RED}Error during setup: {str(e)}")
        logger.exception("Setup error")
        return 1


if __name__ == '__main__':
    sys.exit(main())
