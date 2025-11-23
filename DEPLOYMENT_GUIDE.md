# 🚀 Web Application Deployment Guide

Complete guide for deploying the Vietnamese Sign Language Recognition web application in different environments.

## 📋 Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Troubleshooting](#troubleshooting)

---

## 🛠️ Local Development

### Requirements

- Python 3.8 or higher
- pip package manager
- Git (optional)

### Setup Steps

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare model:**
```bash
# Ensure model exists at:
# models/checkpoints/best.pth
# models/checkpoints/label_map.json
```

4. **Run server:**
```bash
python web_app/server.py
```

5. **Open browser:**
```
http://127.0.0.1:5000
```

### Development Features

- Hot reload on code changes
- Detailed debug logging
- Browser developer tools support
- WebSocket debugging

---

## 🏭 Production Deployment

### Requirements

- Python 3.8+
- Gunicorn (WSGI server)
- Nginx (reverse proxy)
- SSL certificates
- Supervisor (process manager)

### Step 1: Prepare Application

1. Update `server.py` configuration:
```python
# Before deployment:
app.config['ENV'] = 'production'
app.config['DEBUG'] = False
socketio = SocketIO(app, 
    cors_allowed_origins=["https://yourdomain.com"],
    ping_timeout=60,
    ping_interval=25
)
```

2. Create `.env` file:
```env
FLASK_ENV=production
SECRET_KEY=your-random-secret-key-here
MODEL_PATH=/full/path/to/models/checkpoints/best.pth
LABEL_MAP_PATH=/full/path/to/models/checkpoints/label_map.json
NUM_FRAMES=25
CONFIDENCE_THRESHOLD=0.30
```

3. Load environment variables in `server.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()
NUM_FRAMES = int(os.getenv('NUM_FRAMES', 25))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.30))
```

### Step 2: Install Production Dependencies

```bash
pip install gunicorn python-dotenv
pip install -r requirements.txt
```

### Step 3: Configure Gunicorn

Create `gunicorn_config.py`:
```python
# Gunicorn configuration
bind = "127.0.0.1:8000"
workers = 4
worker_class = "eventlet"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
graceful_timeout = 30
error_logfile = "/var/log/gunicorn/error.log"
access_logfile = "/var/log/gunicorn/access.log"
loglevel = "info"
```

### Step 4: Configure Nginx

Create `/etc/nginx/sites-available/sign-language-web`:
```nginx
upstream sign_language_app {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/your-cert.crt;
    ssl_certificate_key /etc/ssl/private/your-key.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://sign_language_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_redirect off;
    }
    
    location /static/ {
        alias /path/to/web_app/static/;
        expires 1d;
        add_header Cache-Control "public, immutable";
    }
    
    # Health check
    location /health {
        access_log off;
        return 200 "OK\n";
        add_header Content-Type text/plain;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/sign-language-web /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 5: Configure Supervisor

Create `/etc/supervisor/conf.d/sign-language.conf`:
```ini
[program:sign_language_web]
directory=/path/to/sign_language_video_to_text
command=/path/to/venv/bin/gunicorn \
    --config gunicorn_config.py \
    --bind 127.0.0.1:8000 \
    web_app.server:app
user=www-data
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/sign_language_web.log
environment=PATH="/path/to/venv/bin"
stopasgroup=true
stopsignal=TERM
```

Start/manage:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start sign_language_web
sudo supervisorctl status
```

### Step 6: Enable SSL Certificates

Using Let's Encrypt:
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot certonly --nginx -d yourdomain.com -d www.yourdomain.com
```

Renew certificates (automatic):
```bash
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

---

## 🐳 Docker Deployment

### Create Dockerfile

Create `Dockerfile` in project root:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run application
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "web_app.server:app"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - NUM_FRAMES=25
      - CONFIDENCE_THRESHOLD=0.30
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - web
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t sign-language-web:1.0 .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f web

# Stop
docker-compose down
```

---

## ☁️ Cloud Deployment

### AWS Elastic Beanstalk

1. **Install AWS CLI:**
```bash
pip install awsebcli
```

2. **Initialize project:**
```bash
eb init -p docker sign-language-web
```

3. **Create environment:**
```bash
eb create production-env
```

4. **Deploy:**
```bash
eb deploy
```

5. **Monitor:**
```bash
eb status
eb logs
```

### Google Cloud Run

1. **Create `.gcloudignore`:**
```
.git
.gitignore
venv/
__pycache__/
*.pyc
```

2. **Deploy:**
```bash
gcloud run deploy sign-language-web \
    --source . \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --allow-unauthenticated
```

### Azure App Service

1. **Create resource group:**
```bash
az group create --name myResourceGroup --location eastus
```

2. **Create app service plan:**
```bash
az appservice plan create --name myPlan --resource-group myResourceGroup --sku B2
```

3. **Create web app:**
```bash
az webapp create --resource-group myResourceGroup --plan myPlan --name sign-language-web
```

4. **Deploy:**
```bash
az webapp up --resource-group myResourceGroup --name sign-language-web
```

---

## ⚡ Performance Optimization

### 1. Model Optimization

- **Quantization:**
```python
import torch

model = torch.load('best.pth')
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model, 'best_quantized.pth')
```

- **ONNX Export:**
```python
import torch
import onnx

dummy_input = torch.randn(1, 25, 225)
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'])
```

### 2. Caching

Add Redis caching:
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@app.route('/cache_example')
@cache.cached(timeout=3600)
def cached_function():
    return expensive_operation()
```

### 3. Load Balancing

Use multiple Gunicorn workers:
```bash
gunicorn --workers 4 --worker-class eventlet web_app.server:app
```

### 4. Database Connection Pooling

```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    'postgresql://user:pass@localhost/db',
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

---

## 🐛 Troubleshooting

### Issue: WebSocket Connection Failed

**Symptoms:** "Failed to connect to WebSocket"

**Solutions:**
```nginx
# Ensure Nginx is configured for WebSocket upgrade
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
```

### Issue: Model Loading Fails

**Symptoms:** "Model file not found" or "Invalid model format"

**Check:**
```bash
# Verify model exists
ls -lh models/checkpoints/best.pth

# Check file permissions
chmod 644 models/checkpoints/best.pth

# Verify model format
python -c "import torch; torch.load('models/checkpoints/best.pth')"
```

### Issue: High Latency

**Symptoms:** Predictions delayed, slow response

**Solutions:**
1. Enable GPU:
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

2. Reduce model size via quantization

3. Increase Gunicorn workers

4. Enable caching

### Issue: Memory Leak

**Symptoms:** Memory usage keeps increasing

**Solutions:**
```python
# Clear GPU cache regularly
torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    predictions = model(input_tensor)
```

---

## 📊 Monitoring

### Key Metrics

- **Inference Time:** Target < 200ms
- **FPS:** Target 25 FPS consistent
- **Memory Usage:** Monitor GPU/CPU RAM
- **Request Queue:** Monitor pending requests
- **Error Rate:** Track prediction errors

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

---

## ✅ Pre-Deployment Checklist

- [ ] Model and label_map files present
- [ ] All dependencies in requirements.txt
- [ ] Environment variables configured
- [ ] SSL certificates installed (production)
- [ ] Firewall rules configured
- [ ] Error logging setup
- [ ] Monitoring configured
- [ ] Health check endpoint working
- [ ] Database connections pooled
- [ ] API rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers set
- [ ] Performance tested under load

---

**Last Updated:** November 22, 2025
**Version:** 1.0.0
