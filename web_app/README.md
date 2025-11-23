# 🤟 Vietnamese Sign Language Recognition - Web Application

Real-time sign language recognition web application using Flask + Socket.IO with MediaPipe and LSTM deep learning model.

## 📋 Features

✅ **Real-time Webcam Streaming**: Capture video frames directly from browser
✅ **WebSocket Communication**: Full-duplex bidirectional communication via Socket.IO
✅ **Deep Learning Inference**: LSTM-based sign recognition
✅ **Dynamic Configuration**: Change number of frames and confidence threshold on-the-fly
✅ **Prediction History**: Accumulate recognized signs into sentences
✅ **Statistics Dashboard**: Track predictions and connection status
✅ **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WEB BROWSER (Client)                    │
│                                                             │
│  • getUserMedia() → Capture webcam frames                  │
│  • Canvas → Encode frames to base64                        │
│  • Socket.IO → Send frames to server                       │
└─────────────────┬───────────────────────────────────────────┘
                  │ WebSocket
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK SERVER (Backend)                   │
│                                                             │
│  • Receive frame buffer (25 frames)                        │
│  • MediaPipe Holistic → Extract keypoints                 │
│  • Stack into (25, 225) tensor                            │
│  • LSTM Model → Prediction                                │
│  • Socket.IO → Send result back                           │
└─────────────────┬───────────────────────────────────────────┘
                  │ WebSocket
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  WEB BROWSER (Display)                      │
│                                                             │
│  • Display prediction label                               │
│  • Show confidence score                                  │
│  • Update prediction history                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained model: `models/checkpoints/best.pth`
- Label map: `models/checkpoints/label_map.json`

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure model is trained:
```bash
python -m src.model.train \
    --data_dir data/splits \
    --seq_len 25 \
    --epochs 100
```

3. Run the web server:
```bash
python web_app/server.py
```

4. Open in browser:
```
http://127.0.0.1:5000
```

## 📊 Configuration

### Environment Variables

In `web_app/server.py`, modify these constants:

```python
NUM_FRAMES = 25                    # Number of frames per sequence
CONFIDENCE_THRESHOLD = 0.30        # Minimum confidence for prediction
IMAGE_WIDTH = 640                  # Webcam image width
IMAGE_HEIGHT = 480                 # Webcam image height
```

### Runtime Configuration

Via the web interface, you can dynamically update:

- **Number of Frames**: 5-100 (default: 25)
- **Confidence Threshold**: 0.0-1.0 (default: 0.30)

## 🔄 Data Flow

### Frame Capture (Client-side)

```javascript
// Capture frames at 25 FPS
// Store in frameBuffer array
// When frameBuffer.length == NUM_FRAMES:
//   → Send to server via socket.emit('image', frameBuffer)
//   → Remove 2 frames for sliding window overlap
```

### Processing (Server-side)

```python
@socketio.on('image')
def process_image(data_images):
    # 1. Decode base64 frames
    # 2. Extract keypoints (225-dim vector)
    # 3. Stack into (NUM_FRAMES, 225) array
    # 4. Pad/truncate to NUM_FRAMES
    # 5. Convert to tensor: (1, NUM_FRAMES, 225)
    # 6. Forward pass through LSTM
    # 7. Get prediction with confidence
    # 8. Emit result back to client
```

### Response (Client-side)

```javascript
socket.on('response_back', function(data) {
    // data = {
    //   label: "người",
    //   confidence: 0.92,
    //   all_probs: {...}
    // }
    
    // Update UI
    // Add to history if confidence > threshold
});
```

## 🎯 API Endpoints & Events

### Socket.IO Events

#### Client → Server

**`image` (emit)**
- Data: `[base64_frame1, base64_frame2, ..., base64_frame_N]`
- Purpose: Send captured frames for processing

**`config_update` (emit)**
- Data: `{num_frames: 25, confidence_threshold: 0.30}`
- Purpose: Update server configuration

#### Server → Client

**`response_back` (emit)**
- Data: `{label, confidence, all_probs}`
- Purpose: Send prediction result

**`config_updated` (emit)**
- Data: `{num_frames, confidence_threshold}`
- Purpose: Confirm configuration update

**`connect_response` (emit)**
- Data: `{data: "Connected to server"}`
- Purpose: Confirm connection

## 🎨 UI Components

### Main Page Sections

1. **Header**
   - Application title
   - Subtitle

2. **Video Section**
   - Live webcam stream
   - Connection and frame count status

3. **Control Section**
   - Prediction display (large label)
   - Confidence percentage
   - Top 5 predictions

4. **Configuration Panel**
   - Number of frames input
   - Confidence threshold slider
   - Reset button

5. **Prediction History**
   - Accumulated signs/sentence
   - Clear history button

6. **Statistics Dashboard**
   - Total predictions count
   - Connection status
   - Last confidence score

## 📊 Keypoint Format

Each frame yields a **225-dimensional feature vector**:

```
Pose Landmarks:    33 × 3 = 99 dims
Left Hand:         21 × 3 = 63 dims
Right Hand:        21 × 3 = 63 dims
─────────────────────────────────
Total:            75 × 3 = 225 dims
```

## 🔧 Troubleshooting

### Issue: "Cannot access webcam"
- **Solution**: Grant camera permission in browser settings
- Check browser privacy settings

### Issue: "Model not found"
- **Solution**: Train model first
```bash
python -m src.model.train --data_dir data/splits
```

### Issue: "WebSocket connection failed"
- **Solution**: Check firewall/proxy settings
- Ensure Flask server is running on correct port

### Issue: Slow predictions
- **Solution**: Reduce NUM_FRAMES or use GPU
- Check server logs for bottlenecks

## 🔗 Model Integration

The web app expects:

```
models/checkpoints/
├── best.pth              # Model weights + optimizer
└── label_map.json        # Class names: ["người", "tôi", "Việt Nam"]
```

To use a custom model:

```python
# In server.py, modify paths:
MODEL_PATH = '/path/to/your/model.pth'
LABEL_MAP_PATH = '/path/to/your/label_map.json'
```

## 📈 Performance Considerations

- **Frame Rate**: 25 FPS capture rate
- **Sequence Length**: 25 frames (configurable)
- **Inference Time**: ~50-100ms (GPU), ~200-300ms (CPU)
- **Latency**: Total round-trip ~200-400ms

## 🚀 Production Deployment

For production deployment:

1. Use production WSGI server (Gunicorn):
```bash
pip install gunicorn
gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:5000 web_app.server:app
```

2. Use HTTPS with SSL certificates

3. Configure CORS properly:
```python
socketio = SocketIO(app, cors_allowed_origins=["https://yourdomain.com"])
```

4. Add authentication if needed

## 📝 File Structure

```
web_app/
├── server.py              # Flask + Socket.IO backend
├── templates/
│   └── index.html         # Main page template
└── static/
    ├── app.js             # Client-side JavaScript
    └── style.css          # Styling
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## 📄 License

[Your License]

## ✉️ Contact

For questions or issues, please contact: [Your Email]

---

**Last Updated**: November 22, 2025
**Version**: 1.0.0
**Status**: Production Ready ✅
