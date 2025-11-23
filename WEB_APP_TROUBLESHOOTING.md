# 🔧 Web Application Troubleshooting Guide

## Common Issues & Solutions

### 🌐 Connection Issues

#### Issue 1: "WebSocket connection failed"
**Symptoms:**
- Red indicator showing "Not connected"
- Console shows WebSocket error
- Cannot send frames to server

**Solutions:**

1. **Check Flask server is running**
   ```bash
   # Terminal 1: Start server
   python web_app/server.py
   
   # Should see output:
   # INFO:werkzeug: * Running on http://127.0.0.1:5000
   # INFO:socketio.server: Server initialized for eventlet.
   ```

2. **Check port is not in use**
   ```bash
   # Windows PowerShell
   Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue
   
   # If in use, kill the process:
   Stop-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess -Force
   ```

3. **Check firewall settings**
   - Allow Python.exe through firewall
   - Check corporate proxy settings
   - Try localhost instead of 127.0.0.1

4. **Browser compatibility**
   - Use Chrome, Firefox, or Edge (not Safari)
   - Clear browser cache
   - Try incognito/private window
   - Update browser to latest version

5. **Check CORS settings** (if behind proxy)
   ```python
   # In server.py, modify:
   socketio = SocketIO(app, 
       cors_allowed_origins=["*"],  # Allow all for testing
       cors_methods=["GET", "POST"]
   )
   ```

---

#### Issue 2: "Connection timeout"
**Symptoms:**
- Long delay before "Not connected"
- Repeated connection attempts
- Slow page load

**Solutions:**

1. **Check network connectivity**
   ```bash
   ping 127.0.0.1
   ```

2. **Increase timeout**
   ```javascript
   // In app.js, modify:
   socket = io.connect(null, {
       reconnection: true,
       reconnectionDelay: 1000,
       reconnectionDelayMax: 5000,
       reconnectionAttempts: 5,
       transports: ['websocket', 'polling']  // Fallback to polling
   });
   ```

3. **Check server logs**
   ```bash
   # Look for error messages in Flask console
   # Check for: ERROR, Exception, Traceback
   ```

---

### 📷 Camera Issues

#### Issue 3: "Cannot access webcam"
**Symptoms:**
- Black video element
- No error message visible
- Permission request not appearing

**Solutions:**

1. **Grant camera permission**
   - Click browser notification/prompt
   - Settings → Privacy → Camera → Allow site
   - Check Chrome: ⋮ → Settings → Privacy → Site permissions → Camera

2. **Check if camera is being used**
   ```bash
   # Windows: Check Device Manager
   # Windows PowerShell
   Get-PnpDevice -FriendlyName "*camera*" -ErrorAction SilentlyContinue
   
   # Close other apps using camera (Zoom, Skype, etc.)
   ```

3. **Test camera directly**
   - Open camera app or Zoom
   - If no video there either, hardware issue
   - Check if camera is enabled in BIOS

4. **Try different browser**
   - Different browser may have different permissions
   - Incognito window (fresh permissions)

5. **Debug in browser console**
   ```javascript
   // Open DevTools (F12) → Console
   // Try to manually access camera:
   navigator.mediaDevices.getUserMedia({ video: true })
       .then(stream => {
           console.log("Camera access granted");
           stream.getTracks().forEach(track => track.stop());
       })
       .catch(err => console.error("Camera error:", err));
   ```

---

### 🤖 Model Issues

#### Issue 4: "Model file not found"
**Symptoms:**
- Flask console error: `FileNotFoundError: best.pth`
- Cannot start server
- Error appears immediately on startup

**Solutions:**

1. **Verify model exists**
   ```bash
   # PowerShell
   Test-Path "models/checkpoints/best.pth"
   Get-Item "models/checkpoints/best.pth" | Select-Object Length
   ```

2. **Train model first**
   ```bash
   python -m src.model.train \
       --data_dir data/splits \
       --seq_len 25 \
       --epochs 100 \
       --batch_size 32
   ```

3. **Check file path in server.py**
   ```python
   # Modify if needed:
   MODEL_PATH = Path(__file__).parent.parent / 'models' / 'checkpoints' / 'best.pth'
   print(f"Looking for model at: {MODEL_PATH}")
   print(f"Model exists: {MODEL_PATH.exists()}")
   ```

4. **Check file permissions**
   ```bash
   # Windows: Right-click → Properties → Security
   # Ensure current user has Read permission
   ```

---

#### Issue 5: "Invalid model format" or "Unexpected model error"
**Symptoms:**
- Error on server startup: `RuntimeError: Invalid model`
- Predictions don't work
- Server crashes when trying to load model

**Solutions:**

1. **Verify model format**
   ```python
   import torch
   
   model_path = 'models/checkpoints/best.pth'
   try:
       checkpoint = torch.load(model_path)
       print("Checkpoint loaded successfully")
       print("Keys:", checkpoint.keys())
       print("Model keys present:", 'model_state_dict' in checkpoint or 'state_dict' in checkpoint)
   except Exception as e:
       print(f"Error: {e}")
   ```

2. **Check model architecture matches**
   ```python
   # In server.py:
   # Verify INPUT_SIZE = 225 (75 landmarks × 3 coordinates)
   # Verify output matches number of classes in label_map.json
   
   # Check label map:
   import json
   with open('models/checkpoints/label_map.json') as f:
       labels = json.load(f)
   print(f"Number of classes: {len(labels)}")  # Should match model output
   ```

3. **Retrain if incompatible**
   ```bash
   python -m src.model.train --data_dir data/splits
   ```

---

#### Issue 6: "Predictions are always wrong" or "Always predicts same class"
**Symptoms:**
- Same prediction no matter what gesture
- Confidence always very low
- Random predictions

**Solutions:**

1. **Check label map**
   ```python
   # Make sure label indices match:
   import json
   with open('models/checkpoints/label_map.json') as f:
       label_map = json.load(f)
   print(label_map)  # Should be: {0: "người", 1: "tôi", 2: "Việt Nam"} or similar
   ```

2. **Verify model checkpoint is correct**
   ```bash
   # Check file was actually trained (not a template)
   ls -lh models/checkpoints/best.pth
   # Should be > 1MB
   ```

3. **Check confidence threshold**
   - Reduce threshold slider to see if any predictions pass
   - If still nothing, model may not be trained properly
   - Check training accuracy in train logs

4. **Verify keypoints are extracted correctly**
   ```python
   # Add debug logging to server.py:
   keypoints = extract_keypoints(results)
   print(f"Keypoint shape: {keypoints.shape}")  # Should be (225,)
   print(f"Keypoint min/max: {keypoints.min()}/{keypoints.max()}")  # Should be normalized
   ```

5. **Test model directly**
   ```bash
   python -c "
   import torch
   import json
   
   model = torch.load('models/checkpoints/best.pth')
   # Create dummy input
   dummy = torch.randn(1, 25, 225)
   with torch.no_grad():
       output = model(dummy)
   print(f'Output shape: {output.shape}')
   print(f'Sample prediction: {output[0]}')
   "
   ```

---

### 🎬 Frame Issues

#### Issue 7: "No frames being sent" or "Frame buffer not filling"
**Symptoms:**
- JavaScript console shows: "Buffer not full: 0/25"
- No predictions ever appear
- Console stuck logging frame count

**Solutions:**

1. **Check frame capture rate**
   ```javascript
   // Open DevTools → Console
   // Monitor frame capture:
   console.log(`Frame buffer: ${frameBuffer.length}/${NUM_FRAMES}`);
   ```

2. **Verify NUM_FRAMES setting**
   ```javascript
   // In app.js, check:
   console.log("NUM_FRAMES:", NUM_FRAMES);  // Should be 25 (or your value)
   ```

3. **Check FPS setting**
   ```python
   # In server.py:
   FPS = 25  # Frames per second
   
   # In app.js:
   const FPS = 25;
   const FRAME_INTERVAL = 1000 / FPS;  // ~40ms for 25 FPS
   ```

4. **Enable detailed logging**
   ```javascript
   // Modify app.js:
   if (frameBuffer.length >= NUM_FRAMES) {
       console.log(`Sending ${frameBuffer.length} frames to server`);
       socket.emit('image', frameBuffer.map(f => f.substring(0, 100) + "..."));
   }
   ```

---

#### Issue 8: "Frame size too large" or "Transmission error"
**Symptoms:**
- Frames not sent: "Frame too large"
- WebSocket disconnects after sending frames
- Memory usage keeps increasing

**Solutions:**

1. **Reduce JPEG quality**
   ```javascript
   // In app.js, modify canvas capture:
   context.canvas.toDataURL('image/jpeg', 0.7)  // 70% quality instead of 100%
   ```

2. **Reduce frame size**
   ```python
   # In server.py:
   IMAGE_WIDTH = 480   # Reduce from 640
   IMAGE_HEIGHT = 360  # Reduce from 480
   ```

3. **Increase WebSocket buffer size**
   ```python
   # In server.py:
   socketio = SocketIO(app, 
       max_http_buffer_size=1e8,  # 100MB buffer
       ping_timeout=60
   )
   ```

4. **Send fewer frames at once**
   ```python
   # Reduce NUM_FRAMES to 15 or 10
   # Smaller sequences = smaller transmission
   ```

---

### 📊 Prediction Issues

#### Issue 9: "Predictions are empty" or "NaN confidence"
**Symptoms:**
- "label: Unknown" or empty prediction
- Confidence shows NaN
- Top 5 predictions are empty

**Solutions:**

1. **Check model output**
   ```python
   # Add to server.py after prediction:
   print(f"Raw model output: {output}")
   print(f"Output shape: {output.shape}")
   print(f"Max confidence: {output.max()}")
   ```

2. **Verify softmax is applied**
   ```python
   # In server.py, ensure:
   import torch.nn.functional as F
   probabilities = F.softmax(output, dim=1)
   confidence, predicted = torch.max(probabilities, 1)
   ```

3. **Check label mapping**
   ```python
   # Make sure index maps to label:
   label_map = json.load(open('models/checkpoints/label_map.json'))
   print(label_map[predicted.item()])
   ```

---

#### Issue 10: "Confidence threshold not working"
**Symptoms:**
- Predictions show even with low confidence
- Changing slider does nothing
- Threshold setting ignored

**Solutions:**

1. **Check configuration update reaches server**
   ```python
   # Add logging to server.py:
   @socketio.on('config_update')
   def config_update(data):
       print(f"Config update received: {data}")
       global CONFIDENCE_THRESHOLD
       CONFIDENCE_THRESHOLD = data['confidence_threshold']
       print(f"New threshold: {CONFIDENCE_THRESHOLD}")
   ```

2. **Verify threshold check in prediction**
   ```python
   # In server.py, check:
   if confidence >= CONFIDENCE_THRESHOLD:
       # Valid prediction
   else:
       # Invalid - should be filtered
   ```

3. **Check JavaScript implementation**
   ```javascript
   // In app.js, verify UI updates:
   confidenceSlider.addEventListener('change', (e) => {
       CONFIDENCE_THRESHOLD = parseFloat(e.target.value);
       socket.emit('config_update', {confidence_threshold: CONFIDENCE_THRESHOLD});
   });
   ```

---

### 💾 Storage & Memory Issues

#### Issue 11: "Out of memory" or "Memory usage keeps growing"
**Symptoms:**
- App slows down over time
- Browser tab becomes unresponsive
- Server crashes after running for hours

**Solutions:**

1. **Clear frame buffer periodically**
   ```javascript
   // In app.js:
   if (frameBuffer.length > NUM_FRAMES * 2) {
       frameBuffer = frameBuffer.slice(-NUM_FRAMES);
       console.log("Cleared excess frames");
   }
   ```

2. **Clear GPU cache**
   ```python
   # In server.py, add after prediction:
   torch.cuda.empty_cache()
   ```

3. **Limit prediction history**
   ```javascript
   // In app.js:
   const MAX_HISTORY = 100;
   if (predictionHistory.length > MAX_HISTORY) {
       predictionHistory = predictionHistory.slice(-MAX_HISTORY);
   }
   ```

4. **Monitor memory in DevTools**
   - F12 → Memory tab
   - Take heap snapshots
   - Check for DOM detached nodes

---

### 🎨 UI/Display Issues

#### Issue 12: "Interface not loading" or "Blank page"
**Symptoms:**
- White/blank page loads
- Console shows 404 errors
- CSS/JS files not loading

**Solutions:**

1. **Check file paths**
   ```bash
   # Verify files exist:
   Test-Path "web_app/templates/index.html"
   Test-Path "web_app/static/app.js"
   Test-Path "web_app/static/style.css"
   ```

2. **Check Flask routing**
   ```python
   # In server.py, verify:
   @app.route('/')
   def index():
       return render_template('index.html')
   
   @app.route('/static/<filename>')
   def static(filename):
       return send_from_directory('static', filename)
   ```

3. **Check browser console for errors**
   - Press F12
   - Click Console tab
   - Look for red errors
   - Check Network tab for 404s

4. **Clear browser cache**
   - Ctrl+Shift+Delete
   - Select "All time"
   - Check "Cookies and cached images"
   - Click Clear

---

#### Issue 13: "Layout broken" or "Elements overlapping"
**Symptoms:**
- Video not visible
- Controls scattered
- Mobile view broken

**Solutions:**

1. **Check CSS file is loaded**
   - DevTools → Sources → static/style.css
   - Should see CSS code, not 404

2. **Test in different browser**
   - CSS rendering varies by browser
   - Try Firefox if Chrome broken

3. **Check responsive breakpoints**
   - Resize window
   - Test at 768px width (mobile)
   - Test at 1200px width (tablet)

4. **Verify CSS syntax**
   ```bash
   # Check for CSS errors:
   # DevTools → Console → Look for CSS parse errors
   ```

---

### ⚡ Performance Issues

#### Issue 14: "Low FPS" or "Predictions are slow"
**Symptoms:**
- Predictions take > 1 second
- Predictions lag behind gestures
- CPU at 100%

**Solutions:**

1. **Enable GPU acceleration**
   ```python
   # In server.py:
   import torch
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   print(f"Using device: {device}")
   
   model = model.to(device)
   input_tensor = input_tensor.to(device)
   ```

2. **Reduce NUM_FRAMES**
   ```python
   # Fewer frames = faster inference
   NUM_FRAMES = 15  # Instead of 25
   ```

3. **Reduce image resolution**
   ```python
   IMAGE_WIDTH = 480   # Instead of 640
   IMAGE_HEIGHT = 360  # Instead of 480
   ```

4. **Profile code**
   ```python
   import time
   
   t1 = time.time()
   keypoints = extract_keypoints(results)
   print(f"Keypoint extraction: {time.time()-t1:.3f}s")
   
   t2 = time.time()
   prediction = model(input_tensor)
   print(f"Model inference: {time.time()-t2:.3f}s")
   ```

---

## 🆘 Getting Help

If issue persists:

1. **Check logs**
   ```bash
   # Server logs in terminal where you ran server.py
   # Browser logs in DevTools Console (F12)
   ```

2. **Enable debug mode**
   ```python
   # In server.py:
   app.config['DEBUG'] = True
   socketio = SocketIO(app, logger=True, engineio_logger=True)
   ```

3. **Verify setup**
   ```bash
   python setup_webapp.py  # Comprehensive health check
   ```

4. **Check documentation**
   - web_app/README.md
   - DEPLOYMENT_GUIDE.md
   - server.py comments

5. **Test components individually**
   ```bash
   # Test model loading
   python -c "import torch; torch.load('models/checkpoints/best.pth')"
   
   # Test Flask
   python web_app/server.py
   
   # Test Socket.IO connection
   # Open browser DevTools → Network → WS
   ```

---

## ✅ Verification Checklist

Before troubleshooting, verify:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] Model file exists: `models/checkpoints/best.pth`
- [ ] Label map exists: `models/checkpoints/label_map.json`
- [ ] Flask server running without errors
- [ ] Browser can reach http://127.0.0.1:5000
- [ ] Camera permission granted in browser
- [ ] WebSocket connection shown as "Connected"
- [ ] Frame capture started (should see count increasing)
- [ ] No JavaScript errors in console (F12)

---

**Last Updated:** November 22, 2025
**Version:** 1.0.0
