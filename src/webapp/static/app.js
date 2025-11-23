/*
 * Vietnamese Sign Language Recognition - Web App
 * Real-time frame capture and transmission to server
 * Server performs FSM-based recognition using motion detection
 */

const socket = io();
let video = null;
let canvas = null;
let ctx = null;

let isConnected = false;
let frameCount = 0;
let totalPredictions = 0;

// Capture parameters
const FPS = 25;
const FRAME_INTERVAL = 1000 / FPS;
let lastFrameTime = 0;

// State tracking
let currentState = {
    fsm_state: 'waiting',
    segment_size: 0,
    still_count: 0,
};

// =====================================================
// SOCKET.IO EVENTS
// =====================================================

socket.on('connect', function () {
    console.log('✅ Connected to server');
    isConnected = true;
    updateStatus('🟢 Connected', 'status-connected');
});

socket.on('connect_error', function (error) {
    console.error('❌ Connection error:', error);
    updateStatus('🔴 Connection Error', 'status-error');
});

socket.on('disconnect', function () {
    console.log('❌ Disconnected from server');
    isConnected = false;
    updateStatus('🔴 Disconnected', 'status-error');
});

socket.on('prediction', function (data) {
    console.log('🎉 Prediction received:', data);
    
    // Update prediction display
    const label = data.label;
    const confidence = (data.confidence * 100).toFixed(2);
    const frames = data.frames;
    
    document.getElementById('prediction-label').textContent = `✅ ${label}`;
    document.getElementById('prediction-confidence').textContent = `Confidence: ${confidence}%`;
    document.getElementById('prediction-frames').textContent = `Frames analyzed: ${frames}`;
    
    // Update statistics
    totalPredictions++;
    document.getElementById('total-predictions').textContent = totalPredictions;
    
    // Add to history
    addToHistory(label, confidence);
});

socket.on('status_response', function (data) {
    // Update UI with current FSM state
    currentState = data;
    updateFSMDisplay();
});

// =====================================================
// UI UPDATE FUNCTIONS
// =====================================================

function updateStatus(text, cssClass) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = text;
    statusEl.className = cssClass;
}

function updateFSMDisplay() {
    const stateEl = document.getElementById('fsm-state');
    const state = currentState.fsm_state;
    
    if (state === 'waiting') {
        stateEl.textContent = '⏳ Waiting';
        stateEl.className = 'state-waiting';
    } else if (state === 'recording') {
        stateEl.textContent = '🎬 Recording';
        stateEl.className = 'state-recording';
    }
    
    document.getElementById('segment-size').textContent = currentState.segment_size;
    document.getElementById('still-count').textContent = currentState.still_count;
}

function addToHistory(label, confidence) {
    const history = document.getElementById('history');
    const entry = document.createElement('div');
    entry.className = 'history-entry';
    
    const time = new Date().toLocaleTimeString();
    entry.textContent = `${time} - ${label} (${confidence}%)`;
    
    history.insertBefore(entry, history.firstChild);
    
    // Keep only last 10 entries
    while (history.children.length > 10) {
        history.removeChild(history.lastChild);
    }
}

// =====================================================
// VIDEO CAPTURE
// =====================================================

async function initializeCamera() {
    console.log('📹 Initializing camera...');
    
    try {
        video = document.getElementById('videoElement');
        canvas = document.getElementById('canvasOutput');
        ctx = canvas.getContext('2d');
        
        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        
        video.srcObject = stream;
        
        // Wait for video to load
        video.onloadedmetadata = function () {
            console.log('✅ Camera initialized');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Start capturing frames
            captureFrames();
        };
        
    } catch (error) {
        console.error('❌ Camera error:', error);
        updateStatus('🔴 Camera Error', 'status-error');
    }
}

function captureFrames() {
    const now = Date.now();
    
    // Capture at FPS rate
    if (now - lastFrameTime >= FRAME_INTERVAL) {
        if (isConnected && video && video.readyState === video.HAVE_ENOUGH_DATA) {
            // Draw video frame to canvas (flipped horizontally to match training data)
            ctx.save();
            ctx.scale(-1, 1);  // Flip horizontally
            ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
            ctx.restore();
            
            // Convert to base64 and send
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            socket.emit('frame', {
                image: imageData,
                timestamp: now
            });
            
            frameCount++;
            lastFrameTime = now;
        }
    }
    
    // Request status from server periodically
    if (frameCount % 10 === 0) {
        socket.emit('status');
    }
    
    // Continue capturing
    requestAnimationFrame(captureFrames);
}

// =====================================================
// INITIALIZATION
// =====================================================

document.addEventListener('DOMContentLoaded', function () {
    console.log('🚀 Vietnamese Sign Language Recognition - Web App');
    console.log('Starting real-time gesture recognition...');
    
    initializeCamera();
});

// Cleanup on page unload
window.addEventListener('beforeunload', function () {
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
});
