/**
 * Vietnamese Sign Language Recognition - Web App Frontend
 * Real-time webcam stream processing with WebSocket communication
 */

// =====================================================
// CONFIGURATION
// =====================================================

const FPS = 25;
let NUM_FRAMES = parseInt(document.getElementById('num-frames').textContent);
const CONFIDENCE_THRESHOLD = 0.30;

// =====================================================
// STATE MANAGEMENT
// =====================================================

let frameBuffer = [];
let predictionHistory = [];
let totalPredictions = 0;
let isConnected = false;
let socket = null;

// Gesture recognition state
let currentPrediction = null;
let predictionLocked = false;  // Lock prediction on screen until new gesture

// =====================================================
// SOCKET.IO INITIALIZATION
// =====================================================

function initializeSocket() {
    socket = io({
        transports: ['polling'],  // Force polling instead of websocket
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 5
    });

    // Connection events
    socket.on('connect', function() {
        console.log('✅ Connected to server');
        isConnected = true;
        updateStatus('🟢 Connected', 'status-connected');
        updateConnectionStatus('Connected');
    });

    socket.on('disconnect', function() {
        console.log('❌ Disconnected from server');
        isConnected = false;
        updateStatus('🔴 Disconnected', 'status-error');
        updateConnectionStatus('Disconnected');
    });

    socket.on('connect_response', function(data) {
        console.log('Server response:', data);
    });

    // Handle predictions from server
    socket.on('response_back', function(data) {
        handlePredictionResponse(data);
    });

    // Handle config updates
    socket.on('config_updated', function(data) {
        console.log('Config updated:', data);
        NUM_FRAMES = data.num_frames;
        document.getElementById('num-frames').textContent = NUM_FRAMES;
    });
}

// =====================================================
// VIDEO & WEBCAM SETUP
// =====================================================

function initializeWebcam() {
    const videoElement = document.getElementById('videoElement');

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                videoElement.srcObject = stream;
                videoElement.play();
                console.log('✅ Webcam initialized');
                startFrameCapture();
            })
            .catch(function(error) {
                console.error('❌ Error accessing webcam:', error);
                alert('Error accessing webcam. Please allow camera access.');
                updateStatus('🔴 Webcam Error', 'status-error');
            });
    } else {
        alert('Your browser does not support webcam access.');
        updateStatus('🔴 Browser Error', 'status-error');
    }
}

// =====================================================
// FRAME CAPTURE & PROCESSING - REAL-TIME SLIDING WINDOW
// =====================================================

function startFrameCapture() {
    const video = document.getElementById('videoElement');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    let frameCounter = 0;  // Track frames for logging

    setInterval(function() {
        // Check if video is ready
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            return;
        }

        // Set canvas size
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw frame on canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to base64 with LOW quality to reduce size
        const frameData = canvas.toDataURL('image/jpeg', 0.5);

        // Add to buffer
        if (frameData.length > 50) {
            frameBuffer.push(frameData);
            frameCounter++;

            // Show every 10th frame capture in console
            if (frameCounter % 10 === 0) {
                console.log(`📹 Captured ${frameCounter} frames total, buffer: ${frameBuffer.length}/${NUM_FRAMES}`);
            }

            // When buffer is full, send to server (TRUE REAL-TIME: NO BUTTON NEEDED)
            if (frameBuffer.length === NUM_FRAMES) {
                sendFrameBufferRealTime();
            }
        }
    }, 1000 / FPS);  // Capture at specified FPS
}

function updateFrameCount() {
    document.getElementById('frame-count').textContent = frameBuffer.length;
}

/**
 * REAL-TIME sending: Sliding window approach
 * - When 25 frames buffered → send immediately (no button click)
 * - Keep last frame, add 1 new frame for overlap (96% overlap)
 * - This creates continuous, fluid predictions like ASL demo
 */
function sendFrameBufferRealTime() {
    if (!isConnected || frameBuffer.length === 0) {
        return;
    }

    try {
        console.log(`🔄 REAL-TIME: Sending ${frameBuffer.length} frames for prediction...`);
        
        // Send all frames as array - server will process immediately
        socket.emit('process_frames', frameBuffer, (response) => {
            if (response && response.success) {
                console.log('✅ Server processed frames');
            }
        });

        // SLIDING WINDOW (ASL style): Keep last frame, remove first frame
        // This prevents re-predicting identical old frames
        if (frameBuffer.length > 1) {
            frameBuffer = frameBuffer.slice(-1);  // Keep only last 1 frame
            console.log('📊 Sliding window: keeping last frame, removing first');
        }
        updateFrameCount();

    } catch (error) {
        console.error('❌ Error sending frames:', error);
    }
}

// =====================================================
// PREDICTION HANDLING
// =====================================================

function handlePredictionResponse(data) {
    console.log('📥 Prediction response:', data);

    const label = data.label || 'ERROR';
    const confidence = data.confidence || 0;

    // Skip if no valid prediction
    if (label === 'PROCESSING' || label === undefined) {
        return;
    }

    // Update display
    updatePredictionLabel(label);
    updateConfidence(confidence);

    // Update statistics only for real predictions (not NONE)
    if (label !== 'ERROR' && label !== 'NONE') {
        totalPredictions++;
        document.getElementById('total-predictions').textContent = totalPredictions;
        document.getElementById('last-confidence').textContent = (confidence * 100).toFixed(2) + '%';

        // Add to history
        if (confidence > CONFIDENCE_THRESHOLD) {
            addToPredictionHistory(label, confidence);
        }
    }

    console.log('✅ Prediction displayed: ' + label);
}

function updatePredictionLabel(label) {
    const element = document.getElementById('prediction-label');
    
    if (label === 'NONE') {
        element.textContent = '🔍 Waiting for gesture...';
        element.style.color = '#999';
    } else if (label === 'ERROR') {
        element.textContent = '❌ Error';
        element.style.color = '#f44336';
    } else {
        element.textContent = label;
        element.style.color = '#667eea';
    }
}

function updateConfidence(confidence) {
    const percentage = (confidence * 100).toFixed(2);
    document.getElementById('prediction-confidence').textContent = `Confidence: ${percentage}%`;
    document.getElementById('last-confidence').textContent = percentage + '%';
}

function updateAllPredictions(allProbs) {
    const container = document.getElementById('all-predictions');
    container.innerHTML = '';

    // Sort by probability descending
    const sorted = Object.entries(allProbs)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);  // Show top 5

    sorted.forEach(([label, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'pred-item';
        div.innerHTML = `
            <div class="pred-label">${label}</div>
            <div class="pred-prob">${percentage}%</div>
        `;
        container.appendChild(div);
    });
}

function addToPredictionHistory(label, confidence) {
    // Avoid duplicates in quick succession
    if (predictionHistory.length > 0) {
        const lastLabel = predictionHistory[predictionHistory.length - 1].label;
        if (lastLabel === label) {
            return;  // Skip duplicate
        }
    }

    predictionHistory.push({
        label: label,
        confidence: confidence,
        timestamp: new Date().toLocaleTimeString()
    });

    updateSentenceDisplay();
}

function updateSentenceDisplay() {
    const display = document.getElementById('sentence-display');
    
    if (predictionHistory.length === 0) {
        display.innerHTML = '<span class="placeholder">Predictions will appear here...</span>';
        return;
    }

    // Build sentence
    let sentence = predictionHistory
        .map(p => `<strong>${p.label}</strong>`)
        .join(' - ');

    display.innerHTML = sentence;
}

// =====================================================
// UI UPDATES
// =====================================================

function updateStatus(text, className) {
    const element = document.getElementById('status');
    element.textContent = text;
    element.className = className;
}

function updateConnectionStatus(status) {
    document.getElementById('connection-status').textContent = status;
}

// =====================================================
// CONTROLS
// =====================================================

function updateNumFrames() {
    const input = document.getElementById('num-frames-input');
    const newValue = parseInt(input.value);

    if (newValue < 5 || newValue > 100) {
        alert('Number of frames must be between 5 and 100');
        input.value = NUM_FRAMES;
        return;
    }

    NUM_FRAMES = newValue;
    frameBuffer = [];  // Clear buffer when NUM_FRAMES changes
    updateFrameCount();

    // Send config update to server
    if (socket && isConnected) {
        socket.emit('config_update', { num_frames: NUM_FRAMES });
        console.log(`✅ Updated NUM_FRAMES to ${NUM_FRAMES}`);
    }
}

function updateThreshold() {
    const input = document.getElementById('confidence-threshold-input');
    const newValue = parseFloat(input.value);

    if (newValue < 0 || newValue > 1) {
        alert('Confidence threshold must be between 0 and 1');
        input.value = CONFIDENCE_THRESHOLD;
        return;
    }

    // Send config update to server
    if (socket && isConnected) {
        socket.emit('config_update', { confidence_threshold: newValue });
        console.log(`✅ Updated confidence threshold to ${newValue}`);
    }
}

function resetPrediction() {
    document.getElementById('prediction-label').textContent = 'Waiting...';
    document.getElementById('prediction-label').style.color = '#999';
    document.getElementById('prediction-confidence').textContent = 'Confidence: --';
    document.getElementById('all-predictions').innerHTML = '';
    console.log('✅ Prediction reset');
}

function clearHistory() {
    predictionHistory = [];
    updateSentenceDisplay();
    totalPredictions = 0;
    document.getElementById('total-predictions').textContent = '0';
    console.log('✅ History cleared');
}

// =====================================================
// INITIALIZATION
// =====================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('📱 Initializing Vietnamese Sign Language Web App...');

    // Initialize Socket.IO
    initializeSocket();

    // Initialize webcam
    setTimeout(initializeWebcam, 500);

    // Set initial values
    document.getElementById('threshold-display').textContent = CONFIDENCE_THRESHOLD.toFixed(2);

    console.log('✅ App initialized successfully');
});

// =====================================================
// UTILITY FUNCTIONS
// =====================================================

function logMessage(type, message) {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${timestamp}] ${type}: ${message}`);
}

// Graceful shutdown
window.addEventListener('beforeunload', function() {
    if (socket) {
        socket.disconnect();
        console.log('🔌 Socket disconnected');
    }
});
