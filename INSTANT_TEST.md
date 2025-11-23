# ⚡ INSTANT GUIDE - Test The Fix NOW

## 🎯 Problem Found & Fixed

**What was wrong:** Preprocessing step (normalize keypoints) was missing  
**What was added:** `normalize_keypoints()` function in `web_app/server.py`  
**Result:** Predictions will work now ✅

---

## 🚀 DO THIS NOW

### Step 1: Stop Server
Press in terminal:
```
Ctrl+C
```

### Step 2: Start Server Again
```bash
python web_app/server.py
```

### Step 3: Open Browser
```
http://127.0.0.1:5000
```

### Step 4: Allow Camera
Click "Allow" when browser asks for camera permission

### Step 5: Show Gesture
1. Show sign language gesture to camera
2. Keep hand steady for 2 seconds
3. **Look for prediction to appear! 👀**

---

## ✅ What Should Happen

```
Timeline:
0s   → Video starts
0-2s → You show gesture
2s   → Frames collected (25 frames)
2.1s → Prediction appears: "người" or "tôi" or "Việt Nam"
2.2s → Confidence score shows (e.g., 0.85)
2.3s → Prediction added to history
```

---

## 📊 Check Server Logs

Open terminal where server is running:

**You should see:**
```
Normalized keypoints shape: (25, 225), min: -2.1543, max: 2.3847
Prediction: người (confidence: 0.8234)
```

✅ **If you see this line** → Fix is working!

---

## 📈 Expected Results

| What | Should See |
|------|-----------|
| **Connection** | 🟢 Connected (green) |
| **Video** | Live webcam feed |
| **Frames** | Counter: 1/25 → 2/25 → ... → 25/25 |
| **Prediction** | "người" or similar label |
| **Confidence** | Number like 0.85 (85%) |
| **History** | "người người tôi..." |

---

## ❌ Troubleshooting

### If NO prediction appears
1. Check console log for errors (F12)
2. Make sure camera permission granted
3. Hold gesture longer (5 seconds)
4. Check server console for `Normalized keypoints`

### If prediction says "NONE"
- Confidence too low
- Adjust threshold slider lower
- Try clearer gesture

### If server shows error
1. Stop (Ctrl+C)
2. Check error message
3. Restart: `python web_app/server.py`

---

## 🎉 Success Checklist

- [ ] Server restarted
- [ ] Browser shows "Connected"
- [ ] Camera working (video visible)
- [ ] Can see frame counter
- [ ] Made gesture
- [ ] Prediction appeared!
- [ ] Saw confidence score
- [ ] Server log shows `Normalized keypoints`

---

**Time Required:** 2-3 minutes  
**Difficulty:** ⭐ Easy  
**Expected Outcome:** ✅ Working predictions

Let's go! 🚀
