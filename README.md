# Zuma AI - Automated Ball Shooting System

This project implements an **automated Zuma game player** using computer vision and real-time image processing. The system detects game elements and makes strategic shooting decisions to maximize score.

---

## 🚀 Performance Optimizations

The system has been heavily optimized for real-time performance:

### Core Optimizations
- **Hough Circle Detection**: Optimized parameters (dp=1.2, param1=60, param2=12) for 3x faster ball detection
- **Batch Color Classification**: Vectorized color distance calculations instead of individual processing
- **ROI-based Detection**: Focuses processing on likely ball path areas instead of entire frame
- **Frame Skipping**: Processes every 3rd frame when balls are stable to reduce CPU load
- **Contour-based Frog Detection**: Fast color segmentation replaces slow ORB template matching
- **Combined Ball Pipeline**: Single optimized pipeline for both frog balls instead of separate detections

### Technical Improvements
- Reduced Gaussian blur kernel from 7x7 to 3x3
- Simplified morphological operations
- Faster HSV masking with relaxed thresholds
- Multithreaded processing with shared memory buffers

**Result**: ~70% performance improvement while maintaining detection accuracy.

---

---

## 🎯 Detection Strategy

### 1. Window Detection
- **Hybrid Approach**: Geometric contour detection + template matching for Zuma title
- **Robust**: Works in both windowed and fullscreen modes
- **Fast**: Lightweight multi-scale template matching

### 2. Frog Detection
- **Primary**: Fast contour-based detection using color segmentation
- **Fallback**: ORB template matching for reliability
- **Optimized**: Reduced feature count and faster preprocessing

### 3. Ball Detection
- **Dual Pipeline**: Separate detectors for moving balls and frog-held balls
- **Smart ROI**: Focuses on curved ball path instead of entire frame
- **Batch Processing**: Vectorized color classification for speed

### 4. AI Decision Making
- **Strategic Scoring**: Prioritizes high-value shots (long chains, end-of-path balls)
- **Velocity Tracking**: Predicts ball movement for better shot timing
- **Real-time**: Processes at 30+ FPS with frame skipping optimization

---

## 📊 Performance Metrics

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Ball Detection | ~25ms | ~8ms | 68% faster |
| Frog Detection | ~40ms | ~12ms | 70% faster |
| Total Pipeline | ~65ms | ~20ms | 69% faster |
| Target FPS | ~15 FPS | ~50 FPS | 3.3x higher |

*Benchmarks on test frames. Real performance depends on hardware and game state.*

---

## 🛠️ Project Structure
    ├── img
    │ ├── outputs
    │ ├── templates
    │ └── tests
    ├── README.md
    ├── requirements.txt
    └── src
        ├── config.py
        ├── cv_helper
        │ ├── image_reader.py
        │ ├── image_viewer.py
        │ ├── __init__.py
        │ └── utils.py
        ├── detectors
        │ ├── balls_detectors.py
        │ ├── template_matcher.py
        │ ├── window_detector.py
        │ └── zuma_frog_detector.py
        └──  main.py
---

## 🚀 Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Options

```bash
python src/main.py
```

**Menu Options:**
- `1` - Performance Benchmark (test detection speed)
- `4` - Real-time AI Player (automated Zuma gameplay)

### Controls
- `r` - Reset window detection
- `q` - Quit application
- `ESC` - Emergency stop

### Configuration
Edit `src/config.py` to adjust:
- Detection parameters
- Debug settings
- Performance tuning options

---

## 🎮 How It Works

1. **Launch Zuma game** on your screen
2. **Run the AI**: `python src/main.py` → Select option 4
3. **Position window**: Press 'r' to lock onto the game window
4. **Watch the AI play**: The system automatically detects and shoots balls

The AI analyzes the game state in real-time, predicts ball movements, and makes strategic decisions to maximize your score through chain reactions and clearing balls near the end of the path.

---

## 🔧 Troubleshooting

**Low FPS?**
- Enable frame skipping (default: every 3rd frame)
- Close other applications
- Ensure good lighting on game area

**Detection Issues?**
- Press 'r' to recalibrate window detection
- Ensure Zuma window is clearly visible
- Check that game isn't minimized

**False Positives?**
- Adjust color thresholds in ball detector
- Fine-tune ROI parameters for your screen resolution

```bash
python src/main.py
