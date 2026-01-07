# Zuma Window Detector (UI Screenshots)

This project detects the **Zuma game window** from UI screenshots using a **two-stage hybrid approach** designed for robustness and real-time use.

---

## Detection Strategy

### 1. Geometric Window Detection
Candidate windows are detected using classical computer vision techniques:
- Sobel gradient magnitude
- Adaptive thresholding
- Morphological filtering
- Contour-based rectangle detection

This stage finds **all plausible UI windows** on the screen.

---

### 2. Title Bar Anchor (Helper Signal)
A lightweight **multi-scale template match** is applied to detect the *Zuma title text* globally.

- If the title is found:
  - The game window is selected as the window **directly below** the title
- If the title is not found (fullscreen mode):
  - The detector **falls back** to the largest detected window

This avoids OCR, works in fullscreen and windowed modes, and guarantees **exactly one window** is selected.

---

## Key Properties

- No OCR
- No hard-coded window positions
- Works in fullscreen and windowed modes
- Deterministic fallback behavior
- Real-time safe
- Modular and reusable design

---

## Project Structure
    img/
    ├── tests/        # Input screenshots\n"
    ├── templates/    # Zuma title template(s)\n"
    └── outputs/      # Detection results\n"
---

## Run

```bash
python src/main.py
