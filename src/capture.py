import time
from pathlib import Path

import numpy as np
import cv2
import mss

from window_detect import find_window_hwnd, get_client_bbox, focus_window

def detect_game_roi(frame_bgr):
    H, W = frame_bgr.shape[:2]

    # تجاهل أعلى الصفحة (التبويبات/الهيدر)
    y0 = int(0.25 * H)
    search = frame_bgr[y0:, :]

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # لوحة اللعب ألوانها قوية => Saturation أعلى من الخلفية الرمادية
    mask = ((s > 55) & (v > 40)).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)  # :contentReference[oaicite:2]{index=2}

    # فلترة منطقية (مساحة ونسبة أبعاد) حتى ما ياخذ أشياء غلط
    area = w * h
    if area < 0.10 * (W * H):
        return None

    ar = w / float(h)
    if not (0.9 <= ar <= 2.5):
        return None

    # رجّع للإحداثيات الأصلية (نعوّض y0)
    pad = 8
    x = max(0, x + pad)
    y = max(0, (y + y0) + pad)
    w = max(1, w - 2 * pad)
    h = max(1, h - 2 * pad)
    return (x, y, w, h)

TITLE = "Zuma Deluxe"   # or "free Zuma game online"
PROC = None             # (recommended) keep None for robustness

SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

SAVE_EVERY_SEC = 1    # half second
LOCK_ROI_AFTER_FIRST = True  # lock ROI after first successful detection
REDETECT_EVERY_SEC = 10.0     # set None to never re-detect

def make_filename(t: float) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)
    return f"{ts}_{ms:03d}.png"


# Find window and focus once
hwnd = find_window_hwnd(title_contains=TITLE, process_name=PROC)
focus_window(hwnd, click_fallback=False)

last_save = 0.0
last_focus_check = 0.0
last_redetect = 0.0

game_roi = None
roi_locked = False

with mss.mss() as sct:
    try:
        while True:
            now = time.time()

            # keep focused (optional)
            if now - last_focus_check > 1.0:
                focus_window(hwnd, click_fallback=False)
                last_focus_check = now

            bbox = get_client_bbox(hwnd)
            frame = np.array(sct.grab(bbox))[:, :, :3]  # BGR (OpenCV default)

            # ROI detection policy
            # داخل while True بعد ما تجيب frame

            # اكتشف ROI (مرة) ثم قفلها
            if game_roi is None or not roi_locked:
                found = detect_game_roi(frame)
                if found is not None:
                    game_roi = found
                    roi_locked = True
                    print("ROI locked:", game_roi)

            # إذا لسا ما لقينا ROI -> لا تحفظ كامل، استنى
            if game_roi is None:
                time.sleep(0.05)
                continue

            # قصّ منطقة اللعب فقط
            x, y, w, h = game_roi
            game = frame[y:y+h, x:x+w]

            # احفظ كل SAVE_EVERY_SEC
            if now - last_save >= SAVE_EVERY_SEC:
                out_path = SAVE_DIR / make_filename(now)
                cv2.imwrite(str(out_path), game)  # :contentReference[oaicite:3]{index=3}
                print("saved:", out_path)
                last_save = now


            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
