import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
import mss

from window_detect import find_window_hwnd, get_client_bbox, focus_window, is_alive


# -------------------------
# CONFIG
# -------------------------
TITLE = "Zuma Deluxe"
PROC = None

SAVE_DIR = Path("captures")
SAVE_DIR.mkdir(exist_ok=True)

SAVE_EVERY_SEC = 1.0

USE_TEMPLATE_CROP = True
FALLBACK_SAVE_FULL = False

SCORE_TEMPLATE = Path("templates/score.png")
X_TEMPLATE = Path("templates/x.png")

MATCH_THRESH = 0.65
SCALES = (0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)

KEEP_FOCUSED = True
FOCUS_EVERY_SEC = 1.0

LOCK_AFTER_FIRST = True
REDETECT_EVERY_SEC = 10.0


# -------------------------
# Load templates once (grayscale)
# -------------------------
_SCORE_T = cv2.imread(str(SCORE_TEMPLATE), cv2.IMREAD_GRAYSCALE)
_X_T = cv2.imread(str(X_TEMPLATE), cv2.IMREAD_GRAYSCALE)

if _SCORE_T is None:
    raise FileNotFoundError(f"Missing template: {SCORE_TEMPLATE}")
if _X_T is None:
    raise FileNotFoundError(f"Missing template: {X_TEMPLATE}")


# -------------------------
# Helpers
# -------------------------
def make_filename(t: float) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)
    return f"{ts}_{ms:03d}.png"


def match_best(
    gray: np.ndarray,
    templ_gray: np.ndarray,
    scales=SCALES,
    method=cv2.TM_CCOEFF_NORMED
) -> Tuple[float, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Multi-scale template matching:
    returns: best_score, best_loc(x,y), best_wh(w,h)
    """
    best_score = -1.0
    best_loc = None
    best_wh = None

    th0, tw0 = templ_gray.shape[:2]

    for s in scales:
        tw, th = int(tw0 * s), int(th0 * s)
        if tw < 5 or th < 5:
            continue

        t = cv2.resize(templ_gray, (tw, th), interpolation=cv2.INTER_AREA)
        if th >= gray.shape[0] or tw >= gray.shape[1]:
            continue

        res = cv2.matchTemplate(gray, t, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if float(max_val) > best_score:
            best_score = float(max_val)
            best_loc = max_loc
            best_wh = (tw, th)

    return best_score, best_loc, best_wh


def detect_roi_from_score_and_x(
    frame_bgr: np.ndarray,
    thresh: float = MATCH_THRESH,
    pad: int = 8
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float]]]:
    """
    Detect SCORE and X templates, then create ROI:
      left = x_score
      right = x_x + w_x
      width = right - left
      height = int(width * 0.75)   # your current rule
      top = min(y_score, y_x)
    Returns:
      ((left, top, w, h), (score_score, score_x)) or None
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    s1, loc1, wh1 = match_best(gray, _SCORE_T)
    s2, loc2, wh2 = match_best(gray, _X_T)

    if loc1 is None or loc2 is None or wh1 is None or wh2 is None:
        return None
    if s1 < thresh or s2 < thresh:
        return None

    x_score, y_score = loc1
    x_btn, y_btn = loc2
    w_btn, h_btn = wh2

    left = x_score
    right = x_btn + w_btn
    width = right - left
    if width <= 0:
        return None

    height = int(width * 0.75)
    top = min(y_score, y_btn)
    bottom = top + height

    H, W = frame_bgr.shape[:2]

    # padding + clamp
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(W, right + pad)
    bottom = min(H, bottom + pad)

    w = max(1, right - left)
    h = max(1, bottom - top)

    return (left, top, w, h), (s1, s2)


def run_capture_saver():
    """
    Your original runner: finds the window, captures frames,
    crops ROI if enabled, and saves images to /captures.
    """
    hwnd = find_window_hwnd(title_contains=TITLE, process_name=PROC)
    focus_window(hwnd, click_fallback=False)

    last_save = 0.0
    last_focus = 0.0
    last_redetect = 0.0

    roi = None
    roi_locked = False

    with mss.mss() as sct:
        try:
            while True:
                now = time.time()

                if not is_alive(hwnd):
                    hwnd = find_window_hwnd(title_contains=TITLE, process_name=PROC)
                    focus_window(hwnd, click_fallback=False)

                if KEEP_FOCUSED and (now - last_focus >= FOCUS_EVERY_SEC):
                    focus_window(hwnd, click_fallback=False)
                    last_focus = now

                bbox = get_client_bbox(hwnd)
                frame = np.array(sct.grab(bbox))[:, :, :3]  # BGR

                # ---- Decide ROI (template crop or not) ----
                if USE_TEMPLATE_CROP:
                    need_redetect = (
                        (roi is None) or
                        (not roi_locked) or
                        (REDETECT_EVERY_SEC is not None and (now - last_redetect >= REDETECT_EVERY_SEC))
                    )

                    if need_redetect:
                        found = detect_roi_from_score_and_x(frame)
                        if found is not None:
                            roi, scores = found
                            last_redetect = now
                            if LOCK_AFTER_FIRST:
                                roi_locked = True
                            print("ROI locked:", roi, "scores:", scores)

                    if roi is None and not FALLBACK_SAVE_FULL:
                        time.sleep(0.05)
                        continue

                to_save = frame
                if USE_TEMPLATE_CROP and roi is not None:
                    x, y, w, h = roi
                    to_save = frame[y:y + h, x:x + w]

                if now - last_save >= SAVE_EVERY_SEC:
                    out_path = SAVE_DIR / make_filename(now)
                    cv2.imwrite(str(out_path), to_save)
                    print("saved:", out_path)
                    last_save = now

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopped (Ctrl+C).")


# IMPORTANT: prevents running the capture loop when imported in main.py :contentReference[oaicite:1]{index=1}
if __name__ == "__main__":
    run_capture_saver()
