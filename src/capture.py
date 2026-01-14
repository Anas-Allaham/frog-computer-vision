import time
from pathlib import Path

import numpy as np
import cv2
import mss
import sys

from window_detect import find_window_hwnd, get_client_bbox, focus_window, is_alive

sys.path.append(str(Path(__file__).parent))

from config import SCORE_TEMPLATE_PATH, X_TEMPLATE_PATH, CAPTURES_DIR, GAME_TITLE

TITLE = GAME_TITLE
PROC = None

SAVE_DIR = CAPTURES_DIR
SAVE_DIR.mkdir(exist_ok=True)

SAVE_EVERY_SEC = 1.0

USE_TEMPLATE_CROP = True         
FALLBACK_SAVE_FULL = False        

SCORE_TEMPLATE = SCORE_TEMPLATE_PATH
X_TEMPLATE = X_TEMPLATE_PATH

MATCH_THRESH = 0.65
SCALES = (0.5 , 0.6,0.7 , 0.8, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)

KEEP_FOCUSED = True
FOCUS_EVERY_SEC = 1.0

LOCK_AFTER_FIRST = True
REDETECT_EVERY_SEC = 10.0  


def make_filename(t: float) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(t))
    ms = int((t - int(t)) * 1000)
    return f"{ts}_{ms:03d}.png"


def match_best(gray: np.ndarray, templ_gray: np.ndarray, scales=SCALES,
               method=cv2.TM_CCOEFF_NORMED):
    """
    Multi-scale template matching:
    - نجرب عدة أحجام للـtemplate ونختار أعلى score.
    matchTemplate يعطي response map، وminMaxLoc يطلع أفضل مكان. :contentReference[oaicite:1]{index=1}
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


def detect_roi_from_score_and_x(frame_bgr: np.ndarray,
                                score_path: Path,
                                x_path: Path,
                                thresh: float = MATCH_THRESH,
                                pad: int = 8):
  
    score_t = cv2.imread(str(score_path), cv2.IMREAD_GRAYSCALE)
    x_t = cv2.imread(str(x_path), cv2.IMREAD_GRAYSCALE)
    if score_t is None or x_t is None:
        return None

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    s1, loc1, wh1 = match_best(gray, score_t)
    s2, loc2, wh2 = match_best(gray, x_t)

    if loc1 is None or loc2 is None:
        return None
    if s1 < thresh or s2 < thresh:
        return None

    x_score, y_score = loc1
    w_score, h_score = wh1

    x_btn, y_btn = loc2
    w_btn, h_btn = wh2

    left = x_score
    right = x_btn + w_btn
    width = right - left
    if width <= 0:
        return None

    height = int(width*0.75)
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
                    found = detect_roi_from_score_and_x(frame, SCORE_TEMPLATE, X_TEMPLATE)
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
                to_save = frame[y:y+h, x:x+w]

            if now - last_save >= SAVE_EVERY_SEC:
                out_path = SAVE_DIR / make_filename(now)
                cv2.imwrite(str(out_path), to_save)
                print("saved:", out_path)
                last_save = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C).")
