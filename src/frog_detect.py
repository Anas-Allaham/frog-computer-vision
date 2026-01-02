from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


__all__ = ["load_latest_capture", "detect_frog_center"]


def load_latest_capture(captures_dir: str = "captures", pattern: str = "*.png") -> Tuple[np.ndarray, Path]:
    d = Path(captures_dir)
    files = list(d.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {d.resolve()}")

    latest = max(files, key=lambda p: p.stat().st_mtime)
    img = cv2.imread(str(latest))
    if img is None:
        raise ValueError(f"Failed to read image: {latest}")
    return img, latest


def _rotate_bound(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image without cropping (rotate_bound style).
    """
    (h, w) = gray.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(gray, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _match_best_multiscale_and_rotate(
    image_gray: np.ndarray,
    templ_gray: np.ndarray,
    scales=(0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
    angle_step_deg: int = 10,  # ✅ 10 degrees
    method=cv2.TM_CCOEFF_NORMED,
) -> Tuple[float, Optional[Tuple[int, int]], Optional[Tuple[int, int]], Optional[float], Optional[float]]:
    """
    Returns:
      best_score,
      best_top_left (x,y) or None,
      best_wh (w,h) or None,
      best_angle_deg or None,
      best_scale or None
    """
    best_score = -1.0
    best_loc = None
    best_wh = None
    best_angle = None
    best_scale = None

    th0, tw0 = templ_gray.shape[:2]

    for angle in range(0, 360, angle_step_deg):  # ✅ 36 rotations
        rot = _rotate_bound(templ_gray, float(angle))

        for s in scales:
            tw, th = int(rot.shape[1] * s), int(rot.shape[0] * s)
            if tw < 8 or th < 8:
                continue

            t = cv2.resize(rot, (tw, th), interpolation=cv2.INTER_AREA)

            if th >= image_gray.shape[0] or tw >= image_gray.shape[1]:
                continue

            res = cv2.matchTemplate(image_gray, t, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            score = float(max_val)
            if score > best_score:
                best_score = score
                best_loc = max_loc
                best_wh = (tw, th)
                best_angle = float(angle)
                best_scale = float(s)

    return best_score, best_loc, best_wh, best_angle, best_scale


def detect_frog_center(
    frame_bgr: np.ndarray,
    frog_template_bgr: np.ndarray,
    *,
    threshold: float = 0.65,
    scales=(0.6,0.80, 0.9, 0.95, 1.0,1.05),
    angle_step_deg: int = 10,        # ✅ new parameter
    draw_debug: bool = True,
) -> Tuple[Optional[Tuple[int, int]], np.ndarray, float]:
    """
    Returns:
      - center (cx, cy) or None
      - debug_frame (with painted center if found)
      - best match score
    """
    img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    templ_gray = cv2.cvtColor(frog_template_bgr, cv2.COLOR_BGR2GRAY)

    score, loc, wh, best_angle, best_scale = _match_best_multiscale_and_rotate(
        img_gray,
        templ_gray,
        scales=scales,
        angle_step_deg=angle_step_deg,
        method=cv2.TM_CCOEFF_NORMED,
    )

    debug = frame_bgr.copy()

    if loc is None or wh is None or score < threshold:
        return None, debug, score

    x, y = loc
    w, h = wh
    cx = x + w // 2
    cy = y + h // 2

    if draw_debug:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(debug, (cx, cy), 6, (0, 0, 255), -1)
        cv2.circle(debug, (cx, cy), 14, (255, 255, 255), 2)

        # optional text info (angle/scale)
        if best_angle is not None and best_scale is not None:
            txt = f"score={score:.3f} ang={best_angle:.0f} sc={best_scale:.2f}"
            cv2.putText(debug, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return (cx, cy), debug, score


if __name__ == "__main__":
    frame, path = load_latest_capture("captures", "*.png")
    print("Latest capture:", path.name)

    frog_templ = cv2.imread("templates/frog.png")
    if frog_templ is None:
        raise FileNotFoundError("Missing template: templates/frog.png")

    center, dbg, score = detect_frog_center(
        frame,
        frog_templ,
        threshold=0.65,
        angle_step_deg=10
    )
    print("Score:", score, "Center:", center)

    out_dir = Path("captures_debug")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{path.stem}_frog_debug.png"
    cv2.imwrite(str(out_path), dbg)
    print("Saved debug:", out_path)
