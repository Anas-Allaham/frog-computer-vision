"""
Smart ROI Detection using template matching for score and X button.
Based on the helping codes implementation for automatic ROI detection.
"""

from __future__ import annotations

import time
import ctypes
from pathlib import Path

import numpy as np
import cv2
import mss
import win32gui
import win32con
import win32process
import psutil

from config import TEMPLATE_DIR


class WindowNotFound(Exception):
    pass


def _enum_windows():
    out = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if title:
            out.append((hwnd, title))

    win32gui.EnumWindows(cb, None)
    return out


def find_window_hwnd(title_contains: str | None = None,
                     process_name: str | None = None) -> int:
    wins = _enum_windows()

    if title_contains:
        t = title_contains.lower()
        wins = [(h, ttl) for (h, ttl) in wins if t in ttl.lower()]

    if process_name:
        p = process_name.lower()
        filtered = []
        for hwnd, ttl in wins:
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if psutil.Process(pid).name().lower() == p:
                    filtered.append((hwnd, ttl))
            except Exception:
                pass
        wins = filtered

    if not wins:
        raise WindowNotFound(
            f"No window found for title_contains={title_contains}, process_name={process_name}"
        )

    fg = win32gui.GetForegroundWindow()
    for hwnd, _ in wins:
        if hwnd == fg:
            return hwnd

    return wins[0][0]


def get_client_bbox(hwnd: int) -> dict:
    l, t, r, b = win32gui.GetClientRect(hwnd)
    tl = win32gui.ClientToScreen(hwnd, (l, t))
    br = win32gui.ClientToScreen(hwnd, (r, b))
    x1, y1 = tl
    x2, y2 = br
    return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}


def is_alive(hwnd: int) -> bool:
    return win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)


def focus_window(hwnd: int, click_fallback: bool = False) -> bool:
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    try:
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.05)
        if win32gui.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        pass

    try:
        fg = win32gui.GetForegroundWindow()
        fg_tid = win32process.GetWindowThreadProcessId(fg)[0]
        this_tid = win32process.GetWindowThreadProcessId(hwnd)[0]

        user32 = ctypes.windll.user32
        user32.AttachThreadInput(fg_tid, this_tid, True)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(fg_tid, this_tid, False)

        time.sleep(0.05)
        if win32gui.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        pass

    if click_fallback:
        try:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            cx, cy = (l + r) // 2, (t + b) // 2
            user32 = ctypes.windll.user32
            user32.SetCursorPos(cx, cy)
            user32.mouse_event(2, 0, 0, 0, 0)  # left down
            user32.mouse_event(4, 0, 0, 0, 0)  # left up
            time.sleep(0.05)
            return win32gui.GetForegroundWindow() == hwnd
        except Exception:
            pass

    return False


class SmartROIDetector:
    """
    Automatically detects game ROI using score and X button templates.
    Based on helping codes implementation.
    """

    def __init__(self, game_title="Zuma", template_dir=TEMPLATE_DIR):
        self.game_title = game_title
        self.template_dir = Path(template_dir)

        # Template paths
        self.score_template_path = self.template_dir / "score.png"
        self.x_template_path = self.template_dir / "x.png"

        # Multi-scale matching parameters
        self.scales = (0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15)
        self.match_thresh = 0.65

        # Window management
        self.hwnd = None
        self.roi_locked = False
        self.last_redetect = 0.0
        self.redeetect_every_sec = 10.0

        # Load templates
        self.score_template = None
        self.x_template = None
        self._load_templates()

    def _load_templates(self):
        """Load score and X button templates."""
        if self.score_template_path.exists():
            self.score_template = cv2.imread(str(self.score_template_path), cv2.IMREAD_GRAYSCALE)
        if self.x_template_path.exists():
            self.x_template = cv2.imread(str(self.x_template_path), cv2.IMREAD_GRAYSCALE)

    def find_game_window(self):
        """Find and focus the game window."""
        try:
            self.hwnd = find_window_hwnd(title_contains=self.game_title)
            focus_window(self.hwnd, click_fallback=False)
            return True
        except WindowNotFound:
            print(f"Game window '{self.game_title}' not found")
            return False

    def match_best(self, gray: np.ndarray, templ_gray: np.ndarray, scales=None,
                   method=cv2.TM_CCOEFF_NORMED):
        """
        Multi-scale template matching.
        Returns best score, location, and size.
        """
        if scales is None:
            scales = self.scales

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

    def detect_roi_from_templates(self, frame_bgr: np.ndarray, thresh: float = None, pad: int = 8):
        """
        Detect ROI using score and X button templates.
        Returns (x, y, w, h) ROI rectangle.
        """
        if thresh is None:
            thresh = self.match_thresh

        if self.score_template is None or self.x_template is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Match both templates
        s1, loc1, wh1 = self.match_best(gray, self.score_template)
        s2, loc2, wh2 = self.match_best(gray, self.x_template)

        if loc1 is None or loc2 is None:
            return None
        if s1 < thresh or s2 < thresh:
            return None

        # Extract positions
        x_score, y_score = loc1
        w_score, h_score = wh1
        x_btn, y_btn = loc2
        w_btn, h_btn = wh2

        # Calculate ROI boundaries
        left = x_score
        right = x_btn + w_btn
        width = right - left
        if width <= 0:
            return None

        height = int(width * 0.75)  # Assume 4:3 aspect ratio
        top = min(y_score, y_btn)
        bottom = top + height

        H, W = frame_bgr.shape[:2]

        # Apply padding and clamp to image boundaries
        left = max(0, left - pad)
        top = max(0, top - pad)
        right = min(W, right + pad)
        bottom = min(H, bottom + pad)

        w = max(1, right - left)
        h = max(1, bottom - top)

        scores = (s1, s2)
        return (left, top, w, h), scores

    def get_game_roi(self, frame_bgr: np.ndarray, force_redetect=False):
        """
        Get the current game ROI, redetecting if necessary.
        Returns (x, y, w, h) ROI rectangle.
        """
        now = time.time()

        # Check if we need to redetect
        need_redetect = (
            self.hwnd is None or
            force_redetect or
            not self.roi_locked or
            (self.redeetect_every_sec is not None and (now - self.last_redetect >= self.redeetect_every_sec))
        )

        if need_redetect:
            # Find game window if needed
            if self.hwnd is None and not self.find_game_window():
                return None

            # Detect ROI from templates
            found = self.detect_roi_from_templates(frame_bgr)
            if found is not None:
                roi, scores = found
                self.last_redetect = now
                if not self.roi_locked:
                    self.roi_locked = True
                    print(f"ROI locked: {roi}, scores: {scores}")
                return roi

        # Return cached ROI if available
        # For now, return a default ROI if templates fail
        if not hasattr(self, '_default_roi'):
            H, W = frame_bgr.shape[:2]
            margin_x = int(W * 0.1)
            margin_y = int(H * 0.15)
            self._default_roi = (margin_x, margin_y, W - 2*margin_x, H - 2*margin_y)

        return self._default_roi

    def capture_game_frame(self):
        """Capture a frame from the game window."""
        if self.hwnd is None:
            return None

        try:
            bbox = get_client_bbox(self.hwnd)
            with mss.mss() as sct:
                screenshot = sct.grab(bbox)
                frame = np.array(screenshot)[:, :, :3]  # BGR
                return frame
        except Exception as e:
            print(f"Failed to capture frame: {e}")
            return None