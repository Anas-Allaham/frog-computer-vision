import cv2 as cv
import numpy as np
from .template_matcher import TemplateMatcher

class ZumaWindowDetector:
    def __init__(
        self,
        template_path,
        min_area=640_000,
        adaptive_block=57,
        adaptive_c=-9,
        fullscreen_ratio=0.80,
        title_threshold=0.65,
        debug=False,
    ):
        self.min_area = min_area
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c
        self.fullscreen_ratio = fullscreen_ratio
        self.title_threshold = title_threshold
        self.debug = debug

        self.vertical_kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (9, 1)
        )
        self.cleanup_kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (3, 3)
        )

        # ---- Title matcher (helper only) ----
        self.matcher = TemplateMatcher(
            template_path=template_path,
            threshold=title_threshold,
            debug=debug,
        )

    # ============================================================
    # Helper: find title anchor Y
    # ============================================================
    def find_anchor_y(self, img):
        ok, score, scale, loc = self.matcher.match(img)
        if not ok or loc is None:
            return None

        _, tmpl = self.matcher.templates[0]
        h = int(tmpl.shape[0] * scale)
        x, y = loc

        anchor_y = y + h

        if self.debug:
            print(f"[ANCHOR] Found at y={anchor_y}, score={score:.3f}")

        return anchor_y

    # ============================================================
    # Window candidate detection
    # ============================================================
    def _detect_windows(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_area = h * w

        sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, 3)
        sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, 3)
        mag = cv.magnitude(sobel_x, sobel_y)
        mag = np.uint8(np.clip(mag, 0, 255))

        edges = cv.adaptiveThreshold(
            mag,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            self.adaptive_block,
            self.adaptive_c,
        )
        edges = cv.bitwise_not(edges)

        edges = cv.morphologyEx(edges, cv.MORPH_OPEN, self.vertical_kernel)
        edges = cv.morphologyEx(
            edges, cv.MORPH_OPEN, self.cleanup_kernel, iterations=3
        )

        contours, _ = cv.findContours(
            edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        windows = []
        fullscreen = None

        for cnt in contours:
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            if not cv.isContourConvex(approx):
                continue

            x, y, ww, hh = cv.boundingRect(approx)
            area_ratio = (ww * hh) / img_area

            if area_ratio >= self.fullscreen_ratio:
                fullscreen = approx
                continue

            if cv.contourArea(cnt) >= self.min_area and len(approx) == 4:
                windows.append(approx)

        if fullscreen is not None:
            return [fullscreen], edges

        return windows, edges

    # ============================================================
    # Public API
    # ============================================================
    def detect(self, img):
        if img is None:
            raise ValueError("Input image is None")

        windows, edges = self._detect_windows(img)

        if self.debug:
            print(f"[ZUMA] Candidate windows: {len(windows)}")

        # ------------------------------------------------
        # 1. Try title anchor (windowed mode)
        # ------------------------------------------------
        anchor_y = self.find_anchor_y(img)

        if anchor_y is not None and windows:
            best = None
            best_dist = float("inf")

            for rect in windows:
                x, y, w, h = cv.boundingRect(rect)

                if y >= anchor_y - 20:  # tolerance
                    dist = y - anchor_y
                    if dist < best_dist:
                        best_dist = dist
                        best = rect

            if best is not None:
                if self.debug:
                    print("[ZUMA] Selected window by title anchor")
                return best, edges

            if self.debug:
                print("[ZUMA] Anchor found but no window below it")

        # ------------------------------------------------
        # 2. FULLSCREEN / FALLBACK (always executed)
        # ------------------------------------------------
        if windows:
            best = max(
                windows,
                key=lambda r: cv.boundingRect(r)[2] * cv.boundingRect(r)[3]
            )

            if self.debug:
                x, y, w, h = cv.boundingRect(best)
                print(f"[ZUMA] Fallback to largest window ({w}x{h})")

            return best, edges

        # ------------------------------------------------
        # 3. Ultimate fallback: whole screen
        # ------------------------------------------------
        h, w = img.shape[:2]
        fullscreen = np.array([
            [[0, 0]],
            [[w - 1, 0]],
            [[w - 1, h - 1]],
            [[0, h - 1]],
        ])

        if self.debug:
            print("[ZUMA] Ultimate fallback: full image")

        return fullscreen, edges

    # ============================================================
    # Draw helper
    # ============================================================
    @staticmethod
    def draw(img, rect, color=(0, 255, 0), thickness=3):
        out = img.copy()
        if rect is not None:
            cv.drawContours(out, [rect], -1, color, thickness)
        return out