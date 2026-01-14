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
        scale=1.0,
    ):
        self.scale = scale
        self.debug = debug

        self.min_area = int(min_area * scale * scale)
        self.adaptive_block = max(11, int(adaptive_block * scale) | 1)
        self.adaptive_c = adaptive_c
        self.fullscreen_ratio = fullscreen_ratio
        self.title_threshold = title_threshold

        self.vertical_kernel = cv.getStructuringElement(
            cv.MORPH_RECT,
            (max(3, int(9 * scale)), 1)
        )
        self.cleanup_kernel = cv.getStructuringElement(
            cv.MORPH_RECT,
            (max(3, int(3 * scale)), max(3, int(3 * scale)))
        )

        self.matcher = TemplateMatcher(
            template_path=template_path,
            threshold=title_threshold,
            debug=debug,
        )

    # --------------------------------------------------

    def find_anchor_y(self, img):
        ok, score, scale, loc = self.matcher.match(img)
        if not ok or loc is None:
            return None

        _, tmpl = self.matcher.templates[0]
        h = int(tmpl.shape[0] * scale)
        return loc[1] + h

    # --------------------------------------------------

    def _detect_windows(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        img_area = h * w

        sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, 3)
        sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, 3)
        mag = np.uint8(np.clip(cv.magnitude(sobel_x, sobel_y), 0, 255))

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
        edges = cv.morphologyEx(edges, cv.MORPH_OPEN, self.cleanup_kernel, iterations=3)

        contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        windows = []
        fullscreen = None

        for cnt in contours:
            approx = cv.approxPolyDP(cnt, 0.02 * cv.arcLength(cnt, True), True)
            if not cv.isContourConvex(approx):
                continue

            x, y, ww, hh = cv.boundingRect(approx)
            area_ratio = (ww * hh) / img_area

            if area_ratio >= self.fullscreen_ratio:
                fullscreen = approx
                continue

            if cv.contourArea(cnt) >= self.min_area and len(approx) == 4:
                windows.append(approx)

        return ([fullscreen] if fullscreen is not None else windows), edges

    # --------------------------------------------------

    def detect(self, img):
        windows, edges = self._detect_windows(img)
        anchor_y = self.find_anchor_y(img)

        best = None

        if anchor_y is not None and windows:
            best = min(
                windows,
                key=lambda r: abs(cv.boundingRect(r)[1] - anchor_y),
            )
        elif windows:
            best = max(
                windows,
                key=lambda r: cv.boundingRect(r)[2] * cv.boundingRect(r)[3],
            )

        if best is None:
            h, w = img.shape[:2]
            best = np.array([
                [[0, 0]],
                [[w - 1, 0]],
                [[w - 1, h - 1]],
                [[0, h - 1]],
            ])

        # --- SCALE back and convert to x, y, w, h tuple ---
        best_scaled = (best / self.scale).astype(np.int32)
        x, y, w, h = cv.boundingRect(best_scaled)

        return (x, y, w, h), edges

    # --------------------------------------------------

    @staticmethod
    def draw(img, rect, color=(0, 255, 0), thickness=3):
        out = img.copy()
        if rect is not None:
            cv.drawContours(out, [rect], -1, color, thickness)
        return out