import cv2 as cv
import numpy as np
from scipy.spatial import distance


class ZumaBallDetector:
    def __init__(self, min_radius=15, max_radius=18, debug=False, scale=1.0):
        self.debug = debug
        self.scale = scale

        # ---- scale-aware params ----
        self.min_radius = int(min_radius * scale)
        self.max_radius = int(max_radius * scale)
        self.min_dist = int(25 * scale)

        # --- COLOR DEFINITIONS ---
        self.colors_bgr = {
            "red":    (50, 50, 255),
            "green":  (50, 200, 50),
            "blue":   (255, 100, 50),
            "cyan":   (255, 255, 0),
            "yellow": (0, 255, 255),
            "orange": (0, 140, 255),
            "purple": (200, 0, 150),
        }
        self.color_names = list(self.colors_bgr.keys())
        self.target_values = list(self.colors_bgr.values())

        target_array = np.array([self.target_values], dtype=np.uint8)
        self.targets_lab = cv.cvtColor(target_array, cv.COLOR_BGR2Lab)[0]

    # --------------------------------------------------

    def _show(self, name, img):
        if self.debug:
            cv.imshow(name, img)
            cv.waitKey(1)

    # --------------------------------------------------

    def preprocess(self, roi):
        k = max(3, int(7 * self.scale) | 1)
        roi_blurred = cv.GaussianBlur(roi, (k, k), 0)
        self._show("1_Blurred", roi_blurred)
        return roi_blurred

    # --------------------------------------------------

    def hsv_mask(self, roi):
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0, 100, 100), (179, 255, 255))

        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (max(3, int(3 * self.scale)),) * 2
        )
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.dilate(mask, kernel)

        self._show("Mask", mask)
        return mask

    # --------------------------------------------------

    def hough_detect(self, gray):
        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp=1.3,
            minDist=self.min_dist,
            param1=80,
            param2=16,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is None:
            return []

        return [(int(x), int(y), int(r)) for x, y, r in circles[0]]

    # --------------------------------------------------

    def classify_color(self, frame_lab, mask):
        mean = cv.mean(frame_lab, mask=mask)[:3]
        if mean[0] < 40:
            return "unknown"

        dists = distance.cdist([mean], self.targets_lab, "euclidean")
        return self.color_names[np.argmin(dists)]

    # --------------------------------------------------

    def detect(self, frame):
        roi = self.preprocess(frame)
        mask = self.hsv_mask(roi)

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray[mask == 0] = 0

        circles = self.hough_detect(gray)
        circles = [c for c in circles if mask[c[1], c[0]] == 255]

        frame_lab = cv.cvtColor(roi, cv.COLOR_BGR2Lab)

        inv = 1.0 / self.scale
        detections = []

        for x, y, r in circles:
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv.circle(ball_mask, (x, y), int(r * 0.6), 255, -1)

            color = self.classify_color(frame_lab, ball_mask)

            detections.append({
                "x": int(x * inv),
                "y": int(y * inv),
                "radius": int(r * inv),
                "color": color,
            })

        return detections

    # --------------------------------------------------

    @staticmethod
    def draw(frame, detections):
        out = frame.copy()
        for d in detections:
            cv.circle(out, (d["x"], d["y"]), d["radius"], (0, 255, 0), 2)
            cv.putText(
                out,
                d["color"],
                (d["x"] - 10, d["y"] - d["radius"] - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        return out
