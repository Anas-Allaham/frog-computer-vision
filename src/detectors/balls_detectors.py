import cv2 as cv
import numpy as np
from scipy.spatial import distance


class ZumaBallDetector:
    def __init__(self, min_radius=15, max_radius=18, debug=False):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.debug = debug

        # --- LIBRARY SETUP: SCIPY ---
        # 1. Define your specific target colors in BGR format
        self.colors_bgr = {
            "red":    (50, 50, 255),    # Pure Red
            "green":  (50, 200, 50),    # Standard Green
            "blue":   (255, 100, 50),   # lighter Blue (prevents confusing with dark purple)
            "cyan":   (255, 255, 0),    # Cyan
            "yellow": (0, 255, 255),    # Yellow
            "orange": (0, 140, 255),    # Orange
            "purple": (200, 0, 150)     # Magenta/Purple (distinct from Blue)
        }

        # 2. Prepare the library data
        self.color_names = list(self.colors_bgr.keys())
        self.target_values = list(self.colors_bgr.values())

        # 3. Convert targets to "Lab" Color Space (Better for matching than HSV)
        target_array = np.array([self.target_values], dtype=np.uint8)
        self.targets_lab = cv.cvtColor(target_array, cv.COLOR_BGR2Lab)[0]
        # ----------------------------

    def _show(self, window_name, image):
        """Helper to display images only if debug is enabled."""
        if self.debug:
            cv.imshow(window_name, image)
            cv.waitKey(1)

    def preprocess(self, roi):
        # DEBUG: Show Raw Input
        self._show("1_Input", roi)

        roi_blurred = cv.GaussianBlur(roi, (7, 7), 0)

        # DEBUG: Show Blurred Image
        self._show("2_Blurred", roi_blurred)
        return roi_blurred

    def hsv_mask(self, roi):
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0, 100, 100), (179, 255, 255))

        # DEBUG: Show Raw Mask (Before cleanup)
        self._show("3_Mask_Raw", mask)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel,)
        mask = cv.dilate(mask, kernel)

        # DEBUG: Show Processed Mask (After cleanup)
        self._show("4_Mask_Processed", mask)
        return mask

    def hough_detect(self, gray):
        # DEBUG: Show the Gray image the detector actually sees
        self._show("5_Hough_Input_Gray", gray)

        circles = cv.HoughCircles(
            gray, cv.HOUGH_GRADIENT, dp=1.3, minDist=25,
            param1=80, param2=16,
            minRadius=self.min_radius, maxRadius=self.max_radius
        )
        if circles is None: return []

        parsed = [(int(x), int(y), int(r)) for x, y, r in circles[0]]

        # DEBUG: Show all raw candidates found by Hough (before filtering)
        if self.debug:
            temp = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
            for x, y, r in parsed:
                cv.circle(temp, (x, y), r, (0, 255, 255), 1)
            self._show("6_Hough_Candidates_Raw", temp)

        return parsed

    def classify_color(self, frame_lab, mask):
        """
        Uses scipy library to find the mathematically closest color.
        """
        mean_color = cv.mean(frame_lab, mask=mask)[:3]

        # Simple filter for dark/shadow objects
        if mean_color[0] < 40:
            return "unknown"

        # LIBRARY MAGIC: Calculate distance to all known colors
        dists = distance.cdist([mean_color], self.targets_lab, 'euclidean')

        # Find index of the smallest distance
        closest_index = np.argmin(dists)

        return self.color_names[closest_index]

    def detect(self, frame):
        roi_pre = self.preprocess(frame)
        mask = self.hsv_mask(roi_pre)

        gray = cv.cvtColor(roi_pre, cv.COLOR_BGR2GRAY)
        gray[mask == 0] = 0

        circles = self.hough_detect(gray)
        circles = [c for c in circles if mask[c[1], c[0]] == 255]

        # DEBUG: Show circles that survived filtering and will be classified
        if self.debug:
            debug_circles = frame.copy()
            for x, y, r in circles:
                cv.circle(debug_circles, (x, y), r, (255, 0, 255), 2)
            self._show("7_Final_Circles_To_Classify", debug_circles)

        # Convert whole frame to Lab for color checking
        frame_lab = cv.cvtColor(roi_pre, cv.COLOR_BGR2Lab)

        detections = []
        for cx, cy, r in circles:
            ball_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv.circle(ball_mask, (cx, cy), int(r * 0.6), 255, -1)

            color_name = self.classify_color(frame_lab, ball_mask)

            detections.append({
                "x": cx, "y": cy, "radius": r, "color": color_name
            })

        return detections

    @staticmethod
    def draw(frame, detections):
        out = frame.copy()
        for d in detections:
            cv.circle(out, (d["x"], d["y"]), d["radius"], (0, 255, 0), 2)
            cv.putText(out, d["color"], (d["x"] - 10, d["y"] - d["radius"] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return out
