import cv2 as cv
import numpy as np
from scipy.spatial import distance

TARGET_WIDTH = 640

# Calibrated HSV ranges from helping codes (much more accurate than LAB)
BALL_COLORS_HSV = {
    "purple": {"lower": np.array([131, 64, 55]), "upper": np.array([179, 203, 255])},
    "blue": {"lower": np.array([91, 109, 51]), "upper": np.array([118, 211, 255])},
    "green": {"lower": np.array([39, 130, 40]), "upper": np.array([82, 219, 255])},
    "yellow": {"lower": np.array([24, 133, 60]), "upper": np.array([30, 255, 255])},
    "orange": {
        "lower": np.array([7, 123, 117]),
        "upper": np.array([22, 237, 255])
    },
}


class ZumaBallDetector:
    def __init__(self, min_radius=15, max_radius=18, debug=False, scale=1.0, viewer=None):
        self.debug = debug
        self.viewer = viewer
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

        # --- ADVANCED DETECTION PARAMETERS ---
        self.target_width = TARGET_WIDTH

    # --------------------------------------------------

    def _show(self, name, img):
        if self.debug and self.viewer:
            self.viewer.add_data(name, img)

    # --------------------------------------------------

    def preprocess(self, roi):
        # Use smaller, faster blur kernel
        k = max(3, int(3 * self.scale) | 1)  # Reduced from 7 to 3
        roi_blurred = cv.GaussianBlur(roi, (k, k), 0)
        self._show("1_Blurred", roi_blurred)
        return roi_blurred

    # --------------------------------------------------

    def hsv_mask(self, roi):
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, (0, 70, 80), (179, 255, 255))  # Relaxed saturation/value thresholds

        # Simplified morphological operations - single close operation instead of close + dilate
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (max(3, int(2 * self.scale)),) * 2  # Smaller kernel
        )
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        self._show("Mask", mask)
        return mask

    # --------------------------------------------------

    def hough_detect(self, gray):
        # Optimized Hough parameters for better performance
        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            dp=1.2,  # Slightly reduced for speed
            minDist=self.min_dist,
            param1=60,  # Lower Canny threshold for more edges but faster
            param2=12,  # Lower accumulator threshold for more detections but faster
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is None:
            return []

        return [(int(x), int(y), int(r)) for x, y, r in circles[0]]

    # --------------------------------------------------

    def detect_balls_advanced(self, frame_bgr, debug=False):
        """
        Ultra-fast ball detection using optimized distance transform and color grouping.
        Optimized for 60 FPS real-time performance.
        """
        if frame_bgr is None:
            return []

        height, width = frame_bgr.shape[:2]
        scale = 1.0
        processing_frame = frame_bgr

        # Auto-scale large images for performance
        if width > self.target_width:
            scale = self.target_width / width
            processing_frame = cv.resize(frame_bgr, (0, 0), fx=scale, fy=scale)

        # Convert to HSV and apply optimized blur
        hsv = cv.cvtColor(processing_frame, cv.COLOR_BGR2HSV)
        hsv = cv.GaussianBlur(hsv, (3, 3), 0)  # Smaller kernel for speed

        detected_balls = []
        
        # Pre-allocate masks array for batch processing
        color_masks = {}
        
        # Process each color with optimized operations
        for color_name, ranges in BALL_COLORS_HSV.items():
            # Create color mask
            mask = cv.inRange(hsv, ranges["lower"], ranges["upper"])
            
            # Optimized morphological operations
            kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
            mask = cv.dilate(mask, kernel, iterations=1)
            
            # Store for batch processing
            color_masks[color_name] = mask

        # Batch process all colors together for efficiency
        for color_name, mask in color_masks.items():
            # Skip empty masks early
            if cv.countNonZero(mask) < 20:  # Minimum pixels for a ball
                continue

            # Optimized distance transform
            dist_transform = cv.distanceTransform(mask, cv.DIST_L2, 3)  # Smaller mask size
            
            # Adaptive threshold based on mask content
            max_dist = dist_transform.max()
            if max_dist < 3:  # Too small, skip
                continue
                
            threshold_val = max(0.5 * max_dist, 3.0)
            _, peaks = cv.threshold(dist_transform, threshold_val, 255, 0)
            peaks = np.uint8(peaks)

            # Find contours efficiently
            contours, _ = cv.findContours(peaks, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv.contourArea(cnt)
                if area < 5:  # Smaller minimum for faster processing
                    continue

                # Get center using minEnclosingCircle (faster than moments)
                (x_small, y_small), _ = cv.minEnclosingCircle(cnt)
                center_x_small, center_y_small = int(x_small), int(y_small)

                # Get radius from distance transform with bounds checking
                if (0 <= center_y_small < dist_transform.shape[0] and 
                    0 <= center_x_small < dist_transform.shape[1]):
                    real_radius_small = dist_transform[center_y_small, center_x_small]
                else:
                    continue

                # Optimized radius filtering
                if real_radius_small < 3 or real_radius_small > 25:
                    continue

                # Scale back to original coordinates
                final_x = int(center_x_small / scale)
                final_y = int(center_y_small / scale)
                final_radius = int(real_radius_small / scale)

                detected_balls.append({
                    "color": color_name,
                    "center": (final_x, final_y),
                    "radius": final_radius,
                    "type": "advanced"
                })

        return detected_balls

    # --------------------------------------------------

    def classify_color(self, frame_lab, mask):
        mean = cv.mean(frame_lab, mask=mask)[:3]
        if mean[0] < 40:
            return "unknown"

        # Use faster numpy operations instead of scipy
        mean_array = np.array(mean)
        targets_array = np.array(self.targets_lab)
        dists = np.sqrt(np.sum((targets_array - mean_array) ** 2, axis=1))
        return self.color_names[np.argmin(dists)]

    # --------------------------------------------------

    def classify_ball_by_color(self, frame, center, radius):
        """Classify a ball by its color.
        
        Args:
            frame: The input image
            center: Tuple of (x, y) coordinates of the ball center
            radius: Radius of the ball
            
        Returns:
            dict: Dictionary containing ball properties including 'x', 'y', 'radius', and 'color'
        """
        x, y = map(int, center)
        radius = int(radius)
        
        # Create a mask for the ball
        mask = np.zeros_like(frame[:, :, 0])
        cv.circle(mask, (x, y), radius, 255, -1)
        
        # Convert to LAB color space for better color difference measurement
        frame_lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        
        # Classify the color
        color = self.classify_color(frame_lab, mask)
        
        return {
            'x': x,
            'y': y,
            'radius': radius,
            'color': color
        }
        
    def detect(self, frame, roi_rect=None, use_advanced=True):
        """
        Detect balls in the frame using either advanced or traditional method.

        Args:
            frame: Input frame
            roi_rect: Optional tuple (x, y, w, h) defining region of interest
            use_advanced: Whether to use advanced distance-transform detection (recommended)
        """
        if use_advanced:
            # Use the advanced distance-transform method
            balls = self.detect_balls_advanced(frame, debug=self.debug)

            # Convert to the expected format for compatibility
            detections = []
            for ball in balls:
                center_x, center_y = ball["center"]
                detections.append({
                    "x": center_x,
                    "y": center_y,
                    "radius": ball["radius"],
                    "color": ball["color"],
                })
            return detections

        else:
            # Use traditional Hough circle method
            return self.detect_traditional(frame, roi_rect)

    def detect_traditional(self, frame, roi_rect=None):
        """
        Traditional Hough circle detection method (kept for compatibility).
        """
        # If ROI is specified, crop to that region
        if roi_rect is not None:
            x, y, w, h = roi_rect
            roi_frame = frame[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            roi_frame = frame
            offset_x, offset_y = 0, 0

        roi = self.preprocess(roi_frame)
        mask = self.hsv_mask(roi)

        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        gray[mask == 0] = 0

        circles = self.hough_detect(gray)
        h, w = mask.shape
        circles = [c for c in circles if 0 <= c[1] < h and 0 <= c[0] < w and mask[c[1], c[0]] == 255]

        frame_lab = cv.cvtColor(roi, cv.COLOR_BGR2Lab)

        inv = 1.0 / self.scale
        detections = []

        # Batch process color classification for better performance
        if circles:
            # Pre-compute LAB conversion for the entire ROI
            ball_colors = []
            for cx, cy, r in circles:
                ball_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                cv.circle(ball_mask, (cx, cy), int(r * 0.6), 255, -1)

                mean = cv.mean(frame_lab, mask=ball_mask)[:3]
                ball_colors.append(mean)

            # Batch classify colors
            ball_colors_array = np.array(ball_colors)
            targets_array = np.array(self.targets_lab)

            # Vectorized distance calculation
            dists = np.sqrt(np.sum((targets_array[np.newaxis, :, :] - ball_colors_array[:, np.newaxis, :]) ** 2, axis=2))
            color_indices = np.argmin(dists, axis=1)

            for i, (cx, cy, r) in enumerate(circles):
                color_name = self.color_names[color_indices[i]] if ball_colors[i][0] >= 40 else "unknown"

                detections.append({
                    "x": int((cx + offset_x) * inv),
                    "y": int((cy + offset_y) * inv),
                    "radius": int(r * inv),
                    "color": color_name,
                })

        return detections

    def estimate_ball_path_roi(self, frame_shape, frog_center=None):
        """
        Estimate the region where balls are likely to be based on typical Zuma layout.
        Returns a rectangle (x, y, w, h) defining the ROI.
        """
        h, w = frame_shape[:2]

        # Default ROI covers most of the frame but avoids UI elements
        # In Zuma, balls typically appear in the central curved path
        margin_x = int(w * 0.1)  # 10% margin on sides
        margin_y = int(h * 0.15)  # 15% margin on top/bottom

        roi_x = margin_x
        roi_y = margin_y
        roi_w = w - 2 * margin_x
        roi_h = h - 2 * margin_y

        # If frog center is known, adjust ROI to focus more on the path
        if frog_center is not None:
            frog_x, frog_y = frog_center
            # Extend ROI towards the direction balls come from (typically from right side)
            roi_x = max(0, frog_x - int(w * 0.4))
            roi_w = min(w - roi_x, int(w * 0.7))

        return (roi_x, roi_y, roi_w, roi_h)

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
