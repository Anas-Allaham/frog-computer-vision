import cv2 as cv
import numpy as np
from .balls_detectors import ZumaBallDetector
from cv_helper.utils import crop_image

class ZumaFrogDetector:
    """
    Detects the frog, its angle, and the balls it holds using fast contour-based methods.
    """
    def __init__(self, template_path, debug=False, viewer=None):
        self.debug = debug
        self.viewer = viewer

        # Load template for fallback/template matching
        self.template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(f"Could not read template: {template_path}")

        # Pre-compute template properties for contour-based detection
        self.template_h, self.template_w = self.template.shape[:2]
        self.frog_aspect_ratio = self.template_w / self.template_h

        # Color ranges for frog detection (typically green/brown)
        self.frog_color_lower = np.array([30, 50, 50])   # Green-brown lower bound
        self.frog_color_upper = np.array([80, 255, 200]) # Green-brown upper bound

        self.ball_detector = ZumaBallDetector(debug=debug, viewer=self.viewer)

    def _show(self, name, img):
        if self.debug and self.viewer:
            self.viewer.add_data(name, img)

    def detect(self, frame):
        """
        Detects the frog, its angle, the current ball (half-circle), and the next ball (full-circle).
        Uses fast contour-based detection with template matching fallback.
        """
        # Try fast contour-based detection first
        frog_center, frog_angle, contour = self._detect_frog_contours(frame)

        # Fallback to template matching if contour detection fails
        if frog_center is None:
            frog_center, frog_angle, contour = self._detect_frog_template(frame)
            if frog_center is None:
                return None, None, None, None, None

        # Initialize current_ball to None
        current_ball = None
        next_ball = None
        
        if frog_center is not None:
            # --- Detect Both Balls in Single Pipeline ---
            current_ball, next_ball = self._detect_frog_balls_optimized(frame, frog_center, frog_angle)
        
        return frog_center, frog_angle, current_ball, next_ball, contour

    def _detect_frog_balls_optimized(self, frame, frog_center, frog_angle):
        """
        Optimized method to detect both current and next balls in the frog area.
        """
        h, w = frame.shape[:2]

        # Define search region around frog (larger area to catch both balls)
        search_radius = 80
        x1 = max(0, frog_center[0] - search_radius)
        y1 = max(0, frog_center[1] - search_radius)
        x2 = min(w, frog_center[0] + search_radius)
        y2 = min(h, frog_center[1] + search_radius)

        frog_roi = frame[y1:y2, x1:x2]
        if frog_roi.size == 0:
            return None, None

        # Convert to grayscale and apply preprocessing once
        gray = cv.cvtColor(frog_roi, cv.COLOR_BGR2GRAY)
        blurred = cv.medianBlur(gray, 3)  # Lighter blur for speed

        # Single Hough circle detection for both balls
        circles = cv.HoughCircles(
            blurred, cv.HOUGH_GRADIENT, 1, 15,  # minDist=15 to separate balls
            param1=40, param2=20,  # Lower thresholds for speed
            minRadius=12, maxRadius=30
        )

        detected_balls = []
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))

            # Convert to LAB for color classification
            roi_lab = cv.cvtColor(frog_roi, cv.COLOR_BGR2LAB)

            for circle in circles:
                cx, cy, radius = circle

                # Create mask for color classification
                mask = np.zeros(frog_roi.shape[:2], dtype=np.uint8)
                cv.circle(mask, (cx, cy), int(radius * 0.7), 255, -1)

                # Classify color
                ball = self.ball_detector.classify_ball_by_color(
                    frog_roi, (cx, cy), radius
                )

                if ball:
                    # Adjust coordinates back to full frame
                    ball['x'] += x1
                    ball['y'] += y1
                    detected_balls.append(ball)

        # Classify balls by position relative to frog
        current_ball = None
        next_ball = None

        for ball in detected_balls:
            # Calculate vector from frog center to ball center
            dx = ball['x'] - frog_center[0]
            dy = ball['y'] - frog_center[1]
            distance = np.sqrt(dx*dx + dy*dy)

            # Calculate angle relative to frog's facing direction
            ball_angle = np.arctan2(dy, dx)
            angle_diff = abs(ball_angle - frog_angle)
            angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Handle angle wraparound

            # Current ball: closer to frog, in front direction
            # Next ball: further back, typically above/behind frog
            if distance < 40 and angle_diff < np.pi/3:  # Within 60 degrees of facing direction
                current_ball = ball
            elif distance > 25 and distance < 70:  # Further away but still in frog area
                next_ball = ball

        return current_ball, next_ball

    def _detect_frog_contours(self, frame):
        """
        Ultra-fast contour-based frog detection using optimized morphological operations.
        """
        # Convert to HSV for color-based segmentation
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Enhanced color ranges for better frog detection
        # Multiple ranges to handle lighting variations
        frog_masks = []
        
        # Green-brown range 1 (normal lighting)
        mask1 = cv.inRange(hsv, np.array([25, 40, 40]), np.array([85, 255, 200]))
        frog_masks.append(mask1)
        
        # Green-brown range 2 (darker lighting)
        mask2 = cv.inRange(hsv, np.array([20, 30, 30]), np.array([75, 200, 150]))
        frog_masks.append(mask2)
        
        # Combine masks
        mask = cv.bitwise_or(mask1, mask2)

        # Optimized morphological operations for speed
        # Use smaller kernels and fewer operations
        kernel_small = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        
        # Fast cleanup with minimal operations
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_small, iterations=1)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_small, iterations=1)
        
        # Remove small noise
        mask = cv.medianBlur(mask, 3)

        # Find contours with optimized parameters
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_center = None
        best_angle = 0
        best_score = 0

        # Pre-calculate frame dimensions for position scoring
        frame_h, frame_w = frame.shape[:2]
        frame_center_x = frame_w // 2
        frame_center_y = frame_h // 2

        for cnt in contours:
            area = cv.contourArea(cnt)
            # Optimized size filters based on typical frog dimensions
            if area < 800 or area > 8000:  # Tighter range for faster processing
                continue

            # Quick aspect ratio check using bounding rect
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.6 < aspect_ratio < 1.6):  # Tighter range
                continue

            # Fast circularity check (frogs are somewhat rectangular)
            perimeter = cv.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.75:  # Too circular
                continue

            # Calculate center using moments (faster than centroid)
            M = cv.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Optimized scoring system
            # Position score: prefer center-lower region (typical frog position)
            position_score = 1.0 - (abs(cx - frame_center_x) / frame_w + abs(cy - frame_center_y * 1.2) / frame_h) * 0.5
            
            # Size score: prefer medium-sized contours
            size_score = min(area / 3000, 1.0)
            
            # Shape score: prefer rectangular shapes
            shape_score = 1.0 - circularity

            # Combined score with optimized weights
            score = (position_score * 0.4 + size_score * 0.3 + shape_score * 0.3)

            if score > best_score:
                best_score = score
                best_center = (cx, cy)
                best_contour = cnt

                # Estimate angle from contour orientation (faster than PCA)
                # Use rotated rectangle to get angle
                rect = cv.minAreaRect(cnt)
                best_angle = rect[2] * np.pi / 180  # Convert to radians

        # Convert contour to format expected by drawing function
        if best_contour is not None:
            rect_pts = cv.boxPoints(cv.minAreaRect(best_contour))
            contour_pts = rect_pts.reshape(-1, 1, 2).astype(np.float32)
        else:
            contour_pts = None

        return best_center, best_angle, contour_pts

    def _detect_frog_template(self, frame):
        """
        Fallback template matching method (slower but more accurate).
        """
        # Use the old ORB-based method as fallback
        orb = cv.ORB_create(nfeatures=500)  # Reduced features for speed
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_kp, frame_des = orb.detectAndCompute(frame_gray, None)

        if frame_des is None or len(frame_des) < 10:
            return None, None, None

        template_kp, template_des = orb.detectAndCompute(self.template, None)
        if template_des is None:
            return None, None, None

        matches = bf.match(template_des, frame_des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:  # Reduced threshold for speed
            return None, None, None

        # Use homography for better pose estimation
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        if M is None:
            return None, None, None

        h, w = self.template.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        frog_center = np.mean(dst.reshape(4, 2), axis=0).astype(int)
        frog_center = (int(frog_center[0]), int(frog_center[1]))

        # Estimate angle from homography
        vec_x = (dst[3][0][0] + dst[2][0][0]) / 2 - (dst[0][0][0] + dst[1][0][0]) / 2
        vec_y = (dst[3][0][1] + dst[2][0][1]) / 2 - (dst[0][0][1] + dst[1][0][1]) / 2
        frog_angle = np.arctan2(vec_y, vec_x)

        return frog_center, frog_angle, dst


    def draw(self, display, center, angle, current_ball, next_ball, contour_pts):
        if center is not None:
            if contour_pts is not None:
                cv.drawContours(display, [np.int32(contour_pts)], -1, (255, 0, 0), 3)
            end_point = (int(center[0] + 50 * np.cos(angle)), int(center[1] + 50 * np.sin(angle)))
            cv.line(display, center, end_point, (0, 0, 255), 2)
            cv.circle(display, center, 5, (0, 0, 255), -1)

        if current_ball:
            cv.circle(display, (current_ball['x'], current_ball['y']), current_ball['radius'], (255, 255, 0), 2)
            cv.putText(display, f"C: {current_ball['color']}", (current_ball['x'], current_ball['y'] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if next_ball:
            cv.circle(display, (next_ball['x'], next_ball['y']), next_ball['radius'], (0, 255, 255), 2)
            cv.putText(display, f"N: {next_ball['color']}", (next_ball['x'], next_ball['y'] - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return display