import threading

class ThreadedBallDetector(threading.Thread):
    def __init__(self, detector, frame_shared, rect_shared, result_shared, frog_center_shared=None):
        super().__init__(daemon=True)
        self.detector = detector
        self.frame_shared = frame_shared
        self.rect_shared = rect_shared
        self.result_shared = result_shared
        self.frog_center_shared = frog_center_shared
        self.running = True

        # Frame skipping parameters
        self.frame_counter = 0
        self.skip_frames = 2  # Process every 3rd frame (skip 2)
        self.last_processed_frame = None
        self.last_detections = []

    def stop(self):
        self.stopped = True

    def run(self):
        while self.running:
            frame, frame_id = self.frame_shared.get_latest_with_id()
            rect, _ = self.rect_shared.get_latest_with_id()

            if frame is None or rect is None:
                continue

            # Frame skipping logic
            self.frame_counter += 1
            should_process = (self.frame_counter % (self.skip_frames + 1) == 0)

            # Always process if this is the first frame or if we don't have previous detections
            if self.last_processed_frame is None or not self.last_detections:
                should_process = True

            if not should_process:
                # Use cached detections but check if balls have moved significantly
                # This is a simplified check - in practice you might want more sophisticated motion detection
                self.result_shared.set(self.last_detections)
                continue

            x, y, w, h = rect
            crop = frame[y:y+h, x:x+w].copy()

            # Get frog center if available for ROI optimization
            frog_center = None
            if self.frog_center_shared is not None:
                frog_data, _ = self.frog_center_shared.get_latest_with_id()
                if frog_data is not None:
                    frog_center = frog_data[0]  # frog_center is first element

            # Use advanced ball detection with ROI optimization
            if frog_center is not None:
                # Adjust frog center relative to crop coordinates
                adjusted_frog_center = (frog_center[0] - x, frog_center[1] - y)
                ball_roi = self.detector.estimate_ball_path_roi(crop.shape, adjusted_frog_center)
                detections = self.detector.detect(crop, ball_roi, use_advanced=True)
            else:
                # Fallback to full frame advanced detection
                detections = self.detector.detect(crop, use_advanced=True)

            # Map detections back to full frame coordinates
            for d in detections:
                d["x"] += x
                d["y"] += y

            # Update cache
            self.last_processed_frame = frame_id
            self.last_detections = detections
            self.result_shared.set(detections)

    def stop(self):
        self.running = False