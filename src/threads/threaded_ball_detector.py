import threading

class ThreadedBallDetector(threading.Thread):
    def __init__(self, detector, frame_shared, rect_shared, result_shared):
        super().__init__(daemon=True)
        self.detector = detector
        self.frame_shared = frame_shared
        self.rect_shared = rect_shared
        self.result_shared = result_shared
        self.running = True

    def run(self):
        while self.running:
            frame, _ = self.frame_shared.get_latest_with_id()
            rect, _ = self.rect_shared.get_latest_with_id()

            if frame is None or rect is None:
                continue

            x, y, w, h = rect
            crop = frame[y:y+h, x:x+w].copy()
            detections = self.detector.detect(crop)

            # Map detections back to full frame coordinates
            for d in detections:
                d["x"] += x
                d["y"] += y

            self.result_shared.set(detections)

    def stop(self):
        self.running = False