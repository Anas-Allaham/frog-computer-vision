import threading

class ThreadedWindowDetector(threading.Thread):
    def __init__(self, detector, frame_shared, rect_shared):
        super().__init__(daemon=True)
        self.detector = detector
        self.frame_shared = frame_shared
        self.rect_shared = rect_shared
        self.running = True

    def run(self):
        while self.running:
            frame, _ = self.frame_shared.get_latest_with_id()
            if frame is None:
                continue

            rect, _ = self.detector.detect(frame)
            if rect is not None:
                x, y, w, h = rect
                self.rect_shared.set((x, y, w, h))

    def stop(self):
        self.running = False
