import threading
import cv2 as cv
import numpy as np
import mss

# =======================
# THREAD-SAFE CONTAINER
# =======================

class SharedFrame:
    """
    Thread-safe container for a frame.
    Each frame has a unique ID to prevent stale reads.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0

    def set(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def get_latest_with_id(self):
        with self.lock:
            return self.frame, self.frame_id
# =======================
# SCREEN CAPTURE THREAD
# =======================

class CaptureThread(threading.Thread):
    """
    Continuously capture the screen in a separate thread.
    """
    def __init__(self, monitor):
        super().__init__(daemon=True)
        self.monitor = monitor
        self.frame = SharedFrame()
        self.running = True

    def update_roi(self, new_monitor):
        self.monitor = new_monitor

    def run(self):
        with mss.mss() as sct:
            while self.running:
                img = np.array(sct.grab(self.monitor))
                frame = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
                self.frame.set(frame)

    def stop(self):
        self.running = False
