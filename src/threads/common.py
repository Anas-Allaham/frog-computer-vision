import threading
import cv2 as cv
import numpy as np
import mss
import time

class SharedData:
    """Thread-safe container for any data (window rect, balls, etc.)"""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None

    def set(self, value):
        with self.lock:
            self.data = value

    def get(self):
        with self.lock:
            return self.data


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
        self.stopped = False
    def update_roi(self, new_monitor):
        self.monitor = new_monitor

    def stop(self):
        self.stopped = True

    def run(self):
        # Leave a 250px high safe zone at the top of the screen for the debug window
        safe_zone_height = 250
        capture_monitor = {
            "top": self.monitor["top"] + safe_zone_height,
            "left": self.monitor["left"],
            "width": self.monitor["width"],
            "height": self.monitor["height"] - safe_zone_height,
        }

        with mss.mss() as sct:
            while not self.stopped:
                screenshot = sct.grab(capture_monitor)
                img = np.array(screenshot)
                img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
                self.frame.set(img)
                time.sleep(0.01) # Avoid busy-waiting
