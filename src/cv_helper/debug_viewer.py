import cv2 as cv
import numpy as np

class DebugViewer:
    def __init__(self, window_name="Debug Panel"):
        self.window_name = window_name
        self.data = {}
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)

    def add_data(self, name, content):
        """Accepts either an image (numpy array) or text (string)."""
        self.data[name] = content

    def show(self):
        if not self.data:
            return

        # --- Prevent Recursion by moving to the safe zone ---
        cv.moveWindow(self.window_name, 0, 0)

        images = {k: v for k, v in self.data.items() if isinstance(v, np.ndarray)}
        texts = {k: v for k, v in self.data.items() if isinstance(v, str)}

        if not images:
            # If there are no images, create a black canvas to display text
            display_img = np.zeros((200, 400, 3), dtype=np.uint8)
        else:
            max_w = max(img.shape[1] for img in images.values())
            processed_images = []
            for name, img in images.items():
                if len(img.shape) == 2:
                    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                h, w = img.shape[:2]
                scale = max_w / w
                resized_img = cv.resize(img, (max_w, int(h * scale)))
                cv.putText(resized_img, name, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                processed_images.append(resized_img)
            display_img = np.vstack(processed_images)

        # --- Render Text Data ---
        if texts:
            y_offset = 30
            for name, text in texts.items():
                cv.putText(display_img, f"{name}: {text}", (10, y_offset), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30

        cv.imshow(self.window_name, display_img)
        self.data = {}      # Clear images for the next frame
