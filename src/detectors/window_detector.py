import cv2 as cv
import numpy as np


class WindowDetector:
    def __init__(
        self,
        min_area=640_000,
        adaptive_block=57,
        adaptive_c=-9,
    ):
        self.min_area = min_area
        self.adaptive_block = adaptive_block
        self.adaptive_c = adaptive_c

        self.vertical_kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (9, 1)
        )
        self.cleanup_kernel = cv.getStructuringElement(
            cv.MORPH_RECT, (3, 3)
        )

    def detect(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Sobel gradients
        sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

        mag = cv.magnitude(sobel_x, sobel_y)
        mag = np.uint8(np.clip(mag, 0, 255))

        # Adaptive threshold
        edges = cv.adaptiveThreshold(
            mag,
            255,
            cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,
            self.adaptive_block,
            self.adaptive_c,
        )
        edges = cv.bitwise_not(edges)

        # Remove vertical noise
        edges = cv.morphologyEx(
            edges, cv.MORPH_OPEN, self.vertical_kernel
        )

        # Cleanup
        edges = cv.morphologyEx(
            edges, cv.MORPH_OPEN, self.cleanup_kernel, iterations=3
        )

        # Contours
        contours, _ = cv.findContours(
            edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        rectangles = []

        largest_area = 0
        largest_idx = -1
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_idx = i

        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)
            if area < self.min_area or i == largest_idx:
                continue

            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4 and cv.isContourConvex(approx):
                rectangles.append(approx)

        return rectangles, edges

    @staticmethod
    def draw(img, rectangles, color=(0, 255, 0), thickness=2):
        out = img.copy()
        for rect in rectangles:
            cv.drawContours(out, [rect], -1, color, thickness)
        return out
