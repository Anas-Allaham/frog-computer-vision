__all__ = [
    'show'
]

import cv2 as cv

WIN_SIZE = (1080, 720)


def show(name, img):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, *WIN_SIZE)
    cv.imshow(name, img)
