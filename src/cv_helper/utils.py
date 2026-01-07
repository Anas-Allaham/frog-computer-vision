import cv2 as cv

def to_gray(img):
    if img.ndim == 3:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def preprocess_for_template(img, blur=True):
    gray = to_gray(img)
    if blur:
        gray = cv.GaussianBlur(gray, (3, 3), 0)
    return gray

def extract_title_bar(img, ratio=0.15):
    h = img.shape[0]
    return img[:int(h * ratio), :]
