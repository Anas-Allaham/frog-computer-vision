import cv2 as cv

def to_color_space(img, flag=cv.COLOR_BGR2GRAY):
    if img.ndim == 3:
        return cv.cvtColor(img, flag)
    return img

def preprocess_for_template(img, blur=True):
    gray = to_color_space(img)
    if blur:
        gray = cv.GaussianBlur(gray, (3, 3), 0)
    return gray

def extract_title_bar(img, ratio=0.15):
    h = img.shape[0]
    return img[:int(h * ratio), :]


def hsv_to_lab(hsv):
    # Convert HSV → BGR → LAB (OpenCV does not support HSV→LAB directly)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return cv.cvtColor(bgr, cv.COLOR_BGR2LAB)


def apply_clahe_lab(lab, clip=2.0, grid=(8, 8)):
    l, a, b = cv.split(lab)

    clahe = cv.createCLAHE(
        clipLimit=clip,
        tileGridSize=grid,
    )

    l_clahe = clahe.apply(l)
    return cv.merge((l_clahe, a, b))
