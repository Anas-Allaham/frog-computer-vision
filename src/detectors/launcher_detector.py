import cv2 as cv
import numpy as np
from config import TEMPLATE_DIR, TEST_DIR

# ────────────────────────────────────────────────
# COLOR REFERENCE SETUP
# ────────────────────────────────────────────────
COLORS_BGR = {
    "green": (50, 200, 50),
    "blue":  (255, 100, 50),
    "cyan":  (255, 255, 0),
    "yellow":(0, 255, 255),
    "orange":(0, 140, 255),
    "purple":(200, 0, 150),
}

COLOR_NAMES   = list(COLORS_BGR.keys())
TARGET_VALUES = list(COLORS_BGR.values())
target_array  = np.array([TARGET_VALUES], dtype=np.uint8)
TARGETS_LAB   = cv.cvtColor(target_array, cv.COLOR_BGR2Lab)[0]

# ────────────────────────────────────────────────
# ROBUST COLOR CLASSIFIER (HSV + Lab fallback)
# ────────────────────────────────────────────────
def classify_color(frame_bgr: np.ndarray, point, radius=20) -> str:
    """
    Robust color classifier for Zuma balls including pink as purple.

    Primary: HSV hue ranges (fast, decisive)
    Fallback: weighted Lab distance (only for purple, blue, green, yellow)

    Returns: COLOR_NAMES or "unknown"
    """
    cx, cy = int(point[0]), int(point[1])
    r = radius
    h, w = frame_bgr.shape[:2]

    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)

    if y0 >= y1 or x0 >= x1:
        return "unknown"

    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return "unknown"

    # Mean BGR color of ROI
    mean_bgr = cv.mean(roi)[:3]
    pix_bgr = np.uint8([[mean_bgr]])

    # Convert to HSV and Lab
    pix_hsv = cv.cvtColor(pix_bgr, cv.COLOR_BGR2HSV)[0, 0]
    pix_lab = cv.cvtColor(pix_bgr, cv.COLOR_BGR2Lab)[0, 0]

    H, S, V = int(pix_hsv[0]), int(pix_hsv[1]), int(pix_hsv[2])

    # Reject very dark / very desaturated areas
    if V < 50 or S < 35:
        return "unknown"

    # ── Hue-based primary classification ──
    # Purple + pink
    if 125 <= H <= 180:
        return "purple"
    if 100 <= H <= 125:
        return "blue"
    if 50 <= H <= 80:
        return "green"
    if 20 <= H <= 35:
        return "yellow"
    if 8 <= H <= 20:
        return "orange"

    # ── Fallback: weighted Lab nearest neighbor ──
    weights = np.array([0.3, 1.0, 1.0], dtype=np.float32)
    diff = (TARGETS_LAB - pix_lab) * weights
    dists = np.linalg.norm(diff, axis=1)
    idx = int(np.argmin(dists))

    # Only fallback for purple, blue, green, yellow
    if COLOR_NAMES[idx] in ["purple", "blue", "green", "yellow"]:
        if dists[idx] <= 55:  # stricter max distance threshold
            return COLOR_NAMES[idx]

    return "purple"

# ────────────────────────────────────────────────
# ORB TEMPLATE MATCHING
# ────────────────────────────────────────────────
def orb_template_match(template_path, scene_path, scales=(0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3),
                       nfeatures=5000, ratio=0.7, min_inliers=15, ransac_thresh=3.0):
    template = cv.imread(str(template_path), cv.IMREAD_GRAYSCALE)
    scene    = cv.imread(str(scene_path), cv.IMREAD_GRAYSCALE)

    if template is None or scene is None:
        raise RuntimeError("Failed to load images")

    orb = cv.ORB_create(nfeatures=nfeatures)
    kp_scene, des_scene = orb.detectAndCompute(scene, None)
    if des_scene is None:
        return None, scene

    bf = cv.BFMatcher(cv.NORM_HAMMING)
    best, best_inliers = None, 0

    for scale in scales:
        tpl = cv.resize(template, None, fx=scale, fy=scale)
        kp_tpl, des_tpl = orb.detectAndCompute(tpl, None)
        if des_tpl is None:
            continue

        matches = bf.knnMatch(des_tpl, des_scene, k=2)
        good = [m for m,n in matches if m.distance < ratio*n.distance]
        if len(good) < 4:
            continue

        src = np.float32([kp_tpl[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv.findHomography(src, dst, cv.RANSAC, ransac_thresh)
        if H is None:
            continue

        inliers = int(mask.sum())
        if inliers < min_inliers:
            continue

        h_tpl, w_tpl = tpl.shape
        corners = np.float32([[0,0],[w_tpl,0],[w_tpl,h_tpl],[0,h_tpl]]).reshape(-1,1,2)
        corners_scene = cv.perspectiveTransform(corners, H)
        area = cv.contourArea(corners_scene.astype(np.int32))
        if area < 800:
            continue

        if inliers > best_inliers:
            center_tpl = np.float32([[w_tpl/2,h_tpl/2]]).reshape(1,1,2)
            center_scene = cv.perspectiveTransform(center_tpl, H)[0][0]
            best = {
                "H": H,
                "corners": corners_scene,
                "center": center_scene,
                "inliers": inliers,
                "scale": scale,
                "template_size": (w_tpl,h_tpl),
            }
            best_inliers = inliers
    return best, scene

# ────────────────────────────────────────────────
# DETECT FROG AND BALLS
# ────────────────────────────────────────────────
def detect_frog_and_balls(template_path, scene_path, mouth_relative=np.float32([-0.4,0.0])):
    scene_bgr = cv.imread(str(scene_path))
    if scene_bgr is None:
        raise RuntimeError("Cannot load scene")

    result, _ = orb_template_match(template_path, scene_path)
    if result is None:
        return None

    h, w = result["template_size"]
    H = result["H"]
    center = result["center"].astype(int)

    mouth_tpl = np.float32([[w/2 + mouth_relative[0]*w, h/2 + mouth_relative[1]*h]]).reshape(1,1,2)
    mouth = cv.perspectiveTransform(mouth_tpl, H)[0][0].astype(int)

    # Updated classifier
    center_color = classify_color(scene_bgr, center, 22)
    mouth_color  = classify_color(scene_bgr, mouth, 22)

    vis = scene_bgr.copy()
    cv.circle(vis, tuple(center), 20, COLORS_BGR.get(center_color,(255,255,255)),-1)
    cv.circle(vis, tuple(center), 20,(255,255,255),3)
    cv.putText(vis, f"NEXT: {center_color.upper()}",(center[0]-40,center[1]-30),
               cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv.circle(vis, tuple(mouth),30, COLORS_BGR.get(mouth_color,(255,255,255)),-1)
    cv.circle(vis, tuple(mouth),30,(255,255,255),3)
    cv.putText(vis, f"MOUTH: {mouth_color.upper()}", (mouth[0]-50, mouth[1]-40),
               cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    corners = result["corners"].reshape(4,2).astype(int)
    for i in range(4):
        cv.line(vis, tuple(corners[i]), tuple(corners[(i+1)%4]), (0,255,0),2)

    return {
        "frog_center": tuple(center),
        "current_ball": tuple(mouth),
        "next_ball_color": center_color,
        "current_ball_color": mouth_color,
        "inliers": result["inliers"],
        "scale": result["scale"],
        "vis": vis,
    }

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
if __name__ == "__main__":
    template_path = TEMPLATE_DIR / "launchers" / "img.png"
    scene_path    = TEST_DIR / "img_1.png"

    info = detect_frog_and_balls(template_path, scene_path)
    if info is None:
        print("Frog not detected")
    else:
        for k,v in info.items():
            if k!="vis":
                print(f"{k}: {v}")

        cv.namedWindow("Frog Detection", cv.WINDOW_NORMAL)
        cv.resizeWindow("Frog Detection",1024,768)
        cv.imshow("Frog Detection",info["vis"])
        cv.waitKey(0)
        cv.destroyAllWindows()
