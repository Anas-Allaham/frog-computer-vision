import cv2 as cv
import numpy as np
import mss
import time
from detectors import ZumaBallDetector, WindowDetector
from config import TEMPLATE_DIR, TEST_DIR, DEBUG, OUTPUT_DIR

# --- CONSTANTS ---
WINDOW_NAME = "Zuma Detector AI"
DISPLAY_SIZE = (800, 600)
DEFAULT_IMG_PATH = TEST_DIR / "img_10.png"


# --- SETUP HELPERS ---

def setup_system():
    """Initializes detectors and window one time."""
    print("Initializing Detectors...")
    window_detector = WindowDetector(template_path=TEMPLATE_DIR / "img.png", debug=DEBUG)
    ball_detector = ZumaBallDetector(debug=DEBUG)

    # Setup Display
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, *DISPLAY_SIZE)

    return window_detector, ball_detector


def show_in_window(img):
    """Resizes and displays the image in the standard window."""
    if img is None: return
    resized_img = cv.resize(img, DISPLAY_SIZE)
    cv.imshow(WINDOW_NAME, resized_img)


def detect_and_draw(img, ball_detector, fps=None):
    """Core logic: Detects balls and draws results."""
    balls = ball_detector.detect(img)
    result_img = ball_detector.draw(img, balls)

    if fps is not None:
        cv.putText(result_img, f"FPS: {int(fps)}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return result_img


def process_full_image(img, window_detector, ball_detector):
    """Finds game window -> Crops -> Detects Balls."""
    rect, _ = window_detector.detect(img)

    if rect is None:
        return None

    x, y, w, h = cv.boundingRect(rect)

    # Boundary checks
    x, y = max(0, x), max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Crop and Detect
    game_crop = img[y:y + h, x:x + w].copy()
    return detect_and_draw(game_crop, ball_detector)


# --- TEST MODES ---

def test_single_image(window_detector, ball_detector, img_path):
    """Test detection on a specific image path."""
    print(f"Testing image: {img_path}")
    img = cv.imread(str(img_path))

    if img is None:
        print(f"Could not load image at {img_path}")
        return

    result = process_full_image(img, window_detector, ball_detector)

    if result is not None:
        show_in_window(result)
        print("Success. Press any key to continue...")
        cv.waitKey(0)
    else:
        print("Game window not found in image.")
        # Show original image if detection failed, just to confirm load
        show_in_window(img)
        cv.waitKey(0)


def batch_test_images(window_detector, ball_detector):
    """Test detection on all images in directory."""
    image_files = list(TEST_DIR.glob("*.png")) + list(TEST_DIR.glob("*.jpg"))

    if not image_files:
        print(f"No images found in {TEST_DIR}")
        return

    print(f"Found {len(image_files)} images. Press 'q' to stop.")

    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        img = cv.imread(str(img_path))
        if img is None: continue

        result = process_full_image(img, window_detector, ball_detector)

        if result is not None:
            show_in_window(result)
            cv.imwrite(f"{OUTPUT_DIR}/Detection-{img_path.name}", result)

            if cv.waitKey(0) == ord('q'): break
        else:
            print(f"  No game window found in {img_path.name}")


def test_video(window_detector, ball_detector):
    """Test detection on webcam."""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("Video Mode. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        result = process_full_image(frame, window_detector, ball_detector)

        if result is not None:
            show_in_window(result)
        else:
            show_in_window(frame)

        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()


def test_realtime_mss_stream(window_detector, ball_detector):
    """Real-time MSS stream: detect balls only inside cropped window."""
    print("=" * 50)
    print("Real-time Streaming Mode")
    print("Press 'q' to quit, 'r' to reset search")
    print("=" * 50)

    with mss.mss() as sct:
        root_monitor = sct.monitors[1]
        capture_area = root_monitor
        is_locked_on = False
        fps = 0
        prev_time = time.time()

        while True:
            # Grab the screen
            img_bgra = np.array(sct.grab(capture_area))
            frame = cv.cvtColor(img_bgra, cv.COLOR_BGRA2BGR)

            if not is_locked_on:
                # --- SEARCH MODE: detect the window in full screen ---
                rect, _ = window_detector.detect(frame)
                display_frame = frame.copy()
                if rect is not None:
                    x, y, w, h = cv.boundingRect(rect)
                    capture_area = {
                        "top": root_monitor["top"] + y,
                        "left": root_monitor["left"] + x,
                        "width": w,
                        "height": h
                    }
                    is_locked_on = True
                    print(f"Locked on Game Window: {w}x{h}")
                    continue  # next loop will capture only the window
                else:
                    cv.putText(display_frame, "Searching for game...", (20, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    show_in_window(display_frame)

            else:
                # --- TRACKING MODE: detect balls only inside the cropped window ---
                balls = ball_detector.detect(frame)
                result_img = ball_detector.draw(frame, balls)

                # FPS calculation
                curr_time = time.time()
                fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else fps
                prev_time = curr_time

                cv.putText(result_img, f"FPS: {int(fps)}", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                show_in_window(result_img)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                is_locked_on = False
                capture_area = root_monitor
                print("Resetting search...")

    cv.destroyAllWindows()


# --- MAIN ---

def main():
    print("Zuma Ball Detection System")
    print("=" * 50)

    # 1. Setup Detectors & Window ONCE
    window_det, ball_det = setup_system()

    print("Select mode:")
    print("  1 - Test single image")
    print("  2 - Test video/webcam")
    print("  3 - Batch test all images")
    print("  4 - Real-time Screen Capture (MSS)")
    print("=" * 50)

    choice = input("Enter choice (1-4): ").strip()

    try:
        if choice == "1":
            # Pass the default image path, or ask user for input here
            test_single_image(window_det, ball_det, DEFAULT_IMG_PATH)
        elif choice == "2":
            test_video(window_det, ball_det)
        elif choice == "3":
            batch_test_images(window_det, ball_det)
        elif choice == "4":
            test_realtime_mss_stream(window_det, ball_det)
        else:
            print("Invalid choice, running default image test.")
            test_single_image(window_det, ball_det, DEFAULT_IMG_PATH)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
