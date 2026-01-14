import cv2 as cv
import mss
import time
from config import TEMPLATE_DIR, DEBUG, DISPLAY_SIZE
from detectors import ZumaBallDetector, WindowDetector
from threads import SharedFrame, CaptureThread, ThreadedBallDetector

WINDOW_NAME = "Zuma Detector AI"

def test_realtime_threaded(window_detector, ball_detector):
    print("Threaded Real-Time Mode")

    # --- Screen capture ---
    with mss.mss() as sct:
        root_monitor = sct.monitors[1]

    capture = CaptureThread(root_monitor)
    capture.start()

    frame_shared = capture.frame
    rect_shared = SharedFrame()
    result_shared = SharedFrame()

    ball_thread = ThreadedBallDetector(ball_detector, frame_shared, rect_shared, result_shared)
    ball_thread.start()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    fps = 0
    prev_time = time.time()
    window_locked = False

    while True:
        frame, _ = frame_shared.get_latest_with_id()
        if frame is None:
            continue

        display = frame.copy()

        # --- Detect window only once or on 'r' ---
        if not window_locked:
            rect, _ = window_detector.detect(frame)
            if rect is not None:
                x, y, w, h = rect
                rect_shared.set((x, y, w, h))
                window_locked = True
                print("Window locked!")
        else:
            rect, _ = rect_shared.get_latest_with_id()
            if rect is not None:
                x, y, w, h = rect
                cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw balls
        balls, _ = result_shared.get_latest_with_id()
        if balls is not None:
            for d in balls:
                cv.circle(display, (d["x"], d["y"]), d["radius"], (0, 255, 0), 2)
                cv.putText(display, d["color"], (d["x"] - 10, d["y"] - d["radius"] - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time
        cv.putText(display, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv.imshow(WINDOW_NAME, cv.resize(display, DISPLAY_SIZE))

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            window_locked = False
            print("Resetting window detection...")

    # Cleanup
    capture.stop()
    ball_thread.stop()
    cv.destroyAllWindows()


def main():
    print("Zuma Ball Detection System")
    print("=" * 50)

    window_detector = WindowDetector(TEMPLATE_DIR / "img.png", debug=DEBUG)
    ball_detector = ZumaBallDetector(debug=DEBUG)

    print("Select mode:")
    print("  4 - Real-time Screen Capture")
    choice = input("Enter choice (1-4): ").strip()

    if choice == "4":
        test_realtime_threaded(window_detector, ball_detector)


if __name__ == "__main__":
    main()
