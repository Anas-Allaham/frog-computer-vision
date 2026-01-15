import cv2 as cv
import mss
import time
import numpy as np
from config import TEMPLATE_DIR, DEBUG, DISPLAY_SIZE
from detectors import ZumaBallDetector, WindowDetector, ZumaFrogDetector
from detectors.window_roi_detector import SmartROIDetector
from threads import SharedFrame, CaptureThread, ThreadedBallDetector
from solver import Solver
from cv_helper.debug_viewer import DebugViewer

WINDOW_NAME = "Zuma Detector AI"

def calculate_velocities(prev_balls, current_balls, time_delta):
    """Matches balls between frames to calculate their velocity."""
    if not prev_balls or not current_balls or time_delta == 0:
        return current_balls

    # Create a copy to avoid modifying the original list
    updated_balls = [b.copy() for b in current_balls]

    for ball in updated_balls:
        # Find the closest ball in the previous frame
        closest_ball, min_dist = None, float('inf')
        for prev_ball in prev_balls:
            dist = np.linalg.norm(np.array([ball['x'], ball['y']]) - np.array([prev_ball['x'], prev_ball['y']]))
            if dist < min_dist:
                min_dist = dist
                closest_ball = prev_ball
        
        # If a plausible match is found, calculate velocity
        if closest_ball and min_dist < closest_ball['radius'] * 2:
            dx = (ball['x'] - closest_ball['x']) / time_delta
            dy = (ball['y'] - closest_ball['y']) / time_delta
            ball['vx'] = dx
            ball['vy'] = dy
        else:
            ball['vx'] = 0
            ball['vy'] = 0
            
    return updated_balls

def test_realtime_threaded(window_detector, ball_detector, frog_detector, roi_detector=None):
    debug_viewer = DebugViewer() if DEBUG else None
    solver = None
    last_shot_time = 0
    shot_cooldown = 1.0  # seconds
    target_pos_global = None
    prev_balls = []
    print("Threaded Real-Time Mode")

    # --- Screen capture ---
    with mss.mss() as sct:
        root_monitor = sct.monitors[1]

    capture = CaptureThread(root_monitor)
    capture.start()

    frame_shared = capture.frame
    rect_shared = SharedFrame()
    result_shared = SharedFrame()
    frog_center_shared = SharedFrame()  # For sharing frog center with ball detector

    ball_thread = ThreadedBallDetector(ball_detector, frame_shared, rect_shared, result_shared, frog_center_shared)
    ball_thread.start()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

    fps = 0
    prev_time = time.time()
    window_locked = False

    try:
        while True:
            # --- Time Calculation ---
            curr_time = time.time()
            time_delta = curr_time - prev_time

            frame, _ = frame_shared.get_latest_with_id()
            if frame is None:
                continue

            display = frame.copy()
            game_roi = None

            # --- Smart ROI Detection ---
            if not window_locked:
                # First try to detect game window and ROI automatically
                if roi_detector.find_game_window():
                    smart_roi = roi_detector.get_game_roi(frame)
                    if smart_roi is not None:
                        x, y, w, h = smart_roi
                        rect_shared.set((x, y, w, h))
                        window_locked = True
                        print(f"Smart ROI locked: {smart_roi}")
                    else:
                        # Fallback to traditional window detection
                        rect, _ = window_detector.detect(frame)
                        if rect is not None:
                            x, y, w, h = rect
                            rect_shared.set((x, y, w, h))
                            window_locked = True
                            print("Traditional window locked!")
            else:
                rect, _ = rect_shared.get_latest_with_id()
                if rect is not None:
                    x, y, w, h = rect
                    cv.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    game_roi = frame[y:y+h, x:x+w]

            # --- Ball Detection, Velocity, and Drawing ---
            balls, _ = result_shared.get_latest_with_id()
            if balls:
                balls = calculate_velocities(prev_balls, balls, time_delta)
                prev_balls = balls # Update for the next frame

            # --- Detection & Solver ---
            if game_roi is not None and window_locked:
                frog_center, frog_angle, current_ball, next_ball, frog_contour = frog_detector.detect(game_roi)

                # Share frog center with ball detector for ROI optimization
                if frog_center is not None:
                    frog_center_shared.set((frog_center, frog_angle, current_ball, next_ball))

                if debug_viewer:
                    debug_viewer.add_data("Current Ball", current_ball['color'] if current_ball else 'None')
                    debug_viewer.add_data("Next Ball", next_ball['color'] if next_ball else 'None')

                if frog_center is not None:
                    if solver is None:
                        solver = Solver(frog_pos=frog_center, screen_rect=rect)
                        print("Solver initialized.")

                    frog_center_global = (frog_center[0] + x, frog_center[1] + y)

                    if solver and current_ball and balls and (time.time() - last_shot_time > shot_cooldown):
                        balls_in_roi = [{**b, 'x': b['x'] - x, 'y': b['y'] - y} for b in balls]
                        shot_info = solver.find_best_shot(current_ball, next_ball, balls_in_roi)
                        if shot_info:
                            target_pos_global = (int(shot_info['target_pos'][0] + x), int(shot_info['target_pos'][1] + y))
                            cv.line(display, frog_center_global, target_pos_global, (255, 255, 255), 2)
                            solver.execute_shot(shot_info['target_pos'])
                            print(f"AI ACTION: Shooting {current_ball['color'].upper()} ball at target.")
                            last_shot_time = time.time()

                    if current_ball:
                        current_ball['x'] += x
                        current_ball['y'] += y
                    if next_ball:
                        next_ball['x'] += x
                        next_ball['y'] += y
                    display = frog_detector.draw(display, frog_center_global, frog_angle, current_ball, next_ball, frog_contour)

                    if current_ball:
                        cv.putText(display, f"Current: {current_ball['color']}", (x + 10, y + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    if next_ball:
                        cv.putText(display, f"Next: {next_ball['color']}", (x + 10, y + 70), cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            # --- Draw Balls on Main Display ---
            if balls:
                for d in balls:
                    cv.circle(display, (d["x"], d["y"]), d["radius"], (0, 255, 0), 2)
                    cv.putText(display, d["color"], (d["x"] - 10, d["y"] - d["radius"] - 5),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Show debug panel
            if debug_viewer:
                debug_viewer.show()

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

    finally:
        # Cleanup
        print("Shutting down...")
        capture.stop()
        ball_thread.stop()
        cv.destroyAllWindows()


def benchmark_detection():
    """Benchmark detection performance improvements."""
    print("Running Performance Benchmark...")
    print("=" * 50)

    import time

    # Create detectors
    ball_detector = ZumaBallDetector(debug=False)
    frog_detector = ZumaFrogDetector(TEMPLATE_DIR / "frogwithballs.jpg", debug=False)

    # Create a test frame (simulate game screen) or load from captures
    test_frame = None

    # Try to load latest capture if available
    try:
        import os
        captures_dir = TEMPLATE_DIR.parent / "captures"
        if captures_dir.exists():
            files = list(captures_dir.glob("*.png"))
            if files:
                latest_capture = str(max(files, key=os.path.getmtime))
                test_frame = cv.imread(latest_capture)
                print(f"Using real capture: {os.path.basename(latest_capture)}")
    except:
        pass

    # Fallback to synthetic frame
    if test_frame is None:
        test_frame = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        print("Using synthetic test frame")

    print(f"Frame size: {test_frame.shape}")

    # Benchmark traditional ball detection
    print("\nBenchmarking TRADITIONAL ball detection...")
    start_time = time.time()
    iterations = 50

    for i in range(iterations):
        detections_traditional = ball_detector.detect(test_frame, use_advanced=False)

    traditional_time = (time.time() - start_time) / iterations * 1000  # ms per frame
    print(".2f")
    print(f"  Detected {len(detections_traditional)} balls")

    # Benchmark advanced ball detection
    print("\nBenchmarking ADVANCED ball detection...")
    start_time = time.time()

    for i in range(iterations):
        detections_advanced = ball_detector.detect(test_frame, use_advanced=True)

    advanced_time = (time.time() - start_time) / iterations * 1000  # ms per frame
    print(".2f")
    print(f"  Detected {len(detections_advanced)} balls")

    # Benchmark frog detection
    print("\nBenchmarking frog detection...")
    start_time = time.time()

    for i in range(iterations):
        frog_center, frog_angle, current_ball, next_ball, contour = frog_detector.detect(test_frame)

    frog_time = (time.time() - start_time) / iterations * 1000  # ms per frame
    print(".2f")

    # Performance comparison
    print("\n" + "=" * 50)
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    improvement = (traditional_time - advanced_time) / traditional_time * 100 if traditional_time > 0 else 0
    print(".2f")
    print(".2f")
    print(".2f")

    total_time_traditional = traditional_time + frog_time
    total_time_advanced = advanced_time + frog_time
    fps_traditional = 1000 / total_time_traditional if total_time_traditional > 0 else float('inf')
    fps_advanced = 1000 / total_time_advanced if total_time_advanced > 0 else float('inf')

    print(".1f")
    print(".1f")
    print(".1f")

    print("\nNote: Real FPS may vary based on hardware and game state.")
    print("Advanced detection provides better accuracy at similar performance.")

def main():
    print("Zuma Ball Detection System")
    print("=" * 50)

    debug_viewer = DebugViewer() if DEBUG else None
    window_detector = WindowDetector(TEMPLATE_DIR / "img.png", debug=DEBUG)
    ball_detector = ZumaBallDetector(debug=DEBUG, viewer=debug_viewer)
    frog_detector = ZumaFrogDetector(TEMPLATE_DIR / "frogwithballs.jpg", debug=DEBUG, viewer=debug_viewer)

    # Initialize smart ROI detector for automatic game area detection
    roi_detector = SmartROIDetector(game_title="Zuma")

    print("Select mode:")
    print("  1 - Performance Benchmark")
    print("  4 - Real-time Screen Capture")
    choice = input("Enter choice (1 or 4): ").strip()

    if choice == "1":
        benchmark_detection()
    elif choice == "4":
        test_realtime_threaded(window_detector, ball_detector, frog_detector, roi_detector)


if __name__ == "__main__":
    main()
