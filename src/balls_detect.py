import cv2
import numpy as np

TARGET_WIDTH = 640

BALL_COLORS = {
	"purple": {"lower": np.array([131, 64, 55]), "upper": np.array([179, 203, 255])},
	"blue": {"lower": np.array([91, 109, 51]), "upper": np.array([118, 211, 255])},
	"green": {"lower": np.array([39, 130, 40]), "upper": np.array([82, 219, 255])},
	"yellow": {"lower": np.array([24, 133, 60]), "upper": np.array([30, 255, 255])},
	"orange": {
		"lower": np.array([7, 123, 117]),
		"upper": np.array([22, 237, 255])
	},
}


def detect_balls(frame_bgr, debug=False):
	if frame_bgr is None:
		return []

	height, width = frame_bgr.shape[:2]
	scale = 1.0
	processing_frame = frame_bgr

	if width > TARGET_WIDTH:
		scale = TARGET_WIDTH / width
		processing_frame = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)

	hsv = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)
	hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

	detected_balls = []

	for color_name, ranges in BALL_COLORS.items():
		mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])

		kernel = np.ones((3, 3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
		mask = cv2.dilate(mask, kernel, iterations=1)

		dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

		_, peaks = cv2.threshold(dist_transform, 0.65 * dist_transform.max(), 255, 0)
		peaks = np.uint8(peaks)

		contours, _ = cv2.findContours(peaks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			area = cv2.contourArea(cnt)

			if area < 10:
				continue

			(x_small, y_small), _ = cv2.minEnclosingCircle(cnt)
			center_x_small, center_y_small = int(x_small), int(y_small)

			if 0 <= center_y_small < dist_transform.shape[0] and 0 <= center_x_small < dist_transform.shape[1]:
				real_radius_small = dist_transform[center_y_small, center_x_small]
			else:
				real_radius_small = 10

			if real_radius_small < 4 or real_radius_small > 35:
				continue

			final_x = int(center_x_small / scale)
			final_y = int(center_y_small / scale)
			final_radius = int(real_radius_small / scale)

			detected_balls.append({
				"color": "red" if "red" in color_name else color_name,
				"center": (final_x, final_y),
				"radius": final_radius,
				"type": "standard"
			})

	return detected_balls


if __name__ == "__main__":
	from config import CAPTURES_DIR
	import os

	files = sorted(list(CAPTURES_DIR.glob("*.png")), key=os.path.getmtime)
	if files:
		last_img_path = str(files[-1])
		print(f"Testing on: {last_img_path}")

		frame = cv2.imread(last_img_path)
		balls = detect_balls(frame, debug=True)

		print(f"Detected {len(balls)} balls.")

		for ball in balls:
			x, y = ball["center"]
			r = ball["radius"]
			c = ball["color"]

			cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
			cv2.putText(frame, c, (x - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

		cv2.imshow("Final Stable Detection", frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
