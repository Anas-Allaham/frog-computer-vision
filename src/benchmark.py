#هذا الكود لحساب fbs
import time
import cv2
import os
from config import CAPTURES_DIR
from balls_detect import detect_balls


def get_latest_capture():
	if not CAPTURES_DIR.exists():
		return None
	files = list(CAPTURES_DIR.glob("*.png"))
	if not files:
		return None
	return str(max(files, key=os.path.getmtime))


def main():
	img_path = get_latest_capture()
	if not img_path:
		print("Error: No images found to test.")
		return

	frame = cv2.imread(img_path)
	if frame is None:
		print("Error reading image.")
		return

	print(f"Testing performance on image: {os.path.basename(img_path)}")
	print("Running warmup...")
	detect_balls(frame)

	iterations = 100
	print(f"Starting benchmark ({iterations} iterations)...")

	start_time = time.time()

	for i in range(iterations):
		# نرسل debug=False لأننا لا نريد الرسم، نريد فقط الحسابات
		_ = detect_balls(frame, debug=False)

	end_time = time.time()

	total_time = end_time - start_time
	fps = iterations / total_time

	print("-" * 30)
	print(f"Total Time: {total_time:.4f} seconds")
	print(f"Average FPS: {fps:.2f} 🚀")
	print("-" * 30)

	if fps > 30:
		print("Result: Excellent! Real-time ready.")
	elif fps > 15:
		print("Result: Good, but be careful with adding more logic.")
	else:
		print("Result: Slow. Optimization needed.")


if __name__ == "__main__":
	main()