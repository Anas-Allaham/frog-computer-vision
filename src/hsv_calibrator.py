#هذا الكود لكشف درحات الالوان (لا حاجة له ضمن العمل لكنه مفيد لكشف الالوان)
import cv2
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
CAPTURES_DIR = BASE_DIR / "captures"


def get_latest_capture():
	"""يبحث عن أحدث صورة تم حفظها في مجلد captures"""
	if not CAPTURES_DIR.exists():
		print(f"Error: Folder {CAPTURES_DIR} not found.")
		return None

	files = list(CAPTURES_DIR.glob("*.png"))
	if not files:
		print("Error: No images found in captures folder. Run capture.py first.")
		return None

	# ترتيب الملفات حسب وقت التعديل واختيار الأحدث
	latest_file = max(files, key=lambda f: f.stat().st_mtime)
	print(f"Using latest image: {latest_file.name}")
	return str(latest_file)


def empty(a):
	pass


def main():
	img_path = get_latest_capture()
	if not img_path:
		return

	# إعداد نافذة التحكم
	cv2.namedWindow("Trackbars")
	cv2.resizeWindow("Trackbars", 640, 300)

	# إنشاء أشرطة التمرير (Hue: 0-179, Sat/Val: 0-255)
	cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
	cv2.createTrackbar("Hue Max", "Trackbars", 179, 179, empty)
	cv2.createTrackbar("Sat Min", "Trackbars", 0, 255, empty)
	cv2.createTrackbar("Sat Max", "Trackbars", 255, 255, empty)
	cv2.createTrackbar("Val Min", "Trackbars", 0, 255, empty)
	cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

	original = cv2.imread(img_path)
	# تصغير الصورة قليلاً إذا كانت كبيرة جداً لسهولة العرض
	original = cv2.resize(original, (0, 0), fx=0.6, fy=0.6)

	while True:
		# تحويل الصورة إلى فضاء HSV
		imgHSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

		# قراءة القيم الحالية من الأشرطة
		h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
		h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
		s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
		s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
		v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
		v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

		# إنشاء القناع (Mask)
		lower = np.array([h_min, s_min, v_min])
		upper = np.array([h_max, s_max, v_max])
		mask = cv2.inRange(imgHSV, lower, upper)

		# عرض النتيجة (الصورة المعزولة)
		result = cv2.bitwise_and(original, original, mask=mask)

		# دمج القناع مع النتيجة للمقارنة
		mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		h_stack = np.hstack([original, result])

		cv2.imshow("Original vs Result (Adjust Trackbars)", h_stack)
		cv2.imshow("Mask (Black & White)", mask)

		key = cv2.waitKey(1)
		if key == ord('q'):
			break
		elif key == ord('s'):
			print(f"\nSaved Range: Lower=[{h_min},{s_min},{v_min}], Upper=[{h_max},{s_max},{v_max}]")

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()