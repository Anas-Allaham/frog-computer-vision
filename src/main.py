import os
import cv2 as cv

from detectors import WindowDetector
from cv_helper import image_show, image_read
from config import TEST_DIR, OUTPUT_DIR, EXTENSIONS


def main():
    detector = WindowDetector()

    for filename in sorted(os.listdir(TEST_DIR)):
        if not filename.lower().endswith(EXTENSIONS):
            continue

        path = TEST_DIR / filename
        img = image_read(path)

        print(f"\nProcessing: {filename}")

        rectangles, _ = detector.detect(img)
        print(f"Detected {len(rectangles)} windows")

        output = detector.draw(img, rectangles)

        # Save result
        out_path = OUTPUT_DIR / filename
        cv.imwrite(str(out_path), output)

        # Optional visualization
        # image_show("Edges", edges)
        image_show("Detected Windows", output)

        if cv.waitKey(0) & 0xFF == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
