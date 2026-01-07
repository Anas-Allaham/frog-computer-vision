import os
import cv2 as cv

from detectors import WindowDetector
from cv_helper import image_show, image_read
from config import TEST_DIR, OUTPUT_DIR, TEMPLATE_DIR, EXTENSIONS#, DEBUG


def main():
    detector = WindowDetector(
        template_path=TEMPLATE_DIR / "img.png",
    )


    for filename in sorted(os.listdir(TEST_DIR)):
        if not filename.lower().endswith(EXTENSIONS):
            continue

        path = TEST_DIR / filename
        img = image_read(path)

        rect, _ = detector.detect(img)

        output = detector.draw(img, rect)

        # Save result
        out_path = OUTPUT_DIR / filename
        cv.imwrite(str(out_path), output)

        image_show("Detected Zuma Window", output)

        if cv.waitKey(0) & 0xFF == 27:
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()