import cv2 as cv
from pathlib import Path
from cv_helper import preprocess_for_template, image_read


class TemplateMatcher:
    def __init__(
        self,
        template_path: str,
        scales=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
        threshold=0.7,
        early_exit=0.9,
        method=cv.TM_CCOEFF_NORMED,
        debug=False,
    ):
        self.template_path = Path(template_path)
        self.threshold = threshold
        self.scales = scales
        self.early_exit = early_exit
        self.method = method
        self.debug = debug

        base = image_read(str(self.template_path), cv.IMREAD_GRAYSCALE)

        self.templates = []
        for s in self.scales:
            resized = cv.resize(
                base,
                None,
                fx=s,
                fy=s,
                interpolation=cv.INTER_LINEAR,
            )
            self.templates.append((s, resized))

        if self.debug:
            print(
                f"[TEMPLATE] Loaded {self.template_path.name} "
                f"({len(self.templates)} scales)"
            )

    def match(self, img):
        img = preprocess_for_template(img)

        best_score = -1.0
        best_scale = None
        best_loc = None

        for scale, tmpl in self.templates:
            th, tw = tmpl.shape[:2]
            if th >= img.shape[0] or tw >= img.shape[1]:
                continue

            res = cv.matchTemplate(img, tmpl, self.method)
            _, max_val, _, max_loc = cv.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_loc = max_loc

            if best_score >= self.early_exit:
                break

        if self.debug:
            print(
                f"[MATCH] score={best_score:.3f}, "
                f"scale={best_scale}"
            )

        return best_score >= self.threshold, best_score, best_scale, best_loc
