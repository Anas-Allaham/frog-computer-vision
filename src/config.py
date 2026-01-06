from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = BASE_DIR / "img" / "tests"
OUTPUT_DIR = BASE_DIR / "img" / "outputs"
TEMPLATE_DIR = BASE_DIR / "img" / "templates"

EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")