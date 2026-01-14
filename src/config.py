from pathlib import Path

# -------------------------------------------------
# Base directory (project root)
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# -------------------------------------------------
# Image directories
# -------------------------------------------------
IMAGE_DIR = BASE_DIR / "img"
TEST_DIR = IMAGE_DIR / "tests"
OUTPUT_DIR = IMAGE_DIR / "outputs"
TEMPLATE_DIR = IMAGE_DIR / "templates"

# -------------------------------------------------
# Ensure directories exist
# -------------------------------------------------
IMAGE_DIR.mkdir(exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Supported image extensions
# -------------------------------------------------
EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# -------------------------------------------------
# Debug flag
# -------------------------------------------------
DEBUG = False
