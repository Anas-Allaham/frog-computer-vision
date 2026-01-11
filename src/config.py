import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEMPLATES_DIR = BASE_DIR / "templates"
CAPTURES_DIR = BASE_DIR / "captures"

CAPTURES_DIR.mkdir(exist_ok=True)

SCORE_TEMPLATE_PATH = TEMPLATES_DIR / "score.png"
X_TEMPLATE_PATH = TEMPLATES_DIR / "x.png"

GAME_TITLE = "Zuma"
DEBUG_MODE = True