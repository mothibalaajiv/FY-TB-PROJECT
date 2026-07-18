from pathlib import Path

# Project root:
# TB-Cough-Detection/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

ZIP_FILE_NAME = "new-audio.zip"

ZIP_FILE_PATH = DATA_DIR / ZIP_FILE_NAME

EXTRACT_FOLDER_NAME = "extracted_dataset"

EXTRACT_PATH = DATA_DIR / EXTRACT_FOLDER_NAME

CLASSES = ["notb", "tb"]

SR = 16000
DURATION = 3
AUDIO_LENGTH = SR * DURATION
N_MELS = 128