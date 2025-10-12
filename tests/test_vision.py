# tests/test_vision.py
from pathlib import Path
import sys
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))           # <— agrega la raíz al sys.path
load_dotenv(ROOT / ".env")              # <— carga GOOGLE_APPLICATION_CREDENTIALS

from voice_vision.google_emotion import detect_emotion_from_url

if __name__ == "__main__":
    print(detect_emotion_from_url(
        "https://thumbs.dreamstime.com/b/retrato-de-la-mujer-de-la-belleza-muchacha-con-la-sonrisa-hermosa-de-la-cara-76138194.jpg"
    ))
