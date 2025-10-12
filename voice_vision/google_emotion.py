# voice_vision/google_emotion.py
from __future__ import annotations
from typing import Dict, List
from functools import lru_cache
from google.cloud import vision
import httpx

# Mapea las "likelihood" de Vision a una escala 0–4
LIKELIHOOD_SCORES = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

def _score(v: int) -> int:
    return LIKELIHOOD_SCORES.get(int(v), 0)

@lru_cache(maxsize=1)
def _get_client() -> vision.ImageAnnotatorClient:
    # Reutiliza el cliente (thread-safe)
    return vision.ImageAnnotatorClient()

def _summarize_face(f) -> Dict:
    scores = {
        "joy": _score(f.joy_likelihood),
        "sorrow": _score(f.sorrow_likelihood),
        "anger": _score(f.anger_likelihood),
        "surprise": _score(f.surprise_likelihood),
    }
    scores["dominant"] = max(scores, key=scores.get) if max(scores.values()) > 0 else "none"
    return scores

def detect_emotion_from_bytes(content: bytes) -> Dict:
    """
    Emoción desde bytes (para fotos subidas por multipart/webcam).
    """
    client = _get_client()
    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    faces: List = response.face_annotations
    if not faces:
        return {"joy": 0, "sorrow": 0, "anger": 0, "surprise": 0, "dominant": "none"}

    return _summarize_face(faces[0])

def detect_emotion_from_url(image_url: str) -> Dict:
    """
    Emoción desde URL pública (útil para pruebas rápidas).
    """
    with httpx.Client(timeout=20.0) as http:
        resp = http.get(image_url)
        resp.raise_for_status()
        content = resp.content
    return detect_emotion_from_bytes(content)
