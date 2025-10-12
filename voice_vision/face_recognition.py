# voice_vision/face_recognition.py
from __future__ import annotations
import os, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

class AzureFaceClient:
    def __init__(self, endpoint: Optional[str] = None, key: Optional[str] = None, person_group_id: Optional[str] = None):
        ep = endpoint or os.getenv("AZURE_FACE_ENDPOINT") or ""
        k  = key or os.getenv("AZURE_FACE_KEY") or ""
        pg = person_group_id or os.getenv("AZURE_FACE_PERSON_GROUP_ID") or "jarvistec_group"
        if not ep or not k:
            raise RuntimeError("Faltan AZURE_FACE_ENDPOINT o AZURE_FACE_KEY (revisa tu .env).")

        self.endpoint = ep.rstrip("/")
        self.key = k
        self.person_group_id = pg
        self.headers_json = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/json"
        }

        # Flag: detección-only si pones AZURE_FACE_DETECTION_ONLY=true en .env
        self.supports_identity = str(os.getenv("AZURE_FACE_DETECTION_ONLY", "false")).lower() not in ("1", "true", "yes")

    # ---------- Detección (URL) ----------
    def detect(self, image_url: str, return_face_id: bool = True) -> List[Dict[str, Any]]:
        url = f"{self.endpoint}/face/v1.0/detect"
        params = {
            "returnFaceId": str(return_face_id).lower(),
            "returnFaceLandmarks": "false",
            "recognitionModel": "recognition_04",
        }
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, params=params, headers=self.headers_json, json={"url": image_url})
            r.raise_for_status()
            return r.json()

    # ---------- Detección (bytes) ----------
    def detect_bytes(self, content: bytes, return_face_id: bool = True) -> List[Dict[str, Any]]:
        """
        Detecta rostros a partir de bytes (p.ej., foto de la webcam enviada por multipart/form-data).
        """
        url = f"{self.endpoint}/face/v1.0/detect"
        params = {
            "returnFaceId": str(return_face_id).lower(),
            "returnFaceLandmarks": "false",
            "recognitionModel": "recognition_04",
        }
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Content-Type": "application/octet-stream",
        }
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, params=params, headers=headers, content=content)
            r.raise_for_status()
            return r.json()

    # ---------- Person Group ----------
    def ensure_person_group(self, name: str = "JarvisTEC Group") -> None:
        if not self.supports_identity:
            return
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}"
        payload = {"name": name, "recognitionModel": "recognition_04"}
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.put(url, headers=self.headers_json, json=payload)
                if r.status_code not in (200, 202, 409):  # 409 = ya existe
                    r.raise_for_status()
        except httpx.HTTPStatusError as he:
            if he.response.status_code == 403:
                # La suscripción no tiene habilitada Identify (person groups / identify)
                self.supports_identity = False
                return
            raise

    def train_person_group(self) -> None:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/train"
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self.headers_json)
            r.raise_for_status()

    def get_training_status(self) -> Dict[str, Any]:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/training"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, headers=self.headers_json)
            r.raise_for_status()
            return r.json()

    def create_person(self, name: str) -> str:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/persons"
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self.headers_json, json={"name": name})
            r.raise_for_status()
            return r.json()["personId"]

    def add_person_face(self, person_id: str, image_url: str) -> str:
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/persons/{person_id}/persistedFaces"
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self.headers_json, json={"url": image_url})
            r.raise_for_status()
            return r.json()["persistedFaceId"]

    def list_persons(self) -> List[Dict[str, Any]]:
        if not self.supports_identity:
            return []  # en modo detección-only devolvemos lista vacía
        url = f"{self.endpoint}/face/v1.0/persongroups/{self.person_group_id}/persons"
        with httpx.Client(timeout=30.0) as client:
            r = client.get(url, headers=self.headers_json)
            r.raise_for_status()
            return r.json()

    # ---------- Verify (1:1, sin PersonGroup) ----------
    def verify_faces(self, face_id1: str, face_id2: str) -> Dict[str, Any]:
        """
        Verifica si dos faceId pertenecen a la misma persona.
        Respuesta: {"isIdentical": bool, "confidence": float}
        """
        url = f"{self.endpoint}/face/v1.0/verify"
        payload = {"faceId1": face_id1, "faceId2": face_id2}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self.headers_json, json=payload)
            r.raise_for_status()
            return r.json()

    # ---------- Identify (con PersonGroup) ----------
    def identify(self, face_ids: List[str], max_candidates: int = 1, confidence_threshold: float = 0.65):
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        url = f"{self.endpoint}/face/v1.0/identify"
        payload = [{
            "faceIds": face_ids,
            "personGroupId": self.person_group_id,
            "maxNumOfCandidatesReturned": max_candidates,
            "confidenceThreshold": confidence_threshold
        }]
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, headers=self.headers_json, json=payload)
            r.raise_for_status()
            return r.json()

    def train_and_wait(self, poll_seconds: float = 1.0, timeout_seconds: float = 60.0):
        if not self.supports_identity:
            raise RuntimeError("IDENTIFICATION_UNAVAILABLE")
        self.train_person_group()
        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.get_training_status()
            if status.get("status") == "succeeded":
                return "succeeded"
            if status.get("status") == "failed":
                raise RuntimeError(f"Entrenamiento falló: {status}")
            time.sleep(poll_seconds)
        raise TimeoutError("Timeout esperando entrenamiento.")
