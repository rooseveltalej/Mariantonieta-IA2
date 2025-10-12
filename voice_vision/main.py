# voice_vision/main.py
from __future__ import annotations
from pathlib import Path
from typing import List
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
from face_recognition import AzureFaceClient  


app = FastAPI(title="Maria Antonieta Face API", version="0.1.0")

# CORS abierto en desarrollo (ajusta dominios en prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instancia el cliente y, si se puede, crea/asegura el PersonGroup
client = AzureFaceClient()
try:
    client.ensure_person_group()
except Exception:
    # Si la suscripción/región no soporta Identify, el cliente quedará en modo detección-only.
    pass


# ------------------------ Modelos de petición ------------------------
class EnrollRequest(BaseModel):
    person_name: str = Field(..., min_length=1)
    images: List[str] = Field(
        ..., min_items=1, description="URLs públicas (3–10 recomendado) de la misma persona"
    )


class IdentifyRequest(BaseModel):
    image: str


# ------------------------ Endpoints ------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "personGroup": client.person_group_id,
        "supportsIdentity": getattr(client, "supports_identity", True),
    }


@app.post("/faces/detect")
def detect(req: IdentifyRequest):
    """
    Detección de rostros (siempre disponible). Devuelve bbox y faceId por rostro.
    """
    try:
        faces = client.detect(req.image, return_face_id=True)
        return {"faces": faces}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/faces/persons")
def list_persons():
    """
    Lista de personas enroladas. En modo detección-only devolverá [].
    """
    try:
        return client.list_persons()
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/faces/enroll")
def enroll(req: EnrollRequest):
    """
    Enrolar persona (requiere Identify). En detección-only devolverá 501.
    """
    if not getattr(client, "supports_identity", True):
        raise HTTPException(
            status_code=501,
            detail="Identification is disabled for this subscription/region. Modo detección-only.",
        )
    try:
        person_id = client.create_person(req.person_name)
        for url in req.images:
            client.add_person_face(person_id, url)
        client.train_and_wait()
        return {"personId": person_id, "enrolledFaces": len(req.images), "trained": True}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/faces/identify")
def identify(req: IdentifyRequest):
    """
    Identificar persona (si Identify está disponible). En detección-only retorna detecciones y un mensaje.
    """
    try:
        faces = client.detect(req.image, return_face_id=True)
        if not faces:
            return {"faces": [], "results": [], "message": "No se detectaron rostros"}

        # Si no hay Identify disponible, devolvemos solo detecciones
        if not getattr(client, "supports_identity", True):
            return {
                "faces": faces,
                "results": [],
                "message": "Identification unavailable: modo detección-only",
            }

        face_ids = [f["faceId"] for f in faces if "faceId" in f]
        results = client.identify(face_ids)

        enriched = []
        for f, r in zip(faces, results):
            cand = (r.get("candidates") or [])
            best = cand[0] if cand else None
            enriched.append(
                {
                    "faceRectangle": f.get("faceRectangle"),
                    "faceId": f.get("faceId"),
                    "bestCandidate": best,  # {personId, confidence} o None
                }
            )
        return {"results": enriched}
    except httpx.HTTPStatusError as he:
        raise HTTPException(status_code=he.response.status_code, detail=he.response.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
