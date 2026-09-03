"""SendCalyx Kidney CT Ensemble Analysis API."""

from __future__ import annotations

import io
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

# Allow `python app.py` from any working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_utils import PredictionResponse, load_model, predict_all_models  # noqa: E402
from architectures import (  # noqa: E402
    CLASS_NAMES,
    MODEL_DISPLAY_NAMES,
    MODEL_INPUT_SIZE,
    StackedEnsembleNet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("sendcalyx")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

model: Optional[StackedEnsembleNet] = None
device = None


def allowed_origins() -> list[str]:
    """CORS origins, extendable through ``SENDCALYX_ALLOWED_ORIGINS`` (comma-separated)."""
    extra = os.environ.get("SENDCALYX_ALLOWED_ORIGINS", "")
    origins = list(DEFAULT_ORIGINS)
    origins.extend(origin.strip() for origin in extra.split(",") if origin.strip())
    return origins


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, device
    model, device = load_model()
    if model is None:
        logger.error("Model failed to load; /predict will return 503")
    yield
    model = None
    device = None
    logger.info("SendCalyx API shutting down")


app = FastAPI(
    title="SendCalyx Kidney CT Ensemble Analysis API",
    description=(
        "Research-oriented ensemble inference and explainability for kidney CT imagery."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Service identity and readiness."""
    return {
        "name": "SendCalyx Kidney CT Ensemble Analysis API",
        "description": (
            "Research-oriented ensemble inference and explainability for "
            "kidney CT imagery."
        ),
        "version": "0.1.0",
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "disclaimer": (
            "SendCalyx is a research and educational prototype and is not intended "
            "for clinical diagnosis or medical decision-making."
        ),
    }


@app.get("/health")
async def health_check():
    """Readiness detail, including which base models actually loaded."""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown",
        "num_base_models": len(model.base_models) if model else 0,
        "base_models": list(model.base_models) if model else [],
        "load_errors": dict(model.load_errors) if model else {},
    }


@app.get("/models")
async def get_models():
    """Describe the loaded ensemble."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "ensemble_architecture": "StackedEnsembleNet",
        "meta_learner": "6 -> 512 -> ReLU -> Dropout(0.2) -> 128 -> ReLU -> 2",
        "base_models": [
            {"id": model_id, "display_name": MODEL_DISPLAY_NAMES.get(model_id, model_id)}
            for model_id in model.base_models
        ],
        "classes": CLASS_NAMES,
        "model_input_size": list(MODEL_INPUT_SIZE),
        "device": str(device),
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Run the ensemble, consensus analysis, and cross-model attribution.

    Accepts a single JPEG, PNG, or WebP image of a kidney CT slice.
    """
    if model is None or device is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a JPEG, PNG, or WebP image.",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400, detail="Image could not be read or is corrupted."
        ) from None

    try:
        if image.format and image.format.upper() not in ALLOWED_PIL_FORMATS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image format: {image.format}.",
            )

        results = predict_all_models(
            model, device, image, file_size_bytes=len(image_bytes)
        )
    finally:
        image.close()

    if not results.get("success"):
        # predict_all_models already logged the traceback; keep it off the wire.
        raise HTTPException(
            status_code=500,
            detail=results.get("message", "Prediction failed."),
        )

    return JSONResponse(content=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SENDCALYX_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
