from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.nid_extractor import (
    build_back_result,
    build_front_result,
    finalize,
    merge_dual,
)
from app.ocr_engine import analyze_card

app = FastAPI(
    title="Sonali Intelligent KYC - NID OCR API",
    description="Bangladesh National ID (NID) OCR and automated KYC data extraction service",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "application/octet-stream"}


def decode_image_bytes(contents: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format. Could not decode image.")
    return image


def _raw_text(docs: Dict[str, Dict[str, list]]) -> str:
    """Readable dump of every OCR pass (useful when debugging a bad scan)."""
    parts = []
    for side, groups in docs.items():
        for group, texts in groups.items():
            for i, t in enumerate(texts):
                parts.append(f"[{side}/{group}#{i}]\n{t}")
    return "\n\n".join(parts)


def process_single_image(image: np.ndarray, expected_side: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Orient -> detect side -> zone OCR -> parse each pass -> vote -> validate."""
    scan = analyze_card(image, expected_side)
    docs, side = scan["docs"], scan["side"]

    candidates = []
    if "front" in docs:
        f = docs["front"]
        rows = {k[4:]: v for k, v in f.items() if k.startswith("row_")}
        candidates.append(finalize(build_front_result(f["column"] + f["full"], rows), "front"))
    if "back" in docs:
        candidates.append(finalize(build_back_result(docs["back"]), "back"))
    if not candidates:
        raise HTTPException(status_code=422, detail="Could not recognise this as an NID card image.")

    # side == 'unknown' produces two candidates: keep the more convincing one
    data = max(candidates, key=lambda r: r["validation"]["confidence_score"])
    data["quality"] = scan["quality"]
    if scan["quality"]["is_blurry"]:
        data["validation"]["warnings"].insert(0, "Image looks blurry; please re-take the photo in good light.")
        data["validation"]["needs_manual_review"] = True
    return data, _raw_text(docs)


async def _read_upload(file: UploadFile, label: str) -> bytes:
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file format '{file.content_type}'. Supported: JPG, PNG, WEBP.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail=f"{label} image file is empty.")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")
    return contents


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Sonali Intelligent KYC - NID OCR API",
        "version": "2.0.0",
        "documentation": "/docs",
        "endpoints": {"health": "/health", "ocr_single": "/api/v1/nid/ocr", "ocr_dual": "/api/v1/nid/ocr/dual"},
    }


@app.get("/health")
def health():
    return {"status": "healthy", "service": "nid-ocr", "version": "2.0.0"}


@app.post("/api/v1/nid/ocr")
async def process_nid(file: UploadFile = File(...)):
    """Process a single NID image (front or back; the side is detected automatically)."""
    contents = await _read_upload(file, "Uploaded")
    try:
        data, raw = process_single_image(decode_image_bytes(contents))
        return {"success": True, "message": "NID OCR completed successfully", "data": data, "raw_text": raw}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e}")


@app.post("/api/v1/nid/ocr/dual")
async def process_dual_nid(front_file: UploadFile = File(...), back_file: Optional[UploadFile] = File(None)):
    """Front + back for full KYC. The two sides are cross-checked through the back MRZ."""
    front_contents = await _read_upload(front_file, "Front")
    back_contents = await _read_upload(back_file, "Back") if back_file else None
    try:
        first, first_raw = process_single_image(decode_image_bytes(front_contents), "front")
        if back_contents is None:
            return {"success": True, "message": "NID OCR completed successfully",
                    "data": first, "raw_text": {"front": first_raw, "back": None}}

        second, second_raw = process_single_image(decode_image_bytes(back_contents), "back")
        front, back, front_raw, back_raw = first, second, first_raw, second_raw
        if first.get("card_side") == "back" and second.get("card_side") == "front":   # files swapped
            front, back, front_raw, back_raw = second, first, second_raw, first_raw

        if front.get("card_side") != "front" or back.get("card_side") != "back":
            merged = dict(front)
            merged["validation"] = dict(front["validation"])
            merged["validation"]["warnings"] = front["validation"]["warnings"] + [
                "Could not identify one front and one back image; results were not merged."]
            merged["validation"]["needs_manual_review"] = True
        else:
            merged = merge_dual(front, back)
            merged["quality"] = {"front": front.get("quality"), "back": back.get("quality")}

        return {"success": True, "message": "Dual-side NID OCR completed successfully",
                "data": merged, "raw_text": {"front": front_raw, "back": back_raw}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dual NID OCR processing failed: {e}")