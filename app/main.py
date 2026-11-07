from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from app.image_processor import preprocess_image
from app.ocr_engine import extract_text, extract_text_multi_pass
from app.nid_extractor import extract_nid_data


app = FastAPI(
    title="Sonali Intelligent KYC - NID OCR API",
    description="Bangladesh National ID (NID) OCR and automated KYC data extraction service",
    version="1.0.0"
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "application/octet-stream"
}


def decode_image_bytes(contents: bytes) -> np.ndarray:
    """Safely decode image bytes into OpenCV format."""
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Could not decode image."
        )
    return image


def process_single_image(image: np.ndarray):
    """
    Run multi-engine OCR and field extraction on an NID card image.
    Uses dedicated English and Bengali passes on clean color/grayscale representations
    to prevent background hologram watermarks from corrupting printed card text.
    """
    # Multi-engine extraction pipeline
    extracted_text = extract_text_multi_pass(image)

    if not extracted_text:
        # Fallback to single preprocessed pass
        processed = preprocess_image(image)
        extracted_text = extract_text(processed)

    # Extract structured fields
    nid_data = extract_nid_data(extracted_text)

    return nid_data, extracted_text


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "Sonali Intelligent KYC - NID OCR API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "ocr_single": "/api/v1/nid/ocr",
            "ocr_dual": "/api/v1/nid/ocr/dual"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "nid-ocr",
        "version": "1.0.0"
    }


@app.post("/api/v1/nid/ocr")
async def process_nid(
    file: UploadFile = File(...)
):
    """
    Process a single Bangladeshi NID card image (Front or Back side).
    Returns structured KYC information and extracted raw text.
    """
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file.content_type}'. Supported: JPG, PNG, WEBP."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB."
        )

    try:
        image = decode_image_bytes(contents)
        nid_data, extracted_text = process_single_image(image)

        if not extracted_text:
            raise HTTPException(
                status_code=422,
                detail="Unable to extract text from this image. Please provide a clearer photo."
            )

        return {
            "success": True,
            "message": "NID OCR completed successfully",
            "data": nid_data,
            "raw_text": extracted_text
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"OCR processing failed: {str(e)}"
        )


@app.post("/api/v1/nid/ocr/dual")
async def process_dual_nid(
    front_file: UploadFile = File(...),
    back_file: Optional[UploadFile] = File(None)
):
    """
    Process both Front and Back sides of a Bangladeshi NID card for complete Bank KYC.
    Combines personal identity fields (Front) with address and verification fields (Back).
    """
    front_contents = await front_file.read()
    if not front_contents:
        raise HTTPException(status_code=400, detail="Front image file is empty.")

    try:
        front_image = decode_image_bytes(front_contents)
        front_data, front_text = process_single_image(front_image)

        back_data = None
        back_text = None

        if back_file:
            back_contents = await back_file.read()
            if back_contents:
                back_image = decode_image_bytes(back_contents)
                back_data, back_text = process_single_image(back_image)

        # Merge front and back data
        merged = dict(front_data)
        if back_data:
            for key in ["address", "blood_group", "place_of_birth", "issue_date"]:
                if back_data.get(key) and not merged.get(key):
                    merged[key] = back_data[key]
            merged["card_side"] = "both"

        return {
            "success": True,
            "message": "Dual-side NID OCR completed successfully",
            "data": merged,
            "raw_text": {
                "front": front_text,
                "back": back_text
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dual NID OCR processing failed: {str(e)}"
        )
