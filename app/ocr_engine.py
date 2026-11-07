import os
import shutil
import cv2
import numpy as np
import pytesseract
from app.image_processor import preprocess_variants

# Ensure Tesseract binary path is configured
tesseract_bin = shutil.which("tesseract")
if not tesseract_bin:
    for candidate in [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/usr/bin/tesseract",
    ]:
        if os.path.exists(candidate):
            tesseract_bin = candidate
            break

if tesseract_bin:
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin


def clean_ocr_text(text: str) -> str:
    """Normalize OCR text: standardize Unicode spaces and punctuation."""
    if not text:
        return ""

    replacements = {
        "\r\n": "\n",
        "\r": "\n",
        "\u3000": " ",
        "\u00a0": " ",
        "\uff1a": ":",
        "\uff1b": ";",
        "|": " ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    cleaned_lines = []
    for line in text.split("\n"):
        line_stripped = " ".join(line.split())
        if line_stripped:
            cleaned_lines.append(line_stripped)

    return "\n".join(cleaned_lines)


def run_tesseract(
    image: np.ndarray,
    lang: str = "ben+eng",
    psm: int = 6
) -> str:
    """Run Tesseract OCR with specified language and page segmentation mode."""
    config = f"--oem 1 --psm {psm}"
    try:
        raw = pytesseract.image_to_string(image, lang=lang, config=config)
        return clean_ocr_text(raw)
    except Exception:
        return ""


def extract_text_robust(image: np.ndarray) -> str:
    """
    Multi-Engine Dedicated Extraction Pipeline:
    1. Dedicated English Pass (lang='eng') on Green channel:
       Suppress cyan/green security watermarks and read Name, DOB, NID No
       with zero Bengali character confusion.
    2. Dedicated Bengali Pass (lang='ben') on Clean Grayscale:
       Read Bengali Name (নাম), Father's Name (পিতা), Mother's Name (মাতা)
       with zero Latin character interference.
    3. Mixed Pass (lang='ben+eng') on Clean Grayscale & Normalized:
       Capture full context and card headers.
    4. Merges all unique lines into a unified OCR document.
    """
    variants = preprocess_variants(image)

    # 1. English passes (Green channel suppresses green/cyan security patterns)
    eng_green_psm6 = run_tesseract(variants["green_channel"], lang="eng", psm=6)
    eng_green_psm11 = run_tesseract(variants["green_channel"], lang="eng", psm=11)
    eng_gray = run_tesseract(variants["clean_gray"], lang="eng", psm=6)

    # 2. Bengali passes (Grayscale preserves Bengali matras and ligatures)
    ben_gray_psm6 = run_tesseract(variants["clean_gray"], lang="ben", psm=6)
    ben_gray_psm11 = run_tesseract(variants["clean_gray"], lang="ben", psm=11)

    # 3. Mixed passes
    mixed_gray = run_tesseract(variants["clean_gray"], lang="ben+eng", psm=6)
    mixed_norm = run_tesseract(variants["normalized"], lang="ben+eng", psm=6)

    # Merge distinct informative lines
    seen_lines = set()
    combined_lines = []

    # Priority order: English lines for English fields, Bengali lines for Bengali fields
    sources = [
        ("eng", eng_green_psm6),
        ("eng", eng_green_psm11),
        ("eng", eng_gray),
        ("ben", ben_gray_psm6),
        ("ben", ben_gray_psm11),
        ("mixed", mixed_gray),
        ("mixed", mixed_norm),
    ]

    for _, text_block in sources:
        for line in text_block.split("\n"):
            normalized = line.strip().lower()
            # Ignore one-character noise or repeated lines
            if len(normalized) >= 2 and normalized not in seen_lines:
                seen_lines.add(normalized)
                combined_lines.append(line.strip())

    return "\n".join(combined_lines)


def extract_text(
    image: np.ndarray,
    lang: str = "ben+eng"
) -> str:
    """
    Extract text from NID image.
    If a color or raw card image is passed, executes the robust multi-engine pipeline.
    """
    if len(image.shape) == 3 or image.shape[0] > 400:
        return extract_text_robust(image)

    # Fallback for 2D preprocessed snippet
    text = run_tesseract(image, lang=lang, psm=6)
    if len(text.strip()) < 25:
        alt = run_tesseract(image, lang=lang, psm=11)
        if len(alt.strip()) > len(text.strip()):
            text = alt
    return text


def extract_text_multi_pass(
    raw_image: np.ndarray,
    lang: str = "ben+eng"
) -> str:
    """Multi-pass entry point."""
    return extract_text_robust(raw_image)
