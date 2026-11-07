"""
Image preparation for Bangladeshi NID cards.

Main ideas
----------
* Normalise every upload to the same size (long side = 1800 px) so that relative
  "zone" crops (text column, address block, MRZ ...) mean the same thing every time.
* Crop to the text areas BEFORE OCR. The photo, chip, hologram and security pattern are
  what turn blurry uploads into garbage lines; Tesseract never needs to see them.
* Produce several enhanced variants of each crop (plain, sharpened, adaptive-binary,
  green channel). Each variant is OCR'd separately and the results are voted on later.
"""
from typing import List, Tuple

import cv2
import numpy as np

CARD_LONG_SIDE = 1800


# --------------------------------------------------------------------------- #
# card detection / normalisation
# --------------------------------------------------------------------------- #
def crop_card_area(image: np.ndarray) -> np.ndarray:
    """Crop to the card if it was photographed on a desk/table; otherwise keep as is."""
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 120)
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_area = None, 0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < 0.25 * h * w:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if 1.3 <= aspect <= 1.9 and area > best_area:
                best, best_area = (x, y, bw, bh), area

        if best is None:
            return image
        x, y, bw, bh = best
        if bw * bh > 0.97 * h * w:          # already just the card
            return image
        pad = int(0.01 * max(bw, bh))
        return image[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)]
    except Exception:
        return image


def prepare_card(image: np.ndarray) -> np.ndarray:
    """Crop to card and rescale so the long side is exactly CARD_LONG_SIDE."""
    card = crop_card_area(image)
    h, w = card.shape[:2]
    scale = CARD_LONG_SIDE / float(max(h, w))
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(card, None, fx=scale, fy=scale, interpolation=interp)


def rotation_candidates(card: np.ndarray) -> List[np.ndarray]:
    """Orientations worth trying. The caller picks the one where card keywords are readable."""
    h, w = card.shape[:2]
    if w >= h:
        return [card, cv2.rotate(card, cv2.ROTATE_180)]
    return [cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)]


def crop_rel(card: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Crop using fractions (0..1) of the normalised card."""
    h, w = card.shape[:2]
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return card[int(h * y1):max(int(h * y2), int(h * y1) + 2), int(w * x1):max(int(w * x2), int(w * x1) + 2)]


def blur_score(card: np.ndarray) -> float:
    """Variance of the Laplacian: lower = blurrier."""
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (1000, int(gray.shape[0] * 1000 / gray.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# --------------------------------------------------------------------------- #
# enhancement variants
# --------------------------------------------------------------------------- #
def add_border(image: np.ndarray, padding: int = 30) -> np.ndarray:
    return cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def clean_gray(image: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise + contrast stretch. Keeps Bengali matras/ligatures intact."""
    gray = to_gray(image)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return add_border(gray)


def green_channel(image: np.ndarray) -> np.ndarray:
    """Green channel suppresses the green/cyan guilloche pattern on the card."""
    g = image[:, :, 1] if image.ndim == 3 else image
    g = cv2.bilateralFilter(g, d=7, sigmaColor=50, sigmaSpace=50)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return add_border(g)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask - helps slightly blurry uploads."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.6)
    return cv2.addWeighted(gray, 1.7, blur, -0.7, 0)


def adaptive_binary(gray: np.ndarray) -> np.ndarray:
    """Local thresholding copes with the pink/green gradient behind the text."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
    return binary


def clahe_gray(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return add_border(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))


def mrz_variants(crop: np.ndarray) -> List[np.ndarray]:
    """MRZ characters are small and thin: upscale, then give Tesseract a hard and a soft version."""
    gray = to_gray(crop)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, otsu = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [add_border(otsu), add_border(gray)]


def front_text_column(card: np.ndarray) -> np.ndarray:
    """
    Right-hand text column of the front: x 29-76 %, y 26-98 %.
    Excludes the photo (left), the chip (right) and the header/hologram (top).
    """
    return crop_rel(card, 0.29, 0.26, 0.76, 0.98)