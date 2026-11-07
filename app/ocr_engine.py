"""
OCR layer: runs several Tesseract passes per card zone and returns each pass as its own
document (never merged). nid_extractor parses every document separately and votes.
"""
import os
import re
import shutil
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract

from app import image_processor as ip
from app import paddle_engine

# One thread per Tesseract process; we parallelise across passes instead.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

tesseract_bin = shutil.which("tesseract")
if not tesseract_bin:
    for candidate in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
        if os.path.exists(candidate):
            tesseract_bin = candidate
            break
if tesseract_bin:
    pytesseract.pytesseract.tesseract_cmd = tesseract_bin

MAX_WORKERS = int(os.environ.get("OCR_WORKERS", "4"))
BLUR_THRESHOLD = 60.0          # Laplacian variance on the normalised card; tune on your own photos
MRZ_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
DEBUG_DIR = os.environ.get("NID_DEBUG_DIR")      # set to a folder to dump the oriented card + crops


def _debug(name: str, img: np.ndarray) -> None:
    if DEBUG_DIR:
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}.png"), img)
        except Exception:
            pass


def clean_ocr_text(text: str) -> str:
    """Normalise Unicode (NFC) and whitespace. '|' is kept: it is often a misread '1'."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    for old, new in {"\r\n": "\n", "\r": "\n", "　": " ", " ": " ", "：": ":",
                     "；": ";", "‘": "'", "’": "'", "“": '"', "”": '"'}.items():
        text = text.replace(old, new)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(l for l in lines if l)


def run_tesseract(image: np.ndarray, lang: str = "ben+eng", psm: int = 6,
                  whitelist: Optional[str] = None) -> str:
    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    try:
        return clean_ocr_text(pytesseract.image_to_string(image, lang=lang, config=config))
    except Exception:
        return ""


def run_jobs(jobs: List[Tuple[np.ndarray, str, int, Optional[str]]]) -> List[str]:
    """Run (image, lang, psm, whitelist) jobs in parallel, keeping the input order."""
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_tesseract, *job) for job in jobs]
        return [f.result() for f in futures]


# --------------------------------------------------------------------------- #
# orientation + side detection (one cheap English pass)
# --------------------------------------------------------------------------- #
FRONT_MARKERS = ("DATE OF BIRTH", "NID", "NATIONAL ID", "GOVERNMENT", "REPUBLIC", "PEOPLE", "NAME")
BACK_MARKERS = ("BLOOD", "GROUP", "PLACE OF", "ISSUE", "BGD", "<<")


def quick_scan(card: np.ndarray, width: int = 1500) -> Tuple[str, List[Dict[str, Any]]]:
    """Small English-only pass returning the text and word positions (as fractions)."""
    gray = ip.to_gray(card)
    h, w = gray.shape[:2]
    if w > width:
        gray = cv2.resize(gray, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    H, W = gray.shape[:2]
    try:
        d = pytesseract.image_to_data(gray, lang="eng", config="--oem 1 --psm 11",
                                      output_type=pytesseract.Output.DICT)
    except Exception:
        return "", []
    words = []
    for i, word in enumerate(d["text"]):
        word = (word or "").strip()
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not word or conf < 0:
            continue
        words.append({"text": word, "conf": conf,
                      "cx": (d["left"][i] + d["width"][i] / 2) / W,
                      "cy": (d["top"][i] + d["height"][i] / 2) / H})
    return " ".join(w["text"] for w in words), words


def score_side(text: str) -> Tuple[int, int]:
    up = text.upper()
    return (sum(1 for m in FRONT_MARKERS if m in up), sum(1 for m in BACK_MARKERS if m in up))


def readability(words: List[Dict[str, Any]]) -> float:
    """How much real Latin text Tesseract found. An upside-down card gives almost none, so this
    picks the right rotation even when no card keyword is recognised (typical for the back)."""
    return sum(w["conf"] / 100.0 for w in words
               if w["conf"] >= 60 and len(w["text"]) >= 3 and re.search(r"[A-Za-z]{3}", w["text"]))


def orient_and_classify(card: np.ndarray, expected_side: Optional[str] = None
                        ) -> Tuple[np.ndarray, str, List[Dict[str, Any]]]:
    """Try the plausible rotations; keep the one that is most readable / has the most card keywords."""
    best = None
    for cand in ip.rotation_candidates(card):
        text, words = quick_scan(cand)
        f, b = score_side(text)
        score = 3 * (f + b) + readability(words)
        if best is None or score > best[0] + 1e-9:
            best = (score, cand, f, b, words)
        if f + b >= 3:
            break
    _, cand, f, b, words = best
    side = "front" if f > b else "back" if b > f else "unknown"
    if side == "unknown" and expected_side in ("front", "back"):
        side = expected_side          # caller knows which upload slot this image came from
    return cand, side, words


# --------------------------------------------------------------------------- #
# zone OCR
# --------------------------------------------------------------------------- #
def locate_front_rows(words: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Centre y (fraction of card height) of the Bengali-name / father / mother rows, derived from
    the English 'Name' and 'Date of Birth' labels. Rows are evenly spaced between those two."""
    def tok(w): return re.sub(r"[^a-z]", "", w["text"].lower())
    names = [w["cy"] for w in words if tok(w) in ("name", "narne", "nane", "mame") and 0.26 <= w["cx"] <= 0.55
             and 0.25 <= w["cy"] <= 0.65]
    if not names:
        return None
    name_y = min(names)
    dobs = [w["cy"] for w in words if tok(w) in ("date", "dale", "oate") and 0.26 <= w["cx"] <= 0.55
            and w["cy"] > name_y + 0.08]
    if not dobs:
        return None
    dob_y = min(dobs)
    step = (dob_y - name_y) / 3.0
    if not 0.04 <= step <= 0.12:
        return None
    return {"name_bn": name_y - step, "father": name_y + step, "mother": name_y + 2 * step, "_step": step}


def _row_variants(crop: np.ndarray) -> List[np.ndarray]:
    crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = ip.clean_gray(crop)
    return [gray, ip.sharpen(gray), ip.add_border(ip.adaptive_binary(ip.to_gray(crop)))]


def ocr_front(card: np.ndarray, words: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[str]]:
    """
    'column' passes read the whole text column; 'full' is a fallback. When the row anchors are
    found, 'row_*' passes OCR father / mother / bn-name as SINGLE-LINE crops (psm 7): the row
    identity then comes from position, so a lost 'পিতা' / 'মাতা' label cannot shift values.
    """
    col = ip.front_text_column(card)
    gray = ip.clean_gray(col)
    sharp = ip.sharpen(gray)
    adaptive = ip.add_border(ip.adaptive_binary(ip.to_gray(col)))
    green = ip.green_channel(col)
    _debug("front_column", col)

    jobs = [
        (gray, "ben+eng", 6, None),
        (gray, "ben+eng", 4, None),
        (sharp, "ben+eng", 6, None),
        (adaptive, "ben+eng", 6, None),
        (green, "eng", 6, None),                # clean source for EN name / DOB / NID digits
        (ip.clean_gray(card), "ben+eng", 6, None),
    ]
    row_keys: List[str] = []
    rows = locate_front_rows(words or [])
    if rows:
        half = rows["_step"] * 0.52
        for key in ("name_bn", "father", "mother"):
            crop = ip.crop_rel(card, 0.29, rows[key] - half, 0.76, rows[key] + half)
            _debug(f"front_row_{key}", crop)
            for v in _row_variants(crop):
                jobs.append((v, "ben", 7, None))
                row_keys.append(key)

    out = run_jobs(jobs)
    result: Dict[str, List[str]] = {"column": out[:5], "full": out[5:6]}
    for key, text in zip(row_keys, out[6:]):
        result.setdefault(f"row_{key}", []).append(text)

    if paddle_engine.enabled():                 # optional VLM pass, most trusted -> first
        result["column"].insert(0, paddle_engine.read_text(col))
    return result


def locate_back_rows(words: List[Dict[str, Any]]) -> Optional[float]:
    """y (fraction of card height) of the 'Blood Group / Place of Birth / Issue Date' row."""
    ys = sorted(w["cy"] for w in words
                if any(k in w["text"].upper() for k in ("BLOOD", "GROUP", "ISSUE", "PLACE")))
    return ys[len(ys) // 2] if ys else None


def ocr_back(card: np.ndarray, words: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    dy = locate_back_rows(words)
    if dy is not None:
        # anchored to the details row, so card margins / framing do not matter
        addr_box = (0.02, dy - 0.28, 0.68, dy - 0.02)
        det_box = (0.02, dy - 0.05, 0.99, dy + 0.06)
        mrz_box = (0.02, dy + 0.07, 0.99, min(dy + 0.42, 1.0))
    else:
        addr_box = (0.02, 0.24, 0.68, 0.54)
        det_box = (0.02, 0.48, 0.99, 0.66)
        mrz_box = (0.02, 0.62, 0.99, 0.98)

    addr = ip.crop_rel(card, *addr_box)
    det = ip.crop_rel(card, *det_box)
    mrz = ip.crop_rel(card, *mrz_box)

    addr_gray = ip.clean_gray(addr)
    addr_green = ip.green_channel(addr)
    det_gray = ip.clean_gray(det)
    det_sharp = ip.sharpen(det_gray)
    mrz_hard, mrz_soft = ip.mrz_variants(mrz)
    _debug("back_address", addr); _debug("back_details", det); _debug("back_mrz", mrz)

    jobs = [
        (addr_gray, "ben+eng", 6, None),
        (ip.sharpen(addr_gray), "ben+eng", 6, None),
        (addr_green, "ben+eng", 6, None),
        (det_gray, "ben+eng", 6, None),
        (det_sharp, "eng", 6, None),
        (mrz_hard, "eng", 6, MRZ_WHITELIST),
        (mrz_soft, "eng", 6, MRZ_WHITELIST),
        (mrz_hard, "eng", 4, MRZ_WHITELIST),
        (ip.clean_gray(card), "ben+eng", 6, None),
    ]
    out = run_jobs(jobs)
    res = {"address": out[0:3], "details": out[3:5], "mrz": out[5:8], "full": out[8:9]}
    if paddle_engine.enabled():                 # MRZ stays Tesseract-only (whitelist + check digits)
        res["address"].insert(0, paddle_engine.read_text(addr))
        res["details"].insert(0, paddle_engine.read_text(det))
    return res


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def analyze_card(image: np.ndarray, expected_side: Optional[str] = None) -> Dict[str, Any]:
    card = ip.prepare_card(image)
    blur = ip.blur_score(card)
    card, side, words = orient_and_classify(card, expected_side)
    _debug("oriented_card", card)

    docs: Dict[str, Dict[str, List[str]]] = {}
    if side in ("front", "unknown"):
        docs["front"] = ocr_front(card, words)
    if side in ("back", "unknown"):
        docs["back"] = ocr_back(card, words)

    return {
        "side": side,
        "docs": docs,
        "quality": {"blur_score": round(blur, 1), "is_blurry": blur < BLUR_THRESHOLD},
    }