"""
Optional PaddleOCR-VL pass (Bengali-capable). Disabled unless USE_PADDLE_VL=1.

NOTE: classic PaddleOCR (PP-OCRv5 det+rec) has NO Bengali recogniser. Bengali is covered
by PaddleOCR-VL (a 0.9B vision-language model) - heavier, slower, but reads Bengali.
Every failure here returns "" so the Tesseract pipeline keeps working on its own.

    pip install paddlepaddle "paddleocr[doc-parser]"
    USE_PADDLE_VL=1 uvicorn app.main:app
"""
import os
import re
import threading
from typing import Any, List, Optional

import cv2
import numpy as np

_pipeline: Any = None
_failed = False
_lock = threading.Lock()          # the VL pipeline is not safe to share between threads


def enabled() -> bool:
    return os.environ.get("USE_PADDLE_VL", "0") == "1" and not _failed


def _get():
    global _pipeline, _failed
    with _lock:
        if _pipeline is None and not _failed:
            try:
                from paddleocr import PaddleOCRVL      # imported lazily: heavy dependency
                _pipeline = PaddleOCRVL()
            except Exception as e:                      # not installed / model download failed
                print(f"[paddle_engine] disabled: {e}")
                _failed = True
        return _pipeline


def _blocks(res: Any) -> List[str]:
    """Pull text out of a PaddleOCR-VL result. The result type differs between releases,
    so try the known shapes in turn."""
    md = getattr(res, "markdown", None)
    if isinstance(md, dict) and md.get("markdown_texts"):
        return [str(md["markdown_texts"])]
    try:
        items = res["parsing_res_list"]
        out = [getattr(b, "content", None) or (b.get("block_content") if isinstance(b, dict) else None)
               for b in items]
        out = [o for o in out if o]
        if out:
            return out
    except Exception:
        pass
    return [str(res)] if res is not None else []


def read_text(image: np.ndarray) -> str:
    pipe = _get() if enabled() else None
    if pipe is None:
        return ""
    try:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        with _lock:
            results = list(pipe.predict(image))
        text = "\n".join(b for r in results for b in _blocks(r))
    except Exception as e:
        print(f"[paddle_engine] predict failed: {e}")
        return ""
    text = re.sub(r"<[^>]+>", "\n", text)                  # html tables / divs
    text = re.sub(r"^[#>*\-\s|]+", "", text, flags=re.M)   # markdown decoration
    text = text.replace("|", " ")
    return "\n".join(" ".join(l.split()) for l in text.split("\n") if l.strip())