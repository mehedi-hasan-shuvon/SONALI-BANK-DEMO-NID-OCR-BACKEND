import cv2
import numpy as np
import pytesseract


def auto_rotate_image(image: np.ndarray) -> np.ndarray:
    """
    Ensure the image is in upright landscape orientation.
    Only rotates 90 or 270 degrees if the image was photographed vertically (h > w * 1.15).
    Never rotates 180 degrees to prevent flipping correctly oriented cards upside down.
    """
    try:
        h, w = image.shape[:2]
        # Only rotate if image is in portrait orientation (height > width)
        if h > w * 1.15:
            osd = pytesseract.image_to_osd(image)
            rot_angle = 0
            for line in osd.split("\n"):
                if line.startswith("Rotate:"):
                    rot_angle = int(line.split(":")[1].strip())
                    break

            if rot_angle == 90:
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rot_angle == 270:
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image
    except Exception:
        h, w = image.shape[:2]
        if h > w * 1.15:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image


def crop_card_area(image: np.ndarray) -> np.ndarray:
    """
    Detect the rectangular NID card contour if photographed against a background
    (such as a desk, tablecloth, or paper) and crop to the card boundary.
    """
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 40, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        max_area = 0
        min_card_area = (h * w) * 0.20

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_card_area:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if 1.2 <= aspect <= 2.0 and area > max_area:
                    max_area = area
                    best_box = (x, y, bw, bh)

        if best_box:
            x, y, bw, bh = best_box
            mx = int(bw * 0.02)
            my = int(bh * 0.02)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + bw + mx)
            y2 = min(h, y + bh + my)
            return image[y1:y2, x1:x2]

        return image
    except Exception:
        return image


def resize_image(image: np.ndarray, target_width: int = 1800) -> np.ndarray:
    """Resize image to sweet-spot width (~1800px) for optimal Tesseract font recognition."""
    h, w = image.shape[:2]
    if w < 1200 or w > 2400:
        scale = target_width / float(w)
        new_h = int(h * scale)
        return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    return image


def add_border(image: np.ndarray, padding: int = 35) -> np.ndarray:
    """Add white margin around image to prevent Tesseract from clipping edge text."""
    return cv2.copyMakeBorder(
        image,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=255
    )


def get_clean_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to clean grayscale with edge-preserving bilateral filtering."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    norm = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return add_border(norm, padding=35)


def get_green_channel(image: np.ndarray) -> np.ndarray:
    """Extract Green color channel to suppress cyan/green security watermarks."""
    if len(image.shape) == 3:
        green = image[:, :, 1]
    else:
        green = image

    denoised = cv2.bilateralFilter(green, d=7, sigmaColor=50, sigmaSpace=50)
    norm = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return add_border(norm, padding=35)


def get_illumination_normalized(image: np.ndarray) -> np.ndarray:
    """Remove shadows or room lighting across the ID card."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    background = cv2.medianBlur(background, 21)

    normalized = cv2.divide(gray, background, scale=255)
    return add_border(normalized, padding=35)


def get_back_card_crops(image: np.ndarray) -> dict:
    """
    Generate targeted crops for Bangladeshi Smart NID back side:
    - Address Box (middle-left, upscaled & thresholded to clear green watermark)
    - Details Line (middle, blood group, place of birth, issue date)
    - MRZ Zone (bottom 38%, ICAO TD1 machine readable zone)
    """
    h, w = image.shape[:2]

    # Address Box
    addr_crop = image[int(h * 0.25):int(h * 0.58), int(w * 0.02):int(w * 0.68)]
    gray_addr = cv2.cvtColor(addr_crop, cv2.COLOR_BGR2GRAY) if len(addr_crop.shape) == 3 else addr_crop
    scaled_addr = cv2.resize(gray_addr, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
    _, bin_addr = cv2.threshold(scaled_addr, 135, 255, cv2.THRESH_BINARY)
    pad_addr = cv2.copyMakeBorder(bin_addr, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)

    # Details Line
    details_crop = image[int(h * 0.47):int(h * 0.65), :]
    gray_det = cv2.cvtColor(details_crop, cv2.COLOR_BGR2GRAY) if len(details_crop.shape) == 3 else details_crop
    scaled_det = cv2.resize(gray_det, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    pad_det = cv2.copyMakeBorder(scaled_det, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)

    # MRZ Zone
    mrz_crop = image[int(h * 0.60):, :]
    gray_mrz = cv2.cvtColor(mrz_crop, cv2.COLOR_BGR2GRAY) if len(mrz_crop.shape) == 3 else mrz_crop
    scaled_mrz = cv2.resize(gray_mrz, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    pad_mrz = cv2.copyMakeBorder(scaled_mrz, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)

    return {
        "address": pad_addr,
        "details": pad_det,
        "mrz": pad_mrz,
    }


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Primary image preprocessing entry point."""
    oriented = auto_rotate_image(image)
    cropped = crop_card_area(oriented)
    resized = resize_image(cropped, target_width=1800)
    return get_clean_grayscale(resized)


def preprocess_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    """Generate multiple preprocessed variations for multi-engine OCR."""
    oriented = auto_rotate_image(image)
    cropped = crop_card_area(oriented)
    resized = resize_image(cropped, target_width=1800)

    clean_gray = get_clean_grayscale(resized)
    green_ch = get_green_channel(resized)
    normalized = get_illumination_normalized(resized)

    _, otsu = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "clean_gray": clean_gray,
        "green_channel": green_ch,
        "normalized": normalized,
        "otsu": otsu,
        "raw_processed": resized
    }
