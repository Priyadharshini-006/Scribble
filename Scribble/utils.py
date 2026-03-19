import numpy as np
import pytesseract
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None


# FIX 4 & 5: Original enhance_image() used only PIL (ignored opencv-python in requirements).
# Original extract_text() was a stub returning hardcoded fake text — no real OCR ran.
# Both functions are now fully implemented with OpenCV fallback.


def enhance_image(image):
    """
    Enhance a numpy image array for better OCR accuracy.

    Uses OpenCV if available, otherwise Pillow fallback.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if cv2 is not None:
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(contrast, h=15, templateWindowSize=7, searchWindowSize=21)
        binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=10)
        return binary

    pil = Image.fromarray(image)
    pil = ImageOps.grayscale(pil)
    pil = ImageOps.autocontrast(pil)
    pil = pil.point(lambda p: 255 if p > 128 else 0)
    return np.array(pil)


def extract_text(image):
    """
    Extract text from an enhanced numpy image using Tesseract OCR.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if cv2 is not None:
        padded = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    else:
        pil = Image.fromarray(image)
        padded = ImageOps.expand(pil, border=20, fill='white')
        padded = np.array(padded)

    custom_config = '--oem 3 --psm 6'
    try:
        raw = pytesseract.image_to_string(padded, config=custom_config)
    except pytesseract.pytesseract.TesseractNotFoundError:
        # Fallback text when tesseract executable is unavailable
        return "[OCR unavailable: install Tesseract OCR and add to PATH]"

    lines = [line.strip() for line in raw.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned
