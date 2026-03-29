# src/plate_reader.py — fast-plate-ocr version
import cv2
import re
from fast_plate_ocr import LicensePlateRecognizer
import config


class PlateReader:

    LETTER_TO_DIGIT = {
        'O': '0', 'I': '1', 'S': '5',
        'B': '8', 'Z': '2', 'G': '6', 'T': '7',
    }
    DIGIT_TO_LETTER = {
        '0': 'O', '1': 'I', '5': 'S',
        '8': 'B', '2': 'Z',
    }

    UK_FORMATS = [
        ('current', 7, {2, 3},    {0, 1, 4, 5, 6}),
        ('prefix',  7, {1, 2, 3}, {0, 4, 5, 6}),
        ('suffix',  7, {3, 4, 5}, {0, 1, 2, 6}),
    ]

    def __init__(self):
        print("[PlateReader] Loading fast-plate-ocr (european model)...")
        # Force CPUExecutionProvider — onnxruntime-gpu requires cublasLt64_12.dll
        # which is a CUDA 12.x library. CUDA 13.x drivers don't ship it.
        # CPU inference is still fast since the model is only 4.75MB.
        self.recognizer = LicensePlateRecognizer(
            'european-plates-mobile-vit-v2-model',
            providers=['CPUExecutionProvider']
        )
        print("[PlateReader] fast-plate-ocr loaded (CPU execution).")

    def preprocess(self, plate_img):
        gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.GaussianBlur(gray, (3, 3), 0)
        return gray

    def _detect_format(self, text: str):
        if len(text) != 7:
            return None
        for fmt in self.UK_FORMATS:
            name, length, digit_pos, letter_pos = fmt
            correct = 0
            for i in digit_pos:
                if text[i].isdigit() or text[i] in self.LETTER_TO_DIGIT:
                    correct += 1
            for i in letter_pos:
                if text[i].isalpha() or text[i] in self.DIGIT_TO_LETTER:
                    correct += 1
            if correct >= 5:
                return fmt
        return None

    def _apply_correction(self, text: str, fmt: tuple) -> str:
        _, _, digit_pos, letter_pos = fmt
        chars = list(text)
        for i in digit_pos:
            if chars[i].isalpha() and chars[i] in self.LETTER_TO_DIGIT:
                chars[i] = self.LETTER_TO_DIGIT[chars[i]]
        for i in letter_pos:
            if chars[i].isdigit() and chars[i] in self.DIGIT_TO_LETTER:
                chars[i] = self.DIGIT_TO_LETTER[chars[i]]
        return ''.join(chars)

    def _strictly_valid(self, text: str, fmt: tuple) -> bool:
        _, _, digit_pos, letter_pos = fmt
        for i in digit_pos:
            if not text[i].isdigit():
                return False
        for i in letter_pos:
            if not text[i].isalpha():
                return False
        return True

    def read(self, plate_crop) -> str | None:
        if plate_crop is None or plate_crop.size == 0:
            return None

        preprocessed = self.preprocess(plate_crop)
        result = self.recognizer.run(preprocessed)

        if not result:
            return None

        # result[0] is a PlatePrediction object, not a plain string.
        # The plate text lives in the .plate attribute.
        # str() fallback handles any version differences gracefully.
        prediction = result[0]
        try:
            raw_text = prediction.plate
        except AttributeError:
            raw_text = str(prediction)

        if not raw_text or not raw_text.strip():
            return None

        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        if len(cleaned) < 4:
            return None

        fmt = self._detect_format(cleaned)
        if fmt is None:
            return None

        corrected = self._apply_correction(cleaned, fmt)
        if not self._strictly_valid(corrected, fmt):
            return None

        return corrected