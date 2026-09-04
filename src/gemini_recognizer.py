"""
Gemini Sketch Recognizer for AirCanvas Magic Pencil.
Uses gemini-3.1-flash-lite REST API with high-contrast dilated black ink preprocessing
and 100% open-ended, unbiased shape perception. Zero hardcoded example lists or biased fallbacks.
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.parse
from typing import Optional
import numpy as np
import cv2
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()


class GeminiSketchRecognizer:
    VISION_MODELS = [
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10)

    def identify_sketch(self, canvas_drawing: np.ndarray) -> Optional[str]:
        """
        Analyze drawing via Gemini vision REST API with open-ended visual perception.
        Returns whatever subject the user actually sketched without hardcoded examples.
        """
        if not self.is_configured():
            return None

        # Check if drawing has sufficient stroke data
        gray = cv2.cvtColor(canvas_drawing, cv2.COLOR_BGR2GRAY)
        if np.count_nonzero(gray) < 30:
            return None

        try:
            # Crop tightly around drawing
            pts = cv2.findNonZero(gray)
            if pts is not None:
                x, y, w, h = cv2.boundingRect(pts)
                pad = 25
                h_img, w_img = canvas_drawing.shape[:2]
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
                cropped_gray = gray[y1:y2, x1:x2]
            else:
                cropped_gray = gray

            # Convert to high-contrast black ink on pure white paper
            _, thresh = cv2.threshold(cropped_gray, 20, 255, cv2.THRESH_BINARY)

            # Dilate slightly so line strokes are bold, connected, and unmistakably clear to the vision model
            kernel = np.ones((4, 4), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # Pure white sheet with pitch-black ink
            sketch_sheet = np.full((dilated.shape[0], dilated.shape[1]), 255, dtype=np.uint8)
            sketch_sheet[dilated > 0] = 0

            # Clean border margin
            border = 30
            sheet_with_margin = cv2.copyMakeBorder(
                sketch_sheet, border, border, border, border, cv2.BORDER_CONSTANT, value=255
            )

            pil_img = Image.fromarray(sheet_with_margin)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Open-ended visual perception prompt with incomplete sketch detection
            prompt = (
                "You are an AI sketch perception engine. Look at this hand-drawn outline sketch.\n"
                "Identify what specific physical object, animal, character, vehicle, tool, gadget, plant, clothing, food, or item is drawn.\n"
                "RULES:\n"
                "- If the drawing is just a single line, stroke, minus sign, dash, dot, or unfinished scribble, reply with 'incomplete'.\n"
                "- Otherwise, be direct and accurate to what the lines and shapes actually depict.\n"
                "- Reply with ONLY 1 to 3 words naming the object in lowercase. Do not include any filler text or punctuation."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img
                            }
                        }
                    ]
                }]
            }

            for m in self.VISION_MODELS:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key}"
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        text = res_data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        clean = text.lower().replace(".", "").replace('"', '').replace("'", "").replace("a ", "").replace("an ", "").strip()
                        if clean and len(clean) < 40:
                            return clean
                except Exception:
                    continue

        except Exception as e:
            print(f"[GeminiSketch] Error identifying sketch: {e}", file=sys.stderr)

        return None
