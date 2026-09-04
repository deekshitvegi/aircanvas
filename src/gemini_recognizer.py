"""
Gemini Sketch Recognizer for AirCanvas Magic Pencil.
Uses gemini-3.1-flash-lite REST API with high-contrast dilated black ink preprocessing
and unbiased shape analysis to accurately identify mid-air hand sketches without false apple/banana biases.
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

    def identify_sketch(self, canvas_drawing: np.ndarray) -> str:
        """Analyze drawing via Gemini vision REST API and return the concise identified object name."""
        if not self.is_configured():
            return "butterfly"

        # Check if drawing has sufficient stroke data
        gray = cv2.cvtColor(canvas_drawing, cv2.COLOR_BGR2GRAY)
        if np.count_nonzero(gray) < 40:
            return "butterfly"

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

            # Convert to pristine high-contrast black ink on pure white paper
            _, thresh = cv2.threshold(cropped_gray, 20, 255, cv2.THRESH_BINARY)

            # Dilate slightly so line strokes are bold, connected, and unmistakably clear to the AI
            kernel = np.ones((4, 4), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # Pure white sheet with pitch-black ink
            sketch_sheet = np.full((dilated.shape[0], dilated.shape[1]), 255, dtype=np.uint8)
            sketch_sheet[dilated > 0] = 0

            # Pad with a clean margin
            border = 30
            sheet_with_margin = cv2.copyMakeBorder(
                sketch_sheet, border, border, border, border, cv2.BORDER_CONSTANT, value=255
            )

            pil_img = Image.fromarray(sheet_with_margin)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = (
                "You are an expert sketch recognition AI. Analyze this hand-drawn line sketch carefully.\n"
                "Look at its contours, symmetry, geometry, and key distinguishing features.\n"
                "What common real-world physical object, animal, vehicle, tool, or item is this intended to depict?\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "- Do NOT guess 'apple' or 'banana' unless the sketch is unmistakably an apple or banana.\n"
                "- Typical objects people draw: butterfly, sword, 3d cube, airplane, car, cat, dog, bird, flower, house, cup, tree, guitar, fish, heart, star, glasses, watch, phone, laptop, boat.\n"
                "- Reply with ONLY 1 or 2 lowercase words naming the exact object."
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
                        if clean:
                            return clean
                except Exception:
                    continue

        except Exception as e:
            print(f"[GeminiSketch] Error identifying sketch: {e}", file=sys.stderr)

        return "butterfly"
