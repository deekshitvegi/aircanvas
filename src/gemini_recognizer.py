"""
Gemini Sketch Recognizer for AirCanvas Magic Pencil.
Uses gemini-3.1-flash-lite REST API to analyze and identify hand sketches in real time.
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
            return "cube"

        # Check if drawing has sufficient stroke data
        gray = cv2.cvtColor(canvas_drawing, cv2.COLOR_BGR2GRAY)
        if np.count_nonzero(gray) < 40:
            return "cube"

        try:
            # Crop to drawing bounding box
            pts = cv2.findNonZero(gray)
            if pts is not None:
                x, y, w, h = cv2.boundingRect(pts)
                pad = 20
                h_img, w_img = canvas_drawing.shape[:2]
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
                cropped = canvas_drawing[y1:y2, x1:x2]
            else:
                cropped = canvas_drawing

            # Invert so black strokes on clean white background are easily read by vision model
            inverted = 255 - cropped
            pil_img = Image.fromarray(cv2.cvtColor(inverted, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": (
                                "Look at this simple outline drawing sketched in mid-air. "
                                "Identify what single physical object was drawn (for example: cube, banana, cricket bat, airpods, sword, apple, car, sunglasses). "
                                "Reply with ONLY 1 to 2 words naming the object in lowercase."
                            )
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
                        clean = text.lower().replace(".", "").replace('"', '').replace("'", "").replace("a ", "").replace("an ", "")
                        if clean:
                            return clean
                except Exception:
                    continue

        except Exception as e:
            print(f"[GeminiSketch] Error identifying sketch: {e}", file=sys.stderr)

        return "cube"
