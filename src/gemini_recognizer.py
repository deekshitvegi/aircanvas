"""
Gemini Sketch Recognizer for AirCanvas Magic Pencil.
Identifies hand-drawn sketches to automatically materialize them into realistic objects.
"""

import os
import sys
from typing import Optional, List
import numpy as np
import cv2
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class GeminiSketchRecognizer:
    MODEL_CANDIDATES = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.active_model_name: Optional[str] = None
        self._model = None
        self._discovered_models: List[str] = []
        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str) -> bool:
        self.api_key = api_key.strip()
        return self._init_client()

    def _init_client(self) -> bool:
        if not self.api_key:
            self._model = None
            return False
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            self._discovered_models = []
            try:
                for m in genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        self._discovered_models.append(m.name.replace("models/", ""))
            except Exception:
                pass

            chosen = None
            for cand in self.MODEL_CANDIDATES:
                if cand in self._discovered_models:
                    chosen = cand
                    break

            if not chosen:
                chosen = self._discovered_models[0] if self._discovered_models else self.MODEL_CANDIDATES[0]

            self.active_model_name = chosen
            self._model = genai.GenerativeModel(chosen)
            return True
        except Exception as e:
            print(f"[GeminiSketch] Init error: {e}", file=sys.stderr)
            self._model = None
            return False

    def is_configured(self) -> bool:
        return self._model is not None and bool(self.api_key)

    def identify_sketch(self, canvas_drawing: np.ndarray) -> str:
        """Analyze drawing and return the identified object name."""
        if not self.is_configured():
            return "banana"

        # Check if drawing has enough strokes
        gray = cv2.cvtColor(canvas_drawing, cv2.COLOR_BGR2GRAY)
        if np.count_nonzero(gray) < 100:
            return "banana"

        prompt = """Look at this stroke drawing drawn in the air.
In 1 to 2 words, what everyday object does this outline resemble?
Examples: banana, sunglasses, crown, apple, pizza, flower, cup, star.
Reply ONLY with the lowercase name of the object."""

        import google.generativeai as genai
        # Convert black canvas to white background for clear sketch contrast
        inv_canvas = 255 - canvas_drawing
        pil_img = Image.fromarray(cv2.cvtColor(inv_canvas, cv2.COLOR_BGR2RGB))

        trials = [self.active_model_name] if self.active_model_name else []
        for m in self._discovered_models + self.MODEL_CANDIDATES:
            if m not in trials:
                trials.append(m)

        for m_name in trials:
            if not m_name:
                continue
            try:
                m_inst = genai.GenerativeModel(m_name)
                resp = m_inst.generate_content([prompt, pil_img])
                if resp and resp.text:
                    clean = resp.text.strip().lower().replace(".", "").replace('"', '').replace("'", "")
                    return clean
            except Exception as err:
                if "404" in str(err) or "not found" in str(err):
                    continue
                break

        return "banana"
