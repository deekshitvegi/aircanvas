"""
Nano Banana Image Generation Engine for AirCanvas.
Integrates Google Gemini / Imagen 3 generative models to create photorealistic
objects from sketches or prompts, and extracts them with transparent backgrounds.
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.error
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io

from .asset_generator import remove_solid_background


class NanoBananaEngine:
    """Handles prompt/sketch-to-image synthesis using Google Gemini / Imagen (Nano Banana)."""

    IMAGEN_MODELS = [
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()
        # Save to .env so it persists across server restarts
        self._persist_key(self.api_key)

    def _persist_key(self, key: str):
        try:
            env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(f"GEMINI_API_KEY={key}\n")
        except Exception as e:
            print(f"[NanoBanana] Could not persist key to .env: {e}", file=sys.stderr)

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10)

    def generate_cutout(self, prompt_text: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate a photorealistic object using Nano Banana / Imagen 3,
        remove the background to produce a transparent RGBA cutout,
        and return as a numpy array (H, W, 4).
        """
        clean_prompt = prompt_text.strip()
        if not clean_prompt:
            clean_prompt = "apple"

        # Enhance prompt for clean product isolation on neutral background
        full_prompt = (
            f"Studio product photography of a single {clean_prompt}, "
            "completely isolated on an even solid white background, high quality, "
            "centered, sharp focus, no shadows on background, cutout style."
        )

        pil_img = None
        error_msg = ""

        # 1. If Gemini API key is configured, call Nano Banana / Imagen REST API
        if self.is_configured():
            for model_name in self.IMAGEN_MODELS:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict?key={self.api_key}"
                    payload = {
                        "instances": [{"prompt": full_prompt}],
                        "parameters": {
                            "sampleCount": 1,
                            "aspectRatio": "1:1",
                            "outputMimeType": "image/png"
                        }
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=18) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        predictions = res_data.get("predictions", [])
                        if predictions and "bytesBase64Encoded" in predictions[0]:
                            img_b64 = predictions[0]["bytesBase64Encoded"]
                            raw_bytes = base64.b64decode(img_b64)
                            pil_img = Image.open(io.BytesIO(raw_bytes))
                            break
                except urllib.error.HTTPError as he:
                    body = he.read().decode("utf-8", errors="ignore")
                    error_msg = f"HTTP {he.code}: {body[:150]}"
                    continue
                except Exception as ex:
                    error_msg = str(ex)
                    continue

        # 2. If Imagen request didn't return an image, fall back to high-res procedural cutout or offline preset
        if pil_img is None:
            # Check if an offline preset matches
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
            preset_path = os.path.join(assets_dir, f"{clean_prompt.lower()}.png")
            if os.path.exists(preset_path):
                pil_img = Image.open(preset_path)
            else:
                # Generate custom stylized high-contrast cutout
                pil_img = self._generate_fallback_symbol(clean_prompt)

        # 3. Strip solid background to guarantee 100% transparent RGBA
        transparent_img = remove_solid_background(pil_img, tolerance=35)
        transparent_img = transparent_img.convert("RGBA")

        # Convert to numpy array (H, W, 4) in BGRA format for OpenCV
        rgba_np = np.array(transparent_img)
        bgra_np = cv2.cvtColor(rgba_np, cv2.COLOR_RGBA2BGRA)

        return bgra_np, ("Success" if self.is_configured() else f"Rendered with offline engine ({error_msg or 'No Gemini key set'})")

    def _generate_fallback_symbol(self, label: str) -> Image.Image:
        """Create a clean, stylized graphic if external API is unreachable."""
        size = 256
        img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        # Vibrant colored circular badge with label
        draw.ellipse([20, 20, size - 20, size - 20], fill=(45, 120, 240, 255), outline=(255, 255, 255, 255), width=4)
        draw.ellipse([45, 45, size - 45, size - 45], fill=(30, 90, 210, 255))

        words = label.upper().split()[:2]
        text = "\n".join(words)
        draw.text((size // 2 - 35, size // 2 - 20), text, fill=(255, 255, 255, 255))
        return img
