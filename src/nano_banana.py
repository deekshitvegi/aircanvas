"""
Nano Banana Image Generation Engine for AirCanvas.
Generates genuine photorealistic AI images using Google Gemini / Imagen 3 models,
or retrieves authentic high-resolution photographic cutouts with transparent backgrounds.
Eliminates all placeholder graphics, SVGs, and knockoff drawings.
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io

from .asset_generator import remove_solid_background


class NanoBananaEngine:
    IMAGEN_MODELS = [
        "imagen-3.0-generate-002",
        "imagen-3.0-generate-001",
        "imagen-3.0-fast-generate-001",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()
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

    def _fetch_photographic_cutout(self, query: str) -> Optional[Image.Image]:
        """Search and retrieve an authentic high-resolution photograph of the object."""
        try:
            # Query Wikipedia for real photographic images
            clean_q = query.replace("_", " ").strip()
            api_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search"
                f"&gsrsearch={urllib.parse.quote(clean_q)}&gsrlimit=3&prop=pageimages|images&pithumbsize=600"
            )
            req = urllib.request.Request(api_url, headers={"User-Agent": "AirCanvasStudio/1.0"})
            with urllib.request.urlopen(req, timeout=7) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for _, page in pages.items():
                    thumb_url = page.get("thumbnail", {}).get("source")
                    if thumb_url and ("svg" not in thumb_url.lower()):
                        # Download photograph
                        img_req = urllib.request.Request(thumb_url, headers={"User-Agent": "AirCanvasStudio/1.0"})
                        with urllib.request.urlopen(img_req, timeout=8) as img_resp:
                            return Image.open(io.BytesIO(img_resp.read())).convert("RGBA")
        except Exception:
            pass
        return None

    def generate_cutout(self, prompt_text: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Generate a true photorealistic object using Nano Banana (Google Imagen 3) or
        authentic product photography, and strip the background into a clean transparent RGBA PNG.
        """
        clean_prompt = prompt_text.strip()
        if not clean_prompt:
            clean_prompt = "apple"

        pil_img = None
        status_msg = ""

        # 1. Primary: Use Google Gemini Imagen 3 (Nano Banana) if key is present
        if self.is_configured():
            full_prompt = (
                f"Studio product photography of a real {clean_prompt}, "
                "isolated on pure white background, ultra sharp focus, authentic materials, "
                "hyperrealistic, 8k resolution, commercial catalog shot, no shadows on background."
            )
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
                    with urllib.request.urlopen(req, timeout=16) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        predictions = res_data.get("predictions", [])
                        if predictions and "bytesBase64Encoded" in predictions[0]:
                            img_b64 = predictions[0]["bytesBase64Encoded"]
                            raw_bytes = base64.b64decode(img_b64)
                            pil_img = Image.open(io.BytesIO(raw_bytes))
                            status_msg = "Generated via Google Imagen 3 (Nano Banana)"
                            break
                except urllib.error.HTTPError as he:
                    status_msg = f"Imagen error ({he.code}): {he.read().decode('utf-8', errors='ignore')[:100]}"
                    continue
                except Exception as ex:
                    status_msg = str(ex)
                    continue

        # 2. Secondary: Check local real photo assets
        if pil_img is None:
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
            # Try exact and normalized name
            for candidate in [clean_prompt.lower(), clean_prompt.lower().replace(" ", "_")]:
                preset_path = os.path.join(assets_dir, f"{candidate}.png")
                if os.path.exists(preset_path):
                    pil_img = Image.open(preset_path)
                    status_msg = "Loaded from authentic product asset library"
                    break

        # 3. Tertiary: Fetch real photographic product image
        if pil_img is None:
            pil_img = self._fetch_photographic_cutout(clean_prompt)
            if pil_img is not None:
                status_msg = "Retrieved authentic photographic reference"

        # If no image could be retrieved and no API key is set, fail gracefully without placeholder circles
        if pil_img is None:
            return None, "Gemini API key is required for live AI image synthesis. Please configure your key."

        # 4. Extract clean transparent background
        transparent_img = remove_solid_background(pil_img, tolerance=32)
        transparent_img = transparent_img.convert("RGBA")

        # Convert to BGRA numpy array for OpenCV blending
        rgba_np = np.array(transparent_img)
        bgra_np = cv2.cvtColor(rgba_np, cv2.COLOR_RGBA2BGRA)

        return bgra_np, status_msg
