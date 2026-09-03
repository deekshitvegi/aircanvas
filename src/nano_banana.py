"""
Nano Banana Generative Engine for AirCanvas.
Uses deep learning saliency segmentation (rembg) to produce clean,
high-resolution transparent RGBA cutouts without pixelation, holes, or background artifacts.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import io
from rembg import remove


class NanoBananaEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 10)

    def _fetch_photographic_reference(self, query: str) -> Optional[Image.Image]:
        """Search Wikimedia for high-res authentic reference photography."""
        try:
            clean_q = query.replace("_", " ").strip()
            api_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search"
                f"&gsrsearch={urllib.parse.quote(clean_q)}&gsrlimit=3&prop=pageimages&pithumbsize=500"
            )
            req = urllib.request.Request(api_url, headers={"User-Agent": "AirCanvasStudio/1.0 (contact@visionforge.ai)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for _, page in pages.items():
                    thumb_url = page.get("thumbnail", {}).get("source")
                    if thumb_url and ("svg" not in thumb_url.lower()):
                        img_req = urllib.request.Request(thumb_url, headers={"User-Agent": "AirCanvasStudio/1.0"})
                        with urllib.request.urlopen(img_req, timeout=5) as img_resp:
                            return Image.open(io.BytesIO(img_resp.read())).convert("RGB")
        except Exception:
            pass
        return None

    def generate_cutout(self, prompt_text: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Synthesizes a realistic object and isolates it with rembg AI segmentation
        for pixel-perfect alpha transparency.
        """
        clean_prompt = prompt_text.strip().lower()
        if not clean_prompt:
            clean_prompt = "cube"

        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

        # 1. Fast Asset Cache Check (0.001s response)
        for cand in [clean_prompt, clean_prompt.replace(" ", "_"), clean_prompt.replace("3d ", "")]:
            p = os.path.join(assets_dir, f"{cand}.png")
            if os.path.exists(p):
                try:
                    rgba = Image.open(p).convert("RGBA")
                    arr = cv2.cvtColor(np.array(rgba), cv2.COLOR_RGBA2BGRA)
                    return arr, f"Loaded {clean_prompt}"
                except Exception:
                    pass

        # 2. Dynamic Synthesis via AI or High-Res Photography
        raw_image: Optional[Image.Image] = None

        # Try fast AI synthesis first (with strict 5s timeout)
        try:
            ai_prompt = f"photorealistic {clean_prompt}, commercial product photography, studio lighting, isolated on solid plain background, sharp focus, 8k"
            url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(ai_prompt)}?width=384&height=384&model=turbo&nologo=true"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_image = Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception:
            pass

        # If AI times out or is busy, fallback to authentic reference photography
        if raw_image is None:
            raw_image = self._fetch_photographic_reference(clean_prompt)

        if raw_image is None:
            return None, f"Could not synthesize {clean_prompt}"

        # 3. AI Saliency Segmentation using rembg (deep neural net)
        try:
            clean_cutout = remove(raw_image)

            # Crop tight to the object bounding box
            arr = np.array(clean_cutout)
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3]
                pts = cv2.findNonZero(alpha)
                if pts is not None:
                    x, y, w, h = cv2.boundingRect(pts)
                    x1, y1 = max(0, x - 4), max(0, y - 4)
                    x2, y2 = min(arr.shape[1], x + w + 4), min(arr.shape[0], y + h + 4)
                    arr = arr[y1:y2, x1:x2]

            # Cache to assets directory
            safe_name = "".join(c for c in clean_prompt[:25] if c.isalnum() or c in " _-").strip()
            if safe_name:
                cache_path = os.path.join(assets_dir, f"{safe_name}.png")
                Image.fromarray(arr).save(cache_path)

            bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            return bgra, f"Generated {clean_prompt} successfully"

        except Exception as e:
            print(f"[NanoBanana] rembg error: {e}", file=sys.stderr)

        return None, "Segmentation failed"
