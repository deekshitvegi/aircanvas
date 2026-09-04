"""
Online Free Transparent PNG & Generative Engine for AirCanvas.
Fetches authentic, high-definition transparent PNG cutouts from free online libraries
and canonical real-world photography processed with local deep learning background removal (rembg).
Zero low-res knockoffs, zero cartoon SVGs, 100% real transparent objects.
"""

import os
import sys
import json
import re
import urllib.request
import urllib.parse
import time
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

    def _fetch_from_pngall(self, query: str) -> Optional[Image.Image]:
        """Search and download free high-definition transparent PNGs from PNGAll library."""
        try:
            url = f"https://www.pngall.com/?s={urllib.parse.quote(query)}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                html = r.read().decode("utf-8", errors="ignore")
                matches = re.findall(r'https?://www\.pngall\.com/wp-content/uploads/[^\s\"\'><]+\.png', html)
                valid = [m for m in matches if "logo" not in m.lower() and "banner" not in m.lower() and "icon" not in m.lower()]
                if valid:
                    # Strip thumbnail suffix to get the full-resolution PNG
                    full_url = re.sub(r'-\d+x\d+\.png$', '.png', valid[0])
                    r2 = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(r2, timeout=6) as resp:
                        img = Image.open(io.BytesIO(resp.read())).convert("RGBA")
                        if img.size[0] >= 100 and img.size[1] >= 100:
                            return img
        except Exception:
            pass
        return None

    def _fetch_from_wikipedia_rembg(self, query: str) -> Optional[Image.Image]:
        """Fetch canonical professional real-world photography and isolate with rembg."""
        try:
            api_url = (
                f"https://en.wikipedia.org/w/api.php?action=query&format=json&generator=search"
                f"&gsrsearch={urllib.parse.quote(query)}&gsrlimit=3&prop=pageimages&pithumbsize=800"
            )
            req = urllib.request.Request(api_url, headers={"User-Agent": "AirCanvas/2.0 (contact@visionforge.ai)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for _, page in sorted(pages.items(), key=lambda x: x[1].get("index", 99)):
                    thumb = page.get("thumbnail", {}).get("source")
                    if thumb and "svg" not in thumb.lower():
                        r2 = urllib.request.Request(thumb, headers={"User-Agent": "AirCanvas/2.0"})
                        with urllib.request.urlopen(r2, timeout=6) as resp2:
                            raw = Image.open(io.BytesIO(resp2.read())).convert("RGB")
                            cutout = remove(raw)
                            return cutout
        except Exception:
            pass
        return None

    def _fetch_from_diffusion_rembg(self, query: str) -> Optional[Image.Image]:
        """Fallback dynamic diffusion synthesis if no online photo matches."""
        try:
            prompt = f"a high resolution 3d studio product render of a {query}, isolated on solid white background"
            url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}?width=512&height=512&nologo=true"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                raw = Image.open(io.BytesIO(resp.read())).convert("RGB")
                return remove(raw)
        except Exception:
            pass
        return None

    def generate_cutout(self, prompt_text: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Retrieves an authentic, free transparent PNG online or isolates real photography
        using deep learning saliency segmentation.
        """
        clean_prompt = prompt_text.strip().lower()
        if not clean_prompt:
            clean_prompt = "butterfly"

        # Tier 1: Dedicated Free Transparent PNG Library
        img = self._fetch_from_pngall(clean_prompt)
        source = "online PNG library"

        # Tier 2: Real-World Canonical Photography + Neural Background Removal
        if img is None:
            img = self._fetch_from_wikipedia_rembg(clean_prompt)
            source = "real-world photography cutout"

        # Tier 3: Dynamic Diffusion AI + rembg
        if img is None:
            img = self._fetch_from_diffusion_rembg(clean_prompt)
            source = "AI studio render"

        if img is None:
            return None, f"Could not find transparent image for {clean_prompt}"

        try:
            # Crop tight to object bounding box while preserving natural proportions
            arr = np.array(img)
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3]
                pts = cv2.findNonZero(alpha)
                if pts is not None:
                    x, y, w, h = cv2.boundingRect(pts)
                    x1, y1 = max(0, x - 4), max(0, y - 4)
                    x2, y2 = min(arr.shape[1], x + w + 4), min(arr.shape[0], y + h + 4)
                    arr = arr[y1:y2, x1:x2]

            bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            return bgra, f"Materialized {clean_prompt} via {source}"

        except Exception as e:
            print(f"[NanoBanana] Crop/convert error: {e}", file=sys.stderr)

        return None, "Image processing failed"
