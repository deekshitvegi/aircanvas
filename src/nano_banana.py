"""
Nano Banana Generative Engine for AirCanvas.
Synthesizes state-of-the-art 768x768 Flux AI imagery and performs deep-learning
saliency segmentation (rembg) to produce clean, high-resolution, un-stretched transparent cutouts.
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

    def generate_cutout(self, prompt_text: str) -> Tuple[Optional[np.ndarray], str]:
        """
        Synthesizes a realistic object using Flux and isolates it with rembg AI segmentation
        preserving authentic aspect ratio and full color fidelity.
        """
        clean_prompt = prompt_text.strip().lower()
        if not clean_prompt:
            clean_prompt = "butterfly"

        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

        # 1. Fast Asset Cache Check
        for cand in [clean_prompt, clean_prompt.replace(" ", "_"), clean_prompt.replace("3d ", "")]:
            p = os.path.join(assets_dir, f"{cand}.png")
            if os.path.exists(p):
                try:
                    rgba = Image.open(p).convert("RGBA")
                    arr = np.array(rgba)
                    bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
                    return bgra, f"Loaded {clean_prompt}"
                except Exception:
                    pass

        # 2. Dynamic Synthesis via Flux (State-of-the-art Diffusion)
        raw_image: Optional[Image.Image] = None

        full_prompt = (
            f"A masterpiece high-end 3D product render of a {clean_prompt}, "
            "commercial studio photography, hyperrealistic, 8k resolution, cinematic lighting, "
            "ultra-detailed texture, completely isolated on solid pure white background, centered"
        )
        encoded = urllib.parse.quote(full_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=768&height=768&model=flux&nologo=true"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=14) as resp:
                raw_image = Image.open(io.BytesIO(resp.read())).convert("RGB")
        except Exception as ex:
            print(f"[NanoBanana] Flux synthesis error: {ex}", file=sys.stderr)

        if raw_image is None:
            return None, f"Could not synthesize {clean_prompt}"

        # 3. AI Saliency Segmentation using rembg (deep neural net)
        try:
            clean_cutout = remove(raw_image)

            # Crop tight to object bounding box while preserving authentic proportions
            arr = np.array(clean_cutout)
            if arr.shape[2] == 4:
                alpha = arr[:, :, 3]
                pts = cv2.findNonZero(alpha)
                if pts is not None:
                    x, y, w, h = cv2.boundingRect(pts)
                    x1, y1 = max(0, x - 4), max(0, y - 4)
                    x2, y2 = min(arr.shape[1], x + w + 4), min(arr.shape[0], y + h + 4)
                    arr = arr[y1:y2, x1:x2]

            # Cache to assets directory for instant reuse
            safe_name = "".join(c for c in clean_prompt[:25] if c.isalnum() or c in " _-").strip()
            if safe_name:
                cache_path = os.path.join(assets_dir, f"{safe_name}.png")
                Image.fromarray(arr).save(cache_path)

            bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            return bgra, f"Generated {clean_prompt} successfully"

        except Exception as e:
            print(f"[NanoBanana] rembg error: {e}", file=sys.stderr)

        return None, "Segmentation failed"
