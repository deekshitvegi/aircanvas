"""
Interactive Virtual Objects & Magic Pencil Engine for AirCanvas.
Allows users to turn hand drawings into transparent, photorealistic objects,
and pinch-grab, move, scale, and play with them in mid-air.
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from PIL import Image

from .asset_generator import generate_all_presets


class VirtualObject:
    def __init__(self, obj_id: int, name: str, rgba_img: np.ndarray, x: int, y: int, width: int = 140, height: int = 140):
        self.id = obj_id
        self.name = name
        self.x = int(x)
        self.y = int(y)
        self.width = max(50, int(width))
        self.height = max(50, int(height))
        self.is_grabbed = False
        self.grab_offset_x = 0
        self.grab_offset_y = 0

        # Resize RGBA to target width/height
        self.rgba = cv2.resize(rgba_img, (self.width, self.height), interpolation=cv2.INTER_AREA)

    def contains(self, px: int, py: int) -> bool:
        """Check if point (px, py) is inside object bounding box."""
        return (self.x <= px <= self.x + self.width) and (self.y <= py <= self.y + self.height)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Alpha-blend the transparent object over the frame."""
        fh, fw = frame.shape[:2]

        # Clamp drawing coordinates to frame
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(fw, self.x + self.width)
        y2 = min(fh, self.y + self.height)

        if x1 >= x2 or y1 >= y2:
            return frame

        # Crop corresponding region from RGBA image
        ox1 = x1 - self.x
        oy1 = y1 - self.y
        ox2 = ox1 + (x2 - x1)
        oy2 = oy1 + (y2 - y1)

        crop_rgba = self.rgba[oy1:oy2, ox1:ox2]
        crop_rgb = crop_rgba[:, :, :3]
        alpha = (crop_rgba[:, :, 3] / 255.0)[:, :, np.newaxis]

        bg_slice = frame[y1:y2, x1:x2]
        blended = (crop_rgb * alpha + bg_slice * (1.0 - alpha)).astype(np.uint8)
        frame[y1:y2, x1:x2] = blended

        # Visual selection border if grabbed
        if self.is_grabbed:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"HOLDING {self.name.upper()}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        return frame


class InteractiveObjectManager:
    def __init__(self, assets_dir: Optional[str] = None):
        if assets_dir and os.path.exists(assets_dir):
            self.assets_dir = os.path.abspath(assets_dir)
        else:
            default_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
            self.assets_dir = default_dir

        os.makedirs(self.assets_dir, exist_ok=True)
        self.objects: List[VirtualObject] = []
        self._next_id = 1
        self._cached_assets: Dict[str, np.ndarray] = {}
        self._load_cached_assets()

    def _load_cached_assets(self):
        """Preload available PNG assets from disk or generate if missing."""
        pngs = [f for f in os.listdir(self.assets_dir) if f.endswith(".png")] if os.path.exists(self.assets_dir) else []
        if not pngs:
            generate_all_presets(self.assets_dir)

        if os.path.exists(self.assets_dir):
            for f in os.listdir(self.assets_dir):
                if f.endswith(".png"):
                    name = os.path.splitext(f)[0].lower()
                    path = os.path.join(self.assets_dir, f)
                    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                    if img is not None and img.shape[2] == 4:
                        self._cached_assets[name] = img

    def add_object(self, name: str, x: int, y: int, width: int = 140, height: int = 140) -> Optional[VirtualObject]:
        """Spawn a new interactive virtual object."""
        name_key = name.strip().lower()
        matched_key = None
        for k in self._cached_assets:
            if k in name_key or name_key in k:
                matched_key = k
                break

        if not matched_key:
            matched_key = "banana" if "banana" in self._cached_assets else (list(self._cached_assets.keys())[0] if self._cached_assets else None)

        if not matched_key:
            return None

        rgba = self._cached_assets[matched_key]
        obj = VirtualObject(self._next_id, matched_key, rgba, x, y, width, height)
        self._next_id += 1
        self.objects.append(obj)
        return obj

    def materialize_from_stroke(self, stroke_points: List[Tuple[int, int]], hint: Optional[str] = None) -> Optional[VirtualObject]:
        """Convert a drawn hand stroke bounding box into a materialized real object."""
        if len(stroke_points) < 4:
            return None

        pts = np.array(stroke_points)
        min_x, min_y = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
        max_x, max_y = int(np.max(pts[:, 0])), int(np.max(pts[:, 1]))

        w = max(90, max_x - min_x)
        h = max(90, max_y - min_y)

        size = max(w, h)
        name = hint or "banana"
        return self.add_object(name, min_x, min_y, size, size)

    def update_hand_interaction(self, index_pt: Tuple[int, int], thumb_pt: Tuple[int, int]) -> Dict[str, Any]:
        """Check for pinch gesture and update grabbed object position."""
        dist = float(np.linalg.norm(np.array(index_pt) - np.array(thumb_pt)))
        is_pinching = bool(dist < 38.0)
        pinch_center = ((index_pt[0] + thumb_pt[0]) // 2, (index_pt[1] + thumb_pt[1]) // 2)

        grabbed_obj = next((o for o in self.objects if o.is_grabbed), None)

        if is_pinching:
            if grabbed_obj is None:
                for obj in reversed(self.objects):
                    if obj.contains(pinch_center[0], pinch_center[1]):
                        obj.is_grabbed = True
                        obj.grab_offset_x = pinch_center[0] - obj.x
                        obj.grab_offset_y = pinch_center[1] - obj.y
                        grabbed_obj = obj
                        break
            else:
                grabbed_obj.x = pinch_center[0] - grabbed_obj.grab_offset_x
                grabbed_obj.y = pinch_center[1] - grabbed_obj.grab_offset_y
        else:
            if grabbed_obj is not None:
                grabbed_obj.is_grabbed = False

        return {
            "is_pinching": is_pinching,
            "grabbed": grabbed_obj.name if grabbed_obj else None,
            "pinch_center": pinch_center
        }

    def render_all(self, frame: np.ndarray) -> np.ndarray:
        """Render all materialized virtual objects."""
        for obj in self.objects:
            frame = obj.render(frame)
        return frame

    def clear(self):
        """Remove all materialized objects."""
        self.objects.clear()
