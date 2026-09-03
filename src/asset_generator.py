"""
Asset generator and background remover for AirCanvas Magic Pencil.
Creates high-resolution transparent PNG cutouts for instant offline materialization,
and provides alpha-masking utilities for custom objects.
"""

import os
import math
from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2


def create_transparent_banana(size: int = 256) -> Image.Image:
    """Generate a high-res, shaded yellow banana with transparent background."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Curved arc points for banana body
    curve_pts = []
    for deg in range(30, 155):
        rad = math.radians(deg)
        # Inner curve
        x = int(size * 0.5 + size * 0.38 * math.cos(rad))
        y = int(size * 0.2 + size * 0.45 * math.sin(rad))
        curve_pts.append((x, y))

    # Outer thickness
    for deg in range(154, 29, -1):
        rad = math.radians(deg)
        x = int(size * 0.5 + size * 0.46 * math.cos(rad))
        y = int(size * 0.2 + size * 0.53 * math.sin(rad))
        curve_pts.append((x, y))

    # Fill base yellow
    draw.polygon(curve_pts, fill=(255, 215, 0, 255))

    # Shading layer
    shade = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shade)
    sdraw.polygon(curve_pts, outline=(230, 180, 0, 255), width=4)

    # Tip and stem
    # Stem at start
    p0 = curve_pts[0]
    draw.rectangle([p0[0] - 6, p0[1] - 12, p0[0] + 6, p0[1]], fill=(60, 100, 30, 255))
    # Tip at end
    p_end = curve_pts[len(curve_pts) // 2]
    draw.ellipse([p_end[0] - 5, p_end[1] - 5, p_end[0] + 5, p_end[1] + 5], fill=(70, 50, 20, 255))

    # Highlight ridge
    ridge_pts = []
    for deg in range(40, 145):
        rad = math.radians(deg)
        x = int(size * 0.5 + size * 0.42 * math.cos(rad))
        y = int(size * 0.2 + size * 0.49 * math.sin(rad))
        ridge_pts.append((x, y))

    for i in range(len(ridge_pts) - 1):
        draw.line([ridge_pts[i], ridge_pts[i + 1]], fill=(255, 245, 150, 180), width=3)

    return img


def create_transparent_sunglasses(size: int = 256) -> Image.Image:
    """Generate sleek transparent aviator sunglasses."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    w, h = size, size
    cy = int(h * 0.5)

    # Left lens
    draw.rounded_rectangle([int(w * 0.15), cy - 35, int(w * 0.45), cy + 45], radius=22, fill=(25, 28, 35, 240), outline=(220, 220, 230, 255), width=5)
    # Right lens
    draw.rounded_rectangle([int(w * 0.55), cy - 35, int(w * 0.85), cy + 45], radius=22, fill=(25, 28, 35, 240), outline=(220, 220, 230, 255), width=5)

    # Bridge
    draw.line([(int(w * 0.45), cy - 10), (int(w * 0.55), cy - 10)], fill=(220, 220, 230, 255), width=6)
    # Top bar
    draw.line([(int(w * 0.2), cy - 35), (int(w * 0.8), cy - 35)], fill=(200, 200, 210, 255), width=4)

    # Glare reflection diagonal
    draw.line([(int(w * 0.22), cy + 30), (int(w * 0.38), cy - 20)], fill=(255, 255, 255, 120), width=3)
    draw.line([(int(w * 0.62), cy + 30), (int(w * 0.78), cy - 20)], fill=(255, 255, 255, 120), width=3)

    return img


def create_transparent_crown(size: int = 256) -> Image.Image:
    """Generate golden royal crown."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size

    base_y = int(h * 0.7)
    pts = [
        (int(w * 0.2), base_y),
        (int(w * 0.18), int(h * 0.38)),
        (int(w * 0.35), int(h * 0.52)),
        (int(w * 0.5), int(h * 0.3)),
        (int(w * 0.65), int(h * 0.52)),
        (int(w * 0.82), int(h * 0.38)),
        (int(w * 0.8), base_y)
    ]
    draw.polygon(pts, fill=(255, 200, 20, 255), outline=(210, 150, 10, 255))
    draw.rectangle([int(w * 0.2), base_y, int(w * 0.8), base_y + 16], fill=(230, 170, 15, 255), outline=(180, 120, 10, 255), width=2)

    # Jewels on peaks
    for tip in [(int(w * 0.18), int(h * 0.38)), (int(w * 0.5), int(h * 0.3)), (int(w * 0.82), int(h * 0.38))]:
        draw.ellipse([tip[0] - 7, tip[1] - 7, tip[0] + 7, tip[1] + 7], fill=(240, 40, 60, 255), outline=(255, 255, 255, 200), width=2)

    return img


def create_transparent_apple(size: int = 256) -> Image.Image:
    """Generate shiny red apple."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size

    # Apple body (two overlapping circles)
    cx, cy = w // 2, int(h * 0.58)
    draw.ellipse([cx - 55, cy - 45, cx + 15, cy + 45], fill=(230, 40, 40, 255))
    draw.ellipse([cx - 15, cy - 45, cx + 55, cy + 45], fill=(240, 45, 45, 255))

    # Stem
    draw.rectangle([cx - 3, cy - 70, cx + 3, cy - 40], fill=(80, 50, 20, 255))
    # Green leaf
    draw.pieslice([cx + 2, cy - 68, cx + 36, cy - 42], 180, 360, fill=(60, 180, 40, 255))

    # Glossy shine
    draw.ellipse([cx - 35, cy - 30, cx - 15, cy - 10], fill=(255, 180, 180, 160))
    return img


def create_transparent_pizza(size: int = 256) -> Image.Image:
    """Generate slice of pizza."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size, size

    # Triangle slice
    pts = [(w // 2, int(h * 0.85)), (int(w * 0.2), int(h * 0.25)), (int(w * 0.8), int(h * 0.25))]
    draw.polygon(pts, fill=(255, 200, 50, 255), outline=(220, 140, 20, 255), width=2)
    # Crust
    draw.line([(int(w * 0.18), int(h * 0.24)), (int(w * 0.82), int(h * 0.24))], fill=(190, 110, 30, 255), width=12)

    # Pepperonis
    peps = [(w // 2, int(h * 0.45)), (int(w * 0.38), int(h * 0.38)), (int(w * 0.62), int(h * 0.4)), (w // 2, int(h * 0.65))]
    for px, py in peps:
        draw.ellipse([px - 10, py - 10, px + 10, py + 10], fill=(180, 40, 30, 255), outline=(130, 25, 20, 255), width=1)
    return img


def generate_all_presets(assets_dir: str):
    """Generate all built-in transparent cutouts."""
    os.makedirs(assets_dir, exist_ok=True)

    generators = {
        "banana": create_transparent_banana,
        "sunglasses": create_transparent_sunglasses,
        "crown": create_transparent_crown,
        "apple": create_transparent_apple,
        "pizza": create_transparent_pizza,
    }

    for name, gen in generators.items():
        path = os.path.join(assets_dir, f"{name}.png")
        img = gen(256)
        img.save(path, format="PNG")

    print(f"[AssetGenerator] Presets generated in {assets_dir}")


def remove_solid_background(pil_img: Image.Image, tolerance: int = 40) -> Image.Image:
    """Convert a solid (white/light) background image into a transparent RGBA PNG."""
    img = pil_img.convert("RGBA")
    arr = np.array(img)

    # Sample corners to determine background color
    corners = [arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]]
    bg_color = np.median(corners, axis=0)

    # Compute Euclidean color distance from background color
    diff = np.linalg.norm(arr[:, :, :3] - bg_color, axis=2)

    # Create smooth alpha channel
    alpha = np.clip((diff - tolerance) / float(tolerance + 10) * 255, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha

    return Image.fromarray(arr)


if __name__ == "__main__":
    generate_all_presets("projects/aircanvas/assets")
