import numpy as np
import pytest
from src.canvas import AirCanvas
from src.geometry import detect_geometric_shape


def test_canvas_processing():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = AirCanvas()
    out, tel = canvas.process_frame(frame)
    assert out.shape == frame.shape
    assert "active_tool" in tel
    assert tel["active_tool"] == "CYAN"

    canvas.set_tool("PURPLE")
    assert canvas.active_color_idx == 1

    canvas.reset()
    assert canvas.prev_pt is None


def test_shape_detection():
    # Synthetic circle points
    theta = np.linspace(0, 2 * np.pi, 30)
    pts = [(int(100 + 40 * np.cos(t)), int(100 + 40 * np.sin(t))) for t in theta]
    detected = detect_geometric_shape(pts)
    assert detected is not None
    assert detected["type"] == "circle"


def test_incomplete_stroke_rejection():
    import cv2
    from src.nano_banana import NanoBananaEngine

    canvas = AirCanvas()
    canvas.canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Empty canvas rejection
    canvas.materialize_current()
    assert "Canvas empty" in (canvas.materialize_status or "")

    # 2. Single 1D line (minus sign) rejection
    cv2.line(canvas.canvas, (100, 200), (180, 200), (240, 90, 160), 3)
    canvas.materialize_current()
    assert "Single stroke" in (canvas.materialize_status or "")

    # 3. NanoBananaEngine safeguard for abstract/incomplete terms
    engine = NanoBananaEngine()
    bgra, status = engine.generate_cutout("minus sign")
    assert bgra is None
    assert "Incomplete drawing" in status

    bgra2, status2 = engine.generate_cutout("incomplete")
    assert bgra2 is None

