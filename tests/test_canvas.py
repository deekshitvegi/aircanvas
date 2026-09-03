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
