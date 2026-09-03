import numpy as np
import pytest
from src.objects import InteractiveObjectManager
from src.canvas import AirCanvas


def test_interactive_objects():
    mgr = InteractiveObjectManager(assets_dir="assets")
    obj = mgr.add_object("banana", 100, 100, 80, 80)
    assert obj is not None
    assert obj.name == "banana"
    assert obj.contains(120, 120) is True
    assert obj.contains(10, 10) is False

    # Simulate pinch gesture on the object
    interaction = mgr.update_hand_interaction((120, 120), (122, 122))
    assert interaction["is_pinching"] is True
    assert obj.is_grabbed is True

    # Release pinch
    mgr.update_hand_interaction((120, 120), (180, 180))
    assert obj.is_grabbed is False


def test_magic_pencil_canvas():
    canvas = AirCanvas()
    canvas.set_tool("MAGIC")
    assert canvas.PALETTE[canvas.active_color_idx]["name"] == "MAGIC"

    # Spawn banana directly
    obj = canvas.spawn_object("banana", 50, 50)
    assert obj is not None
    assert len(canvas.object_mgr.objects) == 1

    canvas.clear_all()
    assert len(canvas.object_mgr.objects) == 0
