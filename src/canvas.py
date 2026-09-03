from typing import Tuple, Dict, Any, List, Optional, Union
import os
import numpy as np
import cv2
import mediapipe as mp
import time

from .geometry import detect_geometric_shape
from .objects import InteractiveObjectManager, VirtualObject
from .gemini_recognizer import GeminiSketchRecognizer


class AirCanvas:
    PALETTE = [
        {"name": "CYAN", "color": (255, 230, 0)},
        {"name": "PURPLE", "color": (255, 0, 180)},
        {"name": "GREEN", "color": (0, 255, 120)},
        {"name": "AMBER", "color": (0, 165, 255)},
        {"name": "MAGIC", "color": (255, 255, 255)},  # Magic Pencil tool!
        {"name": "ERASER", "color": (0, 0, 0)},
    ]

    def __init__(self, max_hands: int = 1, auto_snap: bool = True, assets_dir: Optional[str] = None):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.auto_snap = auto_snap
        self.canvas: Optional[np.ndarray] = None
        self.prev_pt: Optional[Tuple[int, int]] = None
        self.active_color_idx: int = 0
        self.brush_size: int = 6
        self.eraser_size: int = 35
        self.current_stroke: List[Tuple[int, int]] = []
        self.last_detected_shape: Optional[str] = None
        self.fps: float = 30.0
        self._last_time: float = time.time()

        # Virtual Object & Magic Pencil Engine
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_path = assets_dir or os.path.join(base_dir, "assets")
        self.object_mgr = InteractiveObjectManager(assets_dir=assets_path)
        self.recognizer = GeminiSketchRecognizer()
        self.interaction_status = "READY"

    def set_tool(self, tool: Union[int, str]) -> bool:
        if isinstance(tool, int) and 0 <= tool < len(self.PALETTE):
            self.active_color_idx = tool
            return True
        elif isinstance(tool, str):
            tool_upper = tool.strip().upper()
            for i, p in enumerate(self.PALETTE):
                if p["name"] == tool_upper:
                    self.active_color_idx = i
                    return True
        return False

    def reset(self):
        """Clear drawn lines."""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self.prev_pt = None
        self.current_stroke.clear()
        self.last_detected_shape = None

    def clear_all(self):
        """Clear both drawing and active virtual objects."""
        self.reset()
        self.object_mgr.clear()

    def spawn_object(self, name: str, x: Optional[int] = None, y: Optional[int] = None) -> Optional[VirtualObject]:
        """Directly spawn a transparent virtual object to play with."""
        w, h = 640, 480
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]

        pos_x = x if x is not None else (w // 2 - 70)
        pos_y = y if y is not None else (h // 2 - 70)
        return self.object_mgr.add_object(name, pos_x, pos_y, 140, 140)

    def materialize_current(self, hint: Optional[str] = None) -> Optional[VirtualObject]:
        """Convert the current drawn sketch into a real object."""
        if self.canvas is None:
            return None

        # Check if we have active stroke points
        pts = self.current_stroke
        if len(pts) < 4:
            # Try finding stroke from non-zero canvas pixels
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            nonzero = cv2.findNonZero(gray)
            if nonzero is not None:
                pts = [tuple(p[0]) for p in nonzero]

        if len(pts) < 4:
            # Fallback: spawn in center
            return self.spawn_object(hint or "banana")

        # Determine object name (via Gemini recognition or hint)
        obj_name = hint
        if not obj_name and self.recognizer.is_configured():
            obj_name = self.recognizer.identify_sketch(self.canvas)
        if not obj_name:
            obj_name = "banana"

        # Materialize object at sketch position
        obj = self.object_mgr.materialize_from_stroke(pts, hint=obj_name)
        # Clear the rough sketch so the real object takes over!
        self.reset()
        return obj

    def _draw_palette(self, frame: np.ndarray):
        w = frame.shape[1]
        btn_w = min(100, (w - 40) // len(self.PALETTE) - 10)
        btn_h = 52
        spacing = 8
        total_w = len(self.PALETTE) * (btn_w + spacing) - spacing
        start_x = (w - total_w) // 2

        header = frame.copy()
        cv2.rectangle(header, (0, 0), (w, 70), (14, 18, 25), -1)
        cv2.addWeighted(header, 0.85, frame, 0.15, 0, frame)

        for i, item in enumerate(self.PALETTE):
            bx = start_x + i * (btn_w + spacing)
            by = 8
            is_active = (i == self.active_color_idx)

            col = item["color"]
            if item["name"] == "MAGIC":
                col = (255, 105, 180)  # Magical vibrant pink for Magic Pencil button

            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), col, -1)
            border_col = (255, 255, 255) if is_active else (70, 85, 105)
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border_col, 3 if is_active else 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            label = "MAGIC" if item["name"] == "MAGIC" else item["name"]
            (tw, th), _ = cv2.getTextSize(label, font, 0.38, 1)
            tx = bx + (btn_w - tw) // 2
            ty = by + btn_h - 10
            tcol = (20, 20, 20) if item["name"] not in ["ERASER", "MAGIC"] else (240, 240, 240)
            cv2.putText(frame, label, (tx, ty), font, 0.38, tcol, 1, cv2.LINE_AA)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if dt > 0:
            self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)

        h, w = frame.shape[:2]
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

        display_frame = frame.copy()
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        mode = "HOVER"
        active_tool = self.PALETTE[self.active_color_idx]["name"]
        active_col = self.PALETTE[self.active_color_idx]["color"]
        is_eraser = (active_tool == "ERASER")
        is_magic = (active_tool == "MAGIC")

        btn_w = min(100, (w - 40) // len(self.PALETTE) - 10)
        btn_h = 52
        spacing = 8
        total_w = len(self.PALETTE) * (btn_w + spacing) - spacing
        start_x = (w - total_w) // 2

        grabbed_object_name = None

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                lms = hand_lms.landmark
                index_pt = (int(lms[8].x * w), int(lms[8].y * h))
                thumb_pt = (int(lms[4].x * w), int(lms[4].y * h))
                middle_pt = (int(lms[12].x * w), int(lms[12].y * h))

                index_up = lms[8].y < lms[6].y
                middle_up = lms[12].y < lms[10].y

                # Check pinch interaction to grab & play with virtual objects
                interaction = self.object_mgr.update_hand_interaction(index_pt, thumb_pt)
                if interaction["grabbed"]:
                    grabbed_object_name = interaction["grabbed"]
                    mode = f"GRAB [{grabbed_object_name.upper()}]"
                    # Draw pinch reticle
                    cx, cy = interaction["pinch_center"]
                    cv2.circle(display_frame, (cx, cy), 14, (0, 230, 255), 2, cv2.LINE_AA)
                    cv2.line(display_frame, (cx - 18, cy), (cx + 18, cy), (0, 230, 255), 1)
                    cv2.line(display_frame, (cx, cy - 18), (cx, cy + 18), (0, 230, 255), 1)

                # Visual fingertip cursor
                cv2.circle(display_frame, index_pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(display_frame, index_pt, 4, active_col if not is_magic else (255, 105, 180), -1)

                # Touch top palette bar
                if index_pt[1] <= 70 and not interaction["grabbed"]:
                    mode = "TOOL SELECT"
                    self.prev_pt = None
                    self.current_stroke.clear()
                    for i in range(len(self.PALETTE)):
                        bx = start_x + i * (btn_w + spacing)
                        if bx <= index_pt[0] <= bx + btn_w:
                            self.active_color_idx = i
                            break

                # Hover / Selection mode
                elif index_up and middle_up and not interaction["grabbed"]:
                    mode = "HOVER"
                    self.prev_pt = None

                    # If magic tool was active and stroke ended, auto-materialize!
                    if is_magic and len(self.current_stroke) > 12:
                        self.materialize_current()
                    elif len(self.current_stroke) > 10 and self.auto_snap:
                        detected = detect_geometric_shape(self.current_stroke)
                        if detected:
                            self.last_detected_shape = detected["type"]
                            if detected["type"] == "circle":
                                cv2.circle(self.canvas, detected["center"], detected["radius"], active_col, self.brush_size, cv2.LINE_AA)
                            elif detected["type"] == "rectangle":
                                rx, ry, rw, rh = detected["bbox"]
                                cv2.rectangle(self.canvas, (rx, ry), (rx + rw, ry + rh), active_col, self.brush_size, cv2.LINE_AA)
                            elif detected["type"] == "line":
                                cv2.line(self.canvas, detected["start"], detected["end"], active_col, self.brush_size, cv2.LINE_AA)

                    self.current_stroke.clear()

                # Drawing mode
                elif index_up and not middle_up and not interaction["grabbed"]:
                    mode = "MAGIC DRAW" if is_magic else ("DRAWING" if not is_eraser else "ERASING")
                    self.current_stroke.append(index_pt)

                    if self.prev_pt is None:
                        self.prev_pt = index_pt

                    stroke_col = (255, 105, 180) if is_magic else active_col
                    if is_eraser:
                        cv2.circle(self.canvas, index_pt, self.eraser_size, (0, 0, 0), -1)
                        cv2.circle(display_frame, index_pt, self.eraser_size, (180, 180, 180), 1)
                    else:
                        cv2.line(self.canvas, self.prev_pt, index_pt, stroke_col, self.brush_size, cv2.LINE_AA)
                        cv2.circle(display_frame, index_pt, self.brush_size + 2, stroke_col, -1)

                    self.prev_pt = index_pt
                else:
                    self.prev_pt = None

        # Render Virtual Objects (real objects layer)
        display_frame = self.object_mgr.render_all(display_frame)

        # Blend drawing canvas
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(display_frame, display_frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        self._draw_palette(combined)

        # Bottom HUD
        hud_text = f"TOOL: {active_tool} | MODE: {mode} | OBJECTS: {len(self.object_mgr.objects)} | FPS: {self.fps:.1f}"
        if grabbed_object_name:
            hud_text += f" | PLAYING WITH: {grabbed_object_name.upper()}"
        elif self.last_detected_shape:
            hud_text += f" | SNAPPED: {self.last_detected_shape.upper()}"
        cv2.putText(combined, hud_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)

        telemetry = {
            "mode": mode,
            "active_tool": active_tool,
            "brush_size": self.brush_size,
            "fps": round(self.fps, 1),
            "objects_count": len(self.object_mgr.objects),
            "grabbed_object": grabbed_object_name,
            "snapped_shape": self.last_detected_shape
        }

        return combined, telemetry
