from typing import Tuple, Dict, Any, List, Optional, Union
import os
import numpy as np
import cv2
import mediapipe as mp
import time

from .geometry import detect_geometric_shape
from .objects import InteractiveObjectManager, VirtualObject
from .gemini_recognizer import GeminiSketchRecognizer
from .nano_banana import NanoBananaEngine


class AirCanvas:
    PALETTE = [
        {"name": "CYAN", "color": (255, 230, 0)},
        {"name": "PURPLE", "color": (255, 0, 180)},
        {"name": "GREEN", "color": (0, 255, 120)},
        {"name": "AMBER", "color": (0, 165, 255)},
        {"name": "MAGIC", "color": (240, 90, 160)},
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
        self.brush_size: int = 5
        self.eraser_size: int = 35
        self.current_stroke: List[Tuple[int, int]] = []
        self.last_detected_shape: Optional[str] = None
        self.fps: float = 30.0
        self._last_time: float = time.time()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_path = assets_dir or os.path.join(base_dir, "assets")
        self.object_mgr = InteractiveObjectManager(assets_dir=assets_path)
        self.recognizer = GeminiSketchRecognizer()
        self.nano_banana = NanoBananaEngine()

    def set_api_key(self, key: str):
        self.recognizer.set_api_key(key)
        self.nano_banana.set_api_key(key)

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
        """Clear drawn brush strokes."""
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self.prev_pt = None
        self.current_stroke.clear()
        self.last_detected_shape = None

    def clear_all(self):
        """Clear both strokes and active virtual objects."""
        self.reset()
        self.object_mgr.clear()

    def spawn_object(self, name: str, x: Optional[int] = None, y: Optional[int] = None) -> Optional[VirtualObject]:
        """Spawn any transparent virtual object generated via Nano Banana or preset."""
        w, h = 640, 480
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]

        pos_x = x if x is not None else (w // 2 - 70)
        pos_y = y if y is not None else (h // 2 - 70)

        # Generate cutout via Nano Banana
        bgra_img, _ = self.nano_banana.generate_cutout(name)
        if bgra_img is not None:
            obj = VirtualObject(self.object_mgr._next_id, name, bgra_img, pos_x, pos_y, 140, 140)
            self.object_mgr._next_id += 1
            self.object_mgr.objects.append(obj)
            return obj

        return self.object_mgr.add_object(name, pos_x, pos_y, 140, 140)

    def materialize_current(self, hint: Optional[str] = None) -> Optional[VirtualObject]:
        """Convert the current sketch into a real background-free object using Nano Banana."""
        w, h = 640, 480
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]

        pts = self.current_stroke
        if len(pts) < 4 and self.canvas is not None:
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            nonzero = cv2.findNonZero(gray)
            if nonzero is not None:
                pts = [tuple(p[0]) for p in nonzero]

        min_x, min_y = (w // 2 - 70, h // 2 - 70)
        size = 140
        if len(pts) >= 4:
            arr = np.array(pts)
            min_x, min_y = int(np.min(arr[:, 0])), int(np.min(arr[:, 1]))
            max_x, max_y = int(np.max(arr[:, 0])), int(np.max(arr[:, 1]))
            size = max(100, max(max_x - min_x, max_y - min_y))

        # Determine object label
        object_prompt = hint
        if not object_prompt and self.canvas is not None and self.recognizer.is_configured():
            object_prompt = self.recognizer.identify_sketch(self.canvas)

        if not object_prompt:
            object_prompt = "apple"

        # Generate realistic transparent cutout using Nano Banana
        bgra_img, status_info = self.nano_banana.generate_cutout(object_prompt)
        if bgra_img is not None:
            obj = VirtualObject(self.object_mgr._next_id, object_prompt, bgra_img, min_x, min_y, size, size)
            self.object_mgr._next_id += 1
            self.object_mgr.objects.append(obj)
            self.reset()
            return obj

        # Fallback to manager add_object
        obj = self.object_mgr.add_object(object_prompt, min_x, min_y, size, size)
        self.reset()
        return obj

    def _draw_palette(self, frame: np.ndarray):
        w = frame.shape[1]
        btn_w = min(92, (w - 40) // len(self.PALETTE) - 8)
        btn_h = 44
        spacing = 6
        total_w = len(self.PALETTE) * (btn_w + spacing) - spacing
        start_x = (w - total_w) // 2

        # Subtle dark frosted glass header
        header = frame.copy()
        cv2.rectangle(header, (0, 0), (w, 58), (12, 15, 22), -1)
        cv2.addWeighted(header, 0.88, frame, 0.12, 0, frame)

        for i, item in enumerate(self.PALETTE):
            bx = start_x + i * (btn_w + spacing)
            by = 7
            is_active = (i == self.active_color_idx)

            col = item["color"]
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), col, -1)
            border_col = (255, 255, 255) if is_active else (40, 50, 68)
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border_col, 2 if is_active else 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            label = item["name"]
            (tw, th), _ = cv2.getTextSize(label, font, 0.35, 1)
            tx = bx + (btn_w - tw) // 2
            ty = by + btn_h - 12
            tcol = (20, 20, 20) if label not in ["ERASER", "MAGIC"] else (240, 240, 240)
            cv2.putText(frame, label, (tx, ty), font, 0.35, tcol, 1, cv2.LINE_AA)

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

        btn_w = min(92, (w - 40) // len(self.PALETTE) - 8)
        btn_h = 44
        spacing = 6
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

                interaction = self.object_mgr.update_hand_interaction(index_pt, thumb_pt)
                if interaction["grabbed"]:
                    grabbed_object_name = interaction["grabbed"]
                    mode = f"HOLDING {grabbed_object_name.upper()}"
                    cx, cy = interaction["pinch_center"]
                    cv2.circle(display_frame, (cx, cy), 12, (0, 230, 255), 2, cv2.LINE_AA)

                cv2.circle(display_frame, index_pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(display_frame, index_pt, 4, active_col if not is_magic else (240, 90, 160), -1)

                # Touch top bar tool selection
                if index_pt[1] <= 60 and not interaction["grabbed"]:
                    mode = "TOOL SELECT"
                    self.prev_pt = None
                    self.current_stroke.clear()
                    for i in range(len(self.PALETTE)):
                        bx = start_x + i * (btn_w + spacing)
                        if bx <= index_pt[0] <= bx + btn_w:
                            self.active_color_idx = i
                            break

                # Hover / Stroke finish
                elif index_up and middle_up and not interaction["grabbed"]:
                    mode = "HOVER"
                    self.prev_pt = None
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

                # Drawing stroke
                elif index_up and not middle_up and not interaction["grabbed"]:
                    mode = "MAGIC DRAW" if is_magic else ("DRAWING" if not is_eraser else "ERASING")
                    self.current_stroke.append(index_pt)

                    if self.prev_pt is None:
                        self.prev_pt = index_pt

                    stroke_col = (240, 90, 160) if is_magic else active_col
                    if is_eraser:
                        cv2.circle(self.canvas, index_pt, self.eraser_size, (0, 0, 0), -1)
                        cv2.circle(display_frame, index_pt, self.eraser_size, (160, 160, 160), 1)
                    else:
                        cv2.line(self.canvas, self.prev_pt, index_pt, stroke_col, self.brush_size, cv2.LINE_AA)
                        cv2.circle(display_frame, index_pt, self.brush_size + 1, stroke_col, -1)

                    self.prev_pt = index_pt
                else:
                    self.prev_pt = None

        # Render Virtual Objects
        display_frame = self.object_mgr.render_all(display_frame)

        # Blend drawing canvas
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(display_frame, display_frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        self._draw_palette(combined)

        # Streamlined lower telemetry strip
        cv2.putText(combined, f"{active_tool}  ·  {mode}  ·  {len(self.object_mgr.objects)} OBJECTS  ·  {self.fps:.1f} FPS", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 225), 1, cv2.LINE_AA)

        telemetry = {
            "mode": mode,
            "active_tool": active_tool,
            "brush_size": self.brush_size,
            "fps": round(self.fps, 1),
            "objects_count": len(self.object_mgr.objects),
            "grabbed_object": grabbed_object_name,
            "snapped_shape": self.last_detected_shape,
            "has_api_key": self.nano_banana.is_configured()
        }

        return combined, telemetry
