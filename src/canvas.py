from typing import Tuple, Dict, Any, List, Optional, Union
import os
import re
import threading
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

    def __init__(self, max_hands: int = 1, auto_snap: bool = True, assets_dir: Optional[str] = None, show_palette: bool = False):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.auto_snap = auto_snap
        self.show_palette = show_palette
        self.canvas: Optional[np.ndarray] = None
        self.prev_pt: Optional[Tuple[int, int]] = None
        self.active_color_idx: int = 0
        self.brush_size: int = 5
        self.eraser_size: int = 35
        self.current_stroke: List[Tuple[int, int]] = []
        self.last_detected_shape: Optional[str] = None
        self.fps: float = 30.0
        self._last_time: float = time.time()
        self.last_snap_time: float = 0.0
        self.snap_feedback_time: float = 0.0
        self._is_materializing: bool = False
        self.materialize_status: Optional[str] = None
        self._mat_lock = threading.Lock()

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

    def materialize_current(self, hint: Optional[str] = None) -> None:
        """Asynchronously converts current sketch or prompt into a real background-free object."""
        if self._is_materializing:
            return

        w, h = 640, 480
        if self.canvas is not None:
            h, w = self.canvas.shape[:2]

        min_x, min_y = (w // 2 - 70, h // 2 - 70)
        size = 140

        # When materializing by sketch (no text hint), validate that there is an actual sketch
        if not hint and self.canvas is not None:
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            nonzero_count = np.count_nonzero(gray)
            if nonzero_count < 60:
                self.materialize_status = "Canvas empty: sketch an object first!"
                return

            bx, by, bw, bh = cv2.boundingRect(gray)
            min_dim = min(bw, bh)
            max_dim = max(bw, bh)
            # Check for 1D single lines or tiny scribbles (e.g. height < 18 while width > 40)
            if (min_dim < 18 and max_dim > 40) or (min_dim < 12):
                self.materialize_status = "Single stroke detected - finish drawing your object!"
                return

            min_x, min_y = bx, by
            size = max(110, max_dim)
        elif len(self.current_stroke) >= 4:
            arr = np.array(self.current_stroke)
            min_x, min_y = int(np.min(arr[:, 0])), int(np.min(arr[:, 1]))
            max_x, max_y = int(np.max(arr[:, 0])), int(np.max(arr[:, 1]))
            size = max(110, max(max_x - min_x, max_y - min_y))
        elif self.canvas is not None:
            gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            if np.count_nonzero(gray) > 10:
                bx, by, bw, bh = cv2.boundingRect(gray)
                min_x, min_y = bx, by
                size = max(110, max(bw, bh))

        try:
            self._is_materializing = True
            self.materialize_status = "Analyzing sketch with AI..."

            canvas_copy = self.canvas.copy() if self.canvas is not None else np.zeros((h, w, 3), dtype=np.uint8)
            self.current_stroke.clear()

            # Launch in background thread so the camera video NEVER freezes!
            t = threading.Thread(
                target=self._async_materialize_worker,
                args=(canvas_copy, min_x, min_y, size, hint, w, h),
                daemon=True
            )
            t.start()
        except Exception as e:
            print(f"[AirCanvas] Error initiating materialize: {e}", file=sys.stderr)
            self._is_materializing = False
            self.materialize_status = None

    def _async_materialize_worker(self, canvas_copy: np.ndarray, min_x: int, min_y: int, size: int, hint: Optional[str], w: int, h: int):
        try:
            object_prompt = hint
            if not object_prompt and self.recognizer.is_configured():
                self.materialize_status = "Analyzing sketch with Vision AI..."
                object_prompt = self.recognizer.identify_sketch(canvas_copy)

            # Strict rejection of incomplete sketches, math symbols, or abstract line strokes
            INVALID_SUBJECTS = {
                "incomplete", "minus sign", "minus", "dash", "line", "dot",
                "scribble", "hyphen", "straight line", "horizontal line",
                "vertical line", "stroke", "symbol", "math symbol", "punctuation",
                "blank", "nothing", "unknown", "none"
            }

            if not object_prompt or object_prompt.lower().strip() in INVALID_SUBJECTS:
                self.materialize_status = "Incomplete drawing - finish sketching your object!"
                time.sleep(2.5)
                return

            display_name = object_prompt.title()
            self.last_recognized_name = display_name
            self.materialize_status = f"Recognized: '{display_name}'! Finding real PNG..."

            # Retrieve authentic transparent PNG cutout online
            bgra_img, status_info = self.nano_banana.generate_cutout(object_prompt)
            if bgra_img is not None:
                with self._mat_lock:
                    obj = VirtualObject(self.object_mgr._next_id, display_name, bgra_img, min_x, min_y, size, size)
                    self.object_mgr._next_id += 1
                    self.object_mgr.objects.append(obj)
                    if self.canvas is not None:
                        self.canvas = np.zeros_like(self.canvas)

                self.materialize_status = f"Materialized: {display_name}!"
                time.sleep(2.5)
            else:
                self.materialize_status = f"Could not find PNG for {display_name}"
                time.sleep(2.0)

        except Exception as e:
            print(f"[AirCanvas] Error in async materialize: {e}", file=sys.stderr)
        finally:
            self._is_materializing = False
            self.materialize_status = None

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

                # Finger Snap Gesture: Thumb tip touches Middle finger tip
                snap_dist = float(np.hypot(thumb_pt[0] - middle_pt[0], thumb_pt[1] - middle_pt[1]))
                pinch_dist = float(np.hypot(thumb_pt[0] - index_pt[0], thumb_pt[1] - index_pt[1]))

                if snap_dist < 32.0 and pinch_dist > 36.0:
                    if now - self.last_snap_time > 1.8:
                        self.last_snap_time = now
                        self.snap_feedback_time = now
                        # Check if user drew anything on the canvas
                        gray_check = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY) if self.canvas is not None else None
                        if gray_check is not None and np.count_nonzero(gray_check) > 30:
                            self.materialize_current()

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
                    if not is_magic and not is_eraser and len(self.current_stroke) > 10 and self.auto_snap:
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

        if self.show_palette:
            self._draw_palette(combined)

        # On-screen visual feedback for Finger Snap gesture or Materializing in background
        if self.materialize_status:
            clean_status = re.sub(r'[^\x00-\x7F]+', '', self.materialize_status).strip()
            (tw, th), _ = cv2.getTextSize(clean_status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            bx1 = max(10, w // 2 - tw // 2 - 20)
            bx2 = min(w - 10, w // 2 + tw // 2 + 20)
            cv2.rectangle(combined, (bx1, 18), (bx2, 62), (18, 21, 29), -1)
            cv2.rectangle(combined, (bx1, 18), (bx2, 62), (255, 0, 255), 2)
            cv2.putText(combined, clean_status, (bx1 + 18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 255), 2, cv2.LINE_AA)
        elif now - self.snap_feedback_time < 1.5:
            cv2.rectangle(combined, (w // 2 - 210, 18), (w // 2 + 210, 62), (18, 21, 29), -1)
            cv2.rectangle(combined, (w // 2 - 210, 18), (w // 2 + 210, 62), (255, 0, 255), 2)
            cv2.putText(combined, "SNAP DETECTED! MATERIALIZING...", (w // 2 - 195, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2, cv2.LINE_AA)

        # Streamlined lower telemetry strip
        cv2.putText(combined, f"{active_tool}  ·  {mode}  ·  {len(self.object_mgr.objects)} OBJECTS  ·  {self.fps:.1f} FPS", (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 225), 1, cv2.LINE_AA)

        telemetry = {
            "mode": mode,
            "active_tool": active_tool,
            "brush_size": self.brush_size,
            "fps": round(self.fps, 1),
            "objects_count": len(self.object_mgr.objects),
            "grabbed_object": grabbed_object_name,
            "recognized_name": getattr(self, "last_recognized_name", None),
            "materialize_status": self.materialize_status,
            "snapped_shape": self.last_detected_shape,
            "has_api_key": self.nano_banana.is_configured()
        }

        return combined, telemetry
