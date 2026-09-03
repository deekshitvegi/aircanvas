from typing import Tuple, Dict, Any, List, Optional, Union
import numpy as np
import cv2
import mediapipe as mp
import time

from .geometry import detect_geometric_shape


class AirCanvas:
    PALETTE = [
        {"name": "CYAN", "color": (255, 230, 0)},
        {"name": "PURPLE", "color": (255, 0, 180)},
        {"name": "GREEN", "color": (0, 255, 120)},
        {"name": "AMBER", "color": (0, 165, 255)},
        {"name": "RED", "color": (50, 50, 255)},
        {"name": "ERASER", "color": (0, 0, 0)},
    ]

    def __init__(self, max_hands: int = 1, auto_snap: bool = True):
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
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self.prev_pt = None
        self.current_stroke.clear()
        self.last_detected_shape = None

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

            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), item["color"], -1)
            border_col = (255, 255, 255) if is_active else (70, 85, 105)
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), border_col, 3 if is_active else 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(item["name"], font, 0.4, 1)
            tx = bx + (btn_w - tw) // 2
            ty = by + btn_h - 10
            tcol = (20, 20, 20) if item["name"] not in ["ERASER", "RED"] else (240, 240, 240)
            cv2.putText(frame, item["name"], (tx, ty), font, 0.4, tcol, 1, cv2.LINE_AA)

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
        active_col = self.PALETTE[self.active_color_idx]["color"]
        is_eraser = (self.PALETTE[self.active_color_idx]["name"] == "ERASER")

        btn_w = min(100, (w - 40) // len(self.PALETTE) - 10)
        btn_h = 52
        spacing = 8
        total_w = len(self.PALETTE) * (btn_w + spacing) - spacing
        start_x = (w - total_w) // 2

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                lms = hand_lms.landmark
                index_pt = (int(lms[8].x * w), int(lms[8].y * h))
                middle_pt = (int(lms[12].x * w), int(lms[12].y * h))

                index_up = lms[8].y < lms[6].y
                middle_up = lms[12].y < lms[10].y

                cv2.circle(display_frame, index_pt, 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(display_frame, index_pt, 4, active_col, -1)

                # Touch top palette bar
                if index_pt[1] <= 70:
                    mode = "TOOL SELECT"
                    self.prev_pt = None
                    self.current_stroke.clear()
                    for i in range(len(self.PALETTE)):
                        bx = start_x + i * (btn_w + spacing)
                        if bx <= index_pt[0] <= bx + btn_w:
                            self.active_color_idx = i
                            break

                # Hover / Selection mode
                elif index_up and middle_up:
                    mode = "HOVER"
                    self.prev_pt = None
                    if len(self.current_stroke) > 10 and self.auto_snap:
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
                elif index_up and not middle_up:
                    mode = "DRAWING" if not is_eraser else "ERASING"
                    self.current_stroke.append(index_pt)

                    if self.prev_pt is None:
                        self.prev_pt = index_pt

                    if is_eraser:
                        cv2.circle(self.canvas, index_pt, self.eraser_size, (0, 0, 0), -1)
                        cv2.circle(display_frame, index_pt, self.eraser_size, (180, 180, 180), 1)
                    else:
                        cv2.line(self.canvas, self.prev_pt, index_pt, active_col, self.brush_size, cv2.LINE_AA)
                        cv2.circle(display_frame, index_pt, self.brush_size + 2, active_col, -1)

                    self.prev_pt = index_pt
                else:
                    self.prev_pt = None
                    self.current_stroke.clear()

        # Blend drawing canvas
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(display_frame, display_frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        combined = cv2.add(frame_bg, canvas_fg)

        self._draw_palette(combined)

        # Bottom HUD
        active_tool = self.PALETTE[self.active_color_idx]["name"]
        hud_text = f"TOOL: {active_tool} | MODE: {mode} | FPS: {self.fps:.1f}"
        if self.last_detected_shape:
            hud_text += f" | SNAPPED: {self.last_detected_shape.upper()}"
        cv2.putText(combined, hud_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)

        telemetry = {
            "mode": mode,
            "active_tool": active_tool,
            "brush_size": self.brush_size,
            "fps": round(self.fps, 1),
            "snapped_shape": self.last_detected_shape
        }

        return combined, telemetry
