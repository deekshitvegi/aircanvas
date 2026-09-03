import sys
from typing import Union, Tuple, Optional
import numpy as np
import cv2
import time


class VideoStream:
    def __init__(self, source: Union[int, str] = 0, width: int = 854, height: int = 480, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.is_synthetic = (str(source).lower() == "synthetic")
        self.cap: Optional[cv2.VideoCapture] = None
        self._synthetic_tick: int = 0

        if not self.is_synthetic:
            self._open_capture()

    def _open_capture(self):
        src = int(self.source) if (isinstance(self.source, str) and self.source.isdigit()) else self.source
        if isinstance(src, int) and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_opened(self) -> bool:
        return True if self.is_synthetic else (self.cap is not None and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.is_synthetic:
            return True, self._generate_synthetic_frame()

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def _generate_synthetic_frame(self) -> np.ndarray:
        self._synthetic_tick += 1
        t = self._synthetic_tick * 0.05
        w, h = self.width, self.height

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (18, 22, 28)

        cx = int(w / 2 + (w / 3.5) * np.sin(t))
        cy = int(h / 2 + (h / 4.0) * np.cos(t * 1.3))
        cv2.circle(frame, (cx, cy), 35, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(frame, "AIR CANVAS SYNTHETIC BENCHMARK (LIVE)", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 160, 200), 1, cv2.LINE_AA)
        time.sleep(1.0 / self.fps)
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
