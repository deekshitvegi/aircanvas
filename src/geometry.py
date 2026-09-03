from typing import List, Tuple, Optional
import numpy as np
import cv2


def detect_geometric_shape(points: List[Tuple[int, int]], min_points: int = 15) -> Optional[dict]:
    """Detect if hand stroke resembles a geometric line, circle, or rectangle."""
    if len(points) < min_points:
        return None

    pts_arr = np.array(points, dtype=np.int32)
    start_pt = pts_arr[0]
    end_pt = pts_arr[-1]

    stroke_length = 0.0
    for i in range(len(pts_arr) - 1):
        stroke_length += float(np.linalg.norm(pts_arr[i + 1] - pts_arr[i]))

    displacement = float(np.linalg.norm(end_pt - start_pt))

    # Straight line
    if stroke_length > 40 and (displacement / stroke_length) > 0.88:
        return {
            "type": "line",
            "start": (int(start_pt[0]), int(start_pt[1])),
            "end": (int(end_pt[0]), int(end_pt[1]))
        }

    # Closed shape (circle or rectangle)
    is_closed = displacement < (stroke_length * 0.28)
    if is_closed and stroke_length > 60:
        contour = pts_arr.reshape((-1, 1, 2))
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 0:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.65:
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                return {
                    "type": "circle",
                    "center": (int(cx), int(cy)),
                    "radius": int(radius)
                }

            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) in [4, 5]:
                x, y, w, h = cv2.boundingRect(contour)
                return {
                    "type": "rectangle",
                    "bbox": (int(x), int(y), int(w), int(h))
                }

    return None
