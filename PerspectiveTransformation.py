import cv2
import numpy as np


class PerspectiveTransformation:
    """Transform image between front view and bird's-eye view."""

    def __init__(self):
        self.src = np.float32([
            (550, 460),   # top-left
            (150, 720),   # bottom-left
            (1200, 720),  # bottom-right
            (770, 460)    # top-right
        ])

        self.dst = np.float32([
            (100, 0),
            (100, 720),
            (1100, 720),
            (1100, 0)
        ])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, flags=cv2.INTER_LINEAR):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M, (w, h), flags=flags)

    def backward(self, img, flags=cv2.INTER_LINEAR):
        h, w = img.shape[:2]
        return cv2.warpPerspective(img, self.M_inv, (w, h), flags=flags)
