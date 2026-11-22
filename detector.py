"""Stage 2: a clean face detection interface around YuNet.

Later stages use Face and FaceDetector, so they never depend on YuNet's raw format.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from download_model import MODEL_PATH

Point = tuple[float, float]

# YuNet returns 15 numbers per face, in this order:
#   0-3    box: x, y, width, height
#   4-13   five landmarks as x, y pairs
#   14     confidence score
RAW_FACE_LENGTH = 15
LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "right_mouth", "left_mouth")


@dataclass(frozen=True)
class Face:
    """One detected face, in the coordinates of the original full-size frame."""

    x: float
    y: float
    width: float
    height: float
    score: float
    landmarks: dict[str, Point]


def parse_faces(
    raw: np.ndarray | None,
    scale_x: float,
    scale_y: float,
    frame_width: int,
    frame_height: int,
) -> list[Face]:
    """Convert raw model output into Face objects on the original frame."""
    if raw is None or len(raw) == 0:
        return []
    if raw.ndim != 2 or raw.shape[1] < RAW_FACE_LENGTH:
        raise ValueError(f"unexpected YuNet output shape: {raw.shape}")

    faces = []
    for row in raw:
        # Divide by the resize scale to map model coordinates back to the frame.
        left = row[0] / scale_x
        top = row[1] / scale_y
        right = left + row[2] / scale_x
        bottom = top + row[3] / scale_y

        # Faces at the image edge can produce boxes partly outside the frame.
        # Clipping now prevents invalid crops in later stages.
        left, right = np.clip([left, right], 0, frame_width)
        top, bottom = np.clip([top, bottom], 0, frame_height)
        if right <= left or bottom <= top:
            continue

        landmarks = {
            name: (
                float(np.clip(row[4 + 2 * i] / scale_x, 0, frame_width)),
                float(np.clip(row[5 + 2 * i] / scale_y, 0, frame_height)),
            )
            for i, name in enumerate(LANDMARK_NAMES)
        }
        faces.append(
            Face(
                x=float(left),
                y=float(top),
                width=float(right - left),
                height=float(bottom - top),
                score=float(row[14]),
                landmarks=landmarks,
            )
        )
    return sorted(faces, key=lambda face: face.score, reverse=True)


def mirror_face(face: Face, frame_width: int) -> Face:
    """Map a face onto a horizontally flipped frame, keeping anatomical labels.

    Detect on the original frame, then mirror the result. If the model ran on a
    flipped image, it would call the person's left eye their right eye.
    """
    landmarks = {name: (frame_width - x, y) for name, (x, y) in face.landmarks.items()}
    return replace(face, x=frame_width - face.x - face.width, landmarks=landmarks)


class FaceDetector:
    """Runs YuNet on a resized copy of each frame and returns full-frame results."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        input_width: int = 320,
        score_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 50,
    ) -> None:
        if input_width <= 0:
            raise ValueError("input_width must be positive")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Model not found: {model_path}. Run: python download_model.py")
        self.input_width = input_width
        self._input_size: tuple[int, int] | None = None
        self._model = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (input_width, input_width),
            score_threshold,  # Ignore candidates below this confidence.
            nms_threshold,  # Boxes overlapping more than this count as one face.
            top_k,  # Maximum candidates kept before suppression.
        )

    def detect_raw(self, frame: np.ndarray) -> tuple[np.ndarray | None, float, float]:
        """Return raw YuNet output plus the x and y scales used for resizing."""
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be a BGR uint8 image with shape (H, W, 3)")

        frame_height, frame_width = frame.shape[:2]
        input_height = max(1, round(frame_height * self.input_width / frame_width))
        input_size = (self.input_width, input_height)
        if input_size != self._input_size:
            self._model.setInputSize(input_size)
            self._input_size = input_size

        # Rounding can make the two scales differ slightly, so track both.
        scale_x = self.input_width / frame_width
        scale_y = input_height / frame_height
        shrinking = self.input_width < frame_width
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        resized = cv2.resize(frame, input_size, interpolation=interpolation)
        _, raw = self._model.detect(resized)
        return raw, scale_x, scale_y

    def detect(self, frame: np.ndarray) -> list[Face]:
        """Detect faces, sorted from highest to lowest confidence."""
        raw, scale_x, scale_y = self.detect_raw(frame)
        frame_height, frame_width = frame.shape[:2]
        return parse_faces(raw, scale_x, scale_y, frame_width, frame_height)
