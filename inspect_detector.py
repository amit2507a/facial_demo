"""Capture one webcam frame and print YuNet's raw output. No image is saved.

Run: python inspect_detector.py
"""

import argparse
import sys
import time

import numpy as np

from camera import open_camera
from detector import LANDMARK_NAMES, FaceDetector, parse_faces

WARMUP_FRAMES = 15  # Webcams adjust exposure during their first few frames.


def capture_frame(camera_index: int) -> np.ndarray:
    capture = open_camera(camera_index, 640, 480)
    try:
        frame = None
        for _ in range(WARMUP_FRAMES):
            ok, candidate = capture.read()
            if ok and candidate is not None:
                frame = candidate
    finally:
        capture.release()
    if frame is None:
        raise RuntimeError("the camera did not return a frame")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Print raw YuNet output for one frame")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    try:
        detector = FaceDetector(input_width=args.input_width, score_threshold=args.threshold)
        frame = capture_frame(args.camera)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    raw, scale_x, scale_y = detector.detect_raw(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000
    height, width = frame.shape[:2]

    print(f"Frame shape: {frame.shape}")
    print(f"Model input: {args.input_width} x {round(height * scale_y)}")
    print(f"Scale: x={scale_x:.3f}, y={scale_y:.3f}")
    print(f"Inference and resize: {elapsed_ms:.1f} ms")
    print(f"Raw output shape: {None if raw is None else raw.shape}")

    if raw is None:
        print("No face found. Face the camera and run this again.")
        return 0

    for index, (row, face) in enumerate(
        zip(raw, parse_faces(raw, scale_x, scale_y, width, height), strict=False), start=1
    ):
        print(f"\nFace {index}")
        print(f"  Raw box (model pixels): {[round(float(value), 1) for value in row[:4]]}")
        print(f"  Confidence: {row[14]:.3f}")
        print(
            f"  Box on full frame: x={face.x:.1f}, y={face.y:.1f}, "
            f"w={face.width:.1f}, h={face.height:.1f}"
        )
        for name in LANDMARK_NAMES:
            x, y = face.landmarks[name]
            print(f"  {name}: ({x:.1f}, {y:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
