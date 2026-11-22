"""Live webcam viewer with face detection and performance measurements.

Run:  python camera.py
Keys: 1 color | 2 gray | 3 blue | 4 green | 5 red | 6 low-res
      d detection | m mirror | q or Esc quit
"""

import argparse
import sys
import time
from collections.abc import Callable

import cv2
import numpy as np

from detector import Face, FaceDetector, mirror_face

WINDOW_NAME = "Facial Demo - Camera"
MAX_READ_FAILURES = 30  # About one second of failures at 30 FPS.

# OpenCV colors are BGR too, so (0, 255, 0) is green.
BOX_COLOR = (0, 255, 0)
LANDMARK_COLORS = {
    "right_eye": (255, 120, 0),
    "left_eye": (0, 200, 255),
    "nose": (0, 255, 255),
    "right_mouth": (255, 0, 255),
    "left_mouth": (120, 120, 255),
}


class FPSMeter:
    """Measures frame rate, smoothed so the on-screen number stays readable."""

    def __init__(self, smoothing: float = 0.1) -> None:
        if not 0 < smoothing <= 1:
            raise ValueError("smoothing must be in the range (0, 1]")
        self.smoothing = smoothing
        self.last_time: float | None = None
        self.avg_frame_time: float | None = None

    def update(self, now: float | None = None) -> float:
        """Record one frame and return the current FPS."""
        # perf_counter is a high-resolution clock made for measuring durations.
        now = time.perf_counter() if now is None else now
        if self.last_time is not None:
            frame_time = now - self.last_time
            if self.avg_frame_time is None:
                self.avg_frame_time = frame_time
            else:
                # Exponential moving average: recent frames count the most.
                self.avg_frame_time += self.smoothing * (frame_time - self.avg_frame_time)
        self.last_time = now
        if not self.avg_frame_time:
            return 0.0
        return 1.0 / self.avg_frame_time


def as_bgr(single_channel: np.ndarray) -> np.ndarray:
    """Copy a 1-channel image into 3 channels so color overlays can be drawn."""
    return cv2.cvtColor(single_channel, cv2.COLOR_GRAY2BGR)


def view_color(frame: np.ndarray) -> np.ndarray:
    return frame.copy()


def view_gray(frame: np.ndarray) -> np.ndarray:
    # A weighted mix of B, G, R. Human eyes are most sensitive to green.
    return as_bgr(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))


def view_channel(frame: np.ndarray, index: int) -> np.ndarray:
    # OpenCV stores channels as B=0, G=1, R=2. Most other libraries use RGB.
    # Bright pixels here mean "lots of this color".
    return as_bgr(frame[:, :, index])


def view_low_res(frame: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """Shrink then enlarge, showing the detail a model loses at small input sizes."""
    height, width = frame.shape[:2]
    # INTER_AREA averages pixels, which is the right choice when shrinking.
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # INTER_NEAREST keeps the blocks visible so the information loss is obvious.
    # Note that cv2.resize takes (width, height), while .shape is (height, width).
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


VIEWS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "1": ("Color (BGR)", view_color),
    "2": ("Grayscale", view_gray),
    "3": ("Blue channel", lambda frame: view_channel(frame, 0)),
    "4": ("Green channel", lambda frame: view_channel(frame, 1)),
    "5": ("Red channel", lambda frame: view_channel(frame, 2)),
    "6": ("Low-res (1/4 scale)", view_low_res),
}


def draw_overlay(image: np.ndarray, lines: list[str]) -> None:
    """Draw readable text on a dark panel, modifying the image in place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 22
    panel_width = 330
    panel_height = line_height * len(lines) + 10
    # Slicing gives a view of the same memory, so this edits the image itself.
    panel = image[:panel_height, :panel_width]
    panel[:] = (panel * 0.35).astype(np.uint8)
    for i, line in enumerate(lines):
        position = (8, line_height * (i + 1))
        cv2.putText(image, line, position, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_faces(image: np.ndarray, faces: list[Face]) -> None:
    """Draw each face's box, confidence, and landmarks in place."""
    for face in faces:
        # Drawing functions need whole-number pixel coordinates.
        left, top = round(face.x), round(face.y)
        right, bottom = round(face.x + face.width), round(face.y + face.height)
        cv2.rectangle(image, (left, top), (right, bottom), BOX_COLOR, 2)

        # Put the label inside the box when there is no room above it.
        label_y = top - 8 if top >= 24 else top + 20
        cv2.putText(
            image,
            f"{face.score:.2f}",
            (left + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            BOX_COLOR,
            2,
            cv2.LINE_AA,
        )
        for name, (x, y) in face.landmarks.items():
            cv2.circle(image, (round(x), round(y)), 3, LANDMARK_COLORS[name], -1, cv2.LINE_AA)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow opens webcams faster and more reliably on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Close other apps using it and "
            "check Windows Settings > Privacy & security > Camera."
        )
    # These are requests. The driver may pick the closest size it supports.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # A small buffer keeps us on the newest frame, which reduces visible lag.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam viewer and performance probe")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--width", type=int, default=640, help="requested frame width")
    parser.add_argument("--height", type=int, default=480, help="requested frame height")
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run)")
    parser.add_argument("--no-window", action="store_true", help="measure without display")
    parser.add_argument("--no-detect", action="store_true", help="start with detection off")
    parser.add_argument("--input-width", type=int, default=320, help="detector input width")
    parser.add_argument("--threshold", type=float, default=0.8, help="detector confidence")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    try:
        detector = FaceDetector(input_width=args.input_width, score_threshold=args.threshold)
        capture = open_camera(args.camera, args.width, args.height)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    meter = FPSMeter()
    view_key = "1"
    mirror = True
    detecting = not args.no_detect
    frames = 0
    failures = 0
    frames_with_faces = 0
    total_work = 0.0
    total_detect = 0.0
    started = time.perf_counter()
    exit_code = 0

    try:
        while args.frames == 0 or frames < args.frames:
            # read() waits for the camera, so it is not counted as processing work.
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    print("Error: the camera stopped sending frames.", file=sys.stderr)
                    exit_code = 1
                    break
                continue
            failures = 0

            work_start = time.perf_counter()
            height, width, channels = frame.shape

            # Detect before mirroring so left and right landmarks stay correct.
            faces = detector.detect(frame) if detecting else []
            detect_ms = (time.perf_counter() - work_start) * 1000
            total_detect += detect_ms
            frames_with_faces += bool(faces)

            if mirror:
                frame = cv2.flip(frame, 1)  # Flip before drawing so text stays readable.
                faces = [mirror_face(face, width) for face in faces]
            label, view = VIEWS[view_key]
            output = view(frame)
            fps = meter.update()
            work_ms = (time.perf_counter() - work_start) * 1000
            total_work += work_ms
            frames += 1

            if args.no_window:
                continue

            draw_faces(output, faces)
            status = f"{len(faces)} face(s), {detect_ms:.1f} ms" if detecting else "off"
            draw_overlay(
                output,
                [
                    f"View: {label}",
                    f"FPS: {fps:.1f}   Work: {work_ms:.1f} ms",
                    f"Detection: {status}",
                    f"Shape: ({height}, {width}, {channels})  {frame.dtype}",
                    "1-6 views | d detect | m mirror | q quit",
                ],
            )
            cv2.imshow(WINDOW_NAME, output)

            # waitKey also lets the window redraw, so imshow needs it.
            key = cv2.waitKey(1) & 0xFF
            char = chr(key) if key != 255 else ""
            if key == 27 or char == "q":
                break
            if char == "m":
                mirror = not mirror
            elif char == "d":
                detecting = not detecting
            elif char in VIEWS:
                view_key = char
            # Stops cleanly when the user clicks the window's X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Always release the camera, or other apps may be unable to use it.
        capture.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    if frames:
        print(f"Frames: {frames}")
        print(f"Average FPS: {frames / elapsed:.1f}")
        print(f"Average processing: {total_work / frames:.2f} ms per frame")
        print(f"Average detection: {total_detect / frames:.2f} ms per frame")
        print(f"Frames with a face: {frames_with_faces}/{frames}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
