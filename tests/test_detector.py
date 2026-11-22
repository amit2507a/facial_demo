import numpy as np
import pytest

from camera import draw_faces
from detector import FaceDetector, mirror_face, parse_faces
from download_model import MODEL_PATH, is_valid


def raw_face(x=10, y=20, w=30, h=40, score=0.9):
    landmarks = [15, 30, 35, 30, 25, 40, 18, 50, 32, 50]
    return [x, y, w, h, *landmarks, score]


def test_no_detections_returns_empty_list():
    assert parse_faces(None, 0.5, 0.5, 640, 480) == []


def test_coordinates_scale_back_to_full_frame():
    raw = np.array([raw_face()], dtype=np.float32)
    face = parse_faces(raw, 0.5, 0.25, 640, 480)[0]
    assert (face.x, face.y, face.width, face.height) == (20, 80, 60, 160)
    assert face.landmarks["right_eye"] == (30, 120)
    assert face.landmarks["left_mouth"] == (64, 200)
    assert face.score == pytest.approx(0.9)


def test_box_is_clipped_to_frame():
    raw = np.array([raw_face(x=-10, y=-5, w=120, h=120)], dtype=np.float32)
    face = parse_faces(raw, 1.0, 1.0, 100, 80)[0]
    assert (face.x, face.y, face.width, face.height) == (0, 0, 100, 80)


def test_box_fully_outside_frame_is_dropped():
    raw = np.array([raw_face(x=200, y=200)], dtype=np.float32)
    assert parse_faces(raw, 1.0, 1.0, 100, 100) == []


def test_faces_sort_by_confidence():
    raw = np.array([raw_face(score=0.7), raw_face(x=50, score=0.95)], dtype=np.float32)
    assert [round(face.score, 2) for face in parse_faces(raw, 1, 1, 640, 480)] == [0.95, 0.7]


def test_malformed_output_is_rejected():
    with pytest.raises(ValueError):
        parse_faces(np.zeros((1, 5), dtype=np.float32), 1, 1, 640, 480)


def test_mirror_moves_box_and_keeps_landmark_labels():
    raw = np.array([raw_face()], dtype=np.float32)
    face = parse_faces(raw, 1, 1, 100, 100)[0]
    mirrored = mirror_face(face, 100)
    assert (mirrored.x, mirrored.y, mirrored.width, mirrored.height) == (60, 20, 30, 40)
    assert mirrored.landmarks["right_eye"] == (85, 30)
    assert mirrored.landmarks["left_eye"] == (65, 30)
    assert mirror_face(mirrored, 100) == face


def test_draw_faces_marks_image_and_handles_top_edge():
    raw = np.array([raw_face(y=0)], dtype=np.float32)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_faces(image, parse_faces(raw, 1, 1, 100, 100))
    assert image.any()


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_model"):
        FaceDetector(model_path=tmp_path / "missing.onnx")


@pytest.mark.skipif(not is_valid(MODEL_PATH), reason="run download_model.py first")
class TestRealModel:
    def test_blank_frame_has_no_faces(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert FaceDetector().detect(frame) == []

    def test_handles_changing_frame_sizes(self):
        detector = FaceDetector()
        for shape in [(480, 640, 3), (720, 1280, 3), (240, 320, 3)]:
            assert detector.detect(np.zeros(shape, dtype=np.uint8)) == []

    def test_rejects_grayscale_frame(self):
        with pytest.raises(ValueError):
            FaceDetector().detect(np.zeros((480, 640), dtype=np.uint8))
