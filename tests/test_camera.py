import numpy as np
import pytest

from camera import VIEWS, FPSMeter, draw_overlay, view_channel, view_low_res


def test_fps_needs_two_frames():
    meter = FPSMeter()
    assert meter.update(now=1.0) == 0.0
    assert meter.update(now=1.1) == pytest.approx(10.0)


def test_fps_smooths_a_slow_frame():
    meter = FPSMeter(smoothing=0.5)
    meter.update(now=0.0)
    meter.update(now=0.1)
    # Average frame time moves halfway from 0.1 s to 0.3 s, giving 0.2 s.
    assert meter.update(now=0.4) == pytest.approx(5.0)


@pytest.mark.parametrize("smoothing", [0, -0.1, 1.5])
def test_fps_rejects_invalid_smoothing(smoothing):
    with pytest.raises(ValueError):
        FPSMeter(smoothing=smoothing)


def test_channels_follow_bgr_order():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    assert view_channel(frame, 0)[0, 0, 0] == 10
    assert view_channel(frame, 1)[0, 0, 0] == 20
    assert view_channel(frame, 2)[0, 0, 0] == 30


@pytest.mark.parametrize("key", VIEWS)
def test_views_keep_shape_and_do_not_change_input(key):
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    original = frame.copy()
    output = VIEWS[key][1](frame)
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert np.array_equal(frame, original)


def test_low_res_loses_detail():
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    assert not np.array_equal(view_low_res(frame), frame)


def test_overlay_works_on_frame_smaller_than_panel():
    frame = np.full((40, 100, 3), 200, dtype=np.uint8)
    draw_overlay(frame, ["one", "two", "three"])
    assert frame.shape == (40, 100, 3)
    assert frame.min() < 200
