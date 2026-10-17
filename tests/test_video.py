import math

import cv2
import numpy as np
import pytest

from egoagent_eval.video import Video, resize_image


def test_metadata_midpoints_and_actual_frame_indices(video_path):
    with Video(video_path) as video:
        assert video.duration == 8.0
        assert video.fps == 10.0
        assert video.uniform_timestamps(4) == [1.0, 3.0, 5.0, 7.0]
        frame = video.read_frame(1.29)
        assert frame.record.requested_timestamp == 1.29
        assert frame.record.timestamp == 1.2
        assert frame.record.frame_index == 12
        image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert image[120, 120, 2] > 200  # Requested early frame contains the red square.
        assert video.read_frame(7.99).record.frame_index == 79


@pytest.mark.parametrize("timestamp", [-0.1, 8.0, 100, math.nan, math.inf, True, "1"])
def test_reject_timestamp_before_decoding(video_path, timestamp):
    with Video(video_path) as video:
        with pytest.raises(ValueError):
            video.read_frame(timestamp)


@pytest.mark.parametrize("start,end,count", [(0.0, 9.0, 2), (2.0, 2.0, 2), (3.0, 1.0, 2), (-1.0, 2.0, 2),
                                             (0.0, 8.0, 0), (0.0, 8.0, 9), (0.0, 8.0, True)])
def test_invalid_intervals(video_path, start, end, count):
    with Video(video_path) as video:
        with pytest.raises(ValueError):
            video.sample_interval(start, end, count)


def test_resize_preserves_aspect_and_never_upscales():
    image = np.zeros((600, 1200, 3), np.uint8)
    assert resize_image(image, 300).shape == (150, 300, 3)
    small = np.zeros((20, 40, 3), np.uint8)
    assert resize_image(small, 300) is small
    with pytest.raises(ValueError):
        resize_image(image, 1025)


def test_invalid_video(tmp_path):
    with pytest.raises(ValueError):
        Video(tmp_path / "missing.avi")
    path = tmp_path / "bad.avi"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        Video(path)
