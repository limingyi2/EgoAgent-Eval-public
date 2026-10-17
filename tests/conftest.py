from pathlib import Path

import pytest

from egoagent_eval.schema import Question
from scripts.make_test_video import create_video


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "synthetic.avi"
    create_video(path)
    return path


@pytest.fixture
def question(video_path):
    return Question(video_path=str(video_path), question="What shapes are visible?")
