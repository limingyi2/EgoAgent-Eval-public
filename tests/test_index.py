from egoagent_eval.index import build_time_index, scan_interval_change_points
from egoagent_eval.video import Video


def test_local_time_index_scans_coarse_frames_and_keeps_global_coverage(video_path):
    with Video(video_path) as video:
        index = build_time_index(video, coarse_samples=8, candidate_limit=3)
    assert index.scanned_frames == 8
    assert len(index.coarse_points) == 8
    assert len(index.uniform_windows) >= 2
    assert all(0 <= window.start < window.end <= index.duration
               for window in index.uniform_windows + index.candidate_windows)
    assert index.cpu_seconds >= 0
    assert all(point.change_score >= 0 and point.motion_score >= 0
               for point in index.coarse_points)


def test_local_interval_scan_is_separate_and_bounded(video_path):
    with Video(video_path) as video:
        points, scanned, cpu = scan_interval_change_points(video, 1.0, 5.0, samples=8)
    assert len(points) == scanned == 8
    assert all(1 <= point.timestamp < 5 for point in points)
    assert cpu >= 0
