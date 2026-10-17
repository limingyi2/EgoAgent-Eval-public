"""Generate a synthetic codec/flow fixture, never an egocentric evaluation sample."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def create_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Refusing to overwrite an existing video fixture")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (640, 360))
    if not writer.isOpened():
        raise ValueError("MJPG encoder is unavailable")
    try:
        for index in range(80):
            image = np.full((360, 640, 3), 240, dtype=np.uint8)
            if index < 40:
                cv2.rectangle(image, (80 + index, 80), (200 + index, 200), (0, 0, 255), -1)
            else:
                cv2.circle(image, (320 + index, 180), 65, (255, 0, 0), -1)
            writer.write(image)
    finally:
        writer.release()
    return {"video": str(path.resolve()), "is_synthetic": True, "purpose": "codec and control-flow fixture; NOT model accuracy",
            "frame_count": 80, "fps": 10.0, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fixtures/synthetic.avi"))
    args = parser.parse_args()
    metadata = create_video(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
