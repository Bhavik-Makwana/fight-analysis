"""Stage 1: run YOLO pose + ByteTrack over the prepared clip, dump raw per-frame data.

Output: pickle {"fps", "size", "frames"} where frames[i] is a list of people:
  {id, box[x1,y1,x2,y2], kpts[17,3] (x, y, conf)}
Fighter identity is decided later (analyze.py) from the fight config's colour signatures.

    uv run python track.py                        # <workdir>/data/fight.mp4 -> <workdir>/data/tracks.pkl
    uv run python track.py --max-frames 300       # quick speed/quality check
"""
import argparse
import os
import pickle
import sys

import cv2
from ultralytics import YOLO

import fightconfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fightconfig.add_arg(ap)
    ap.add_argument("video", nargs="?", help="default: <workdir>/data/fight.mp4")
    ap.add_argument("out", nargs="?", help="default: <workdir>/data/tracks.pkl")
    ap.add_argument("--model", default="yolo11l-pose.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: auto)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    if not (args.video and args.out):
        cfg = fightconfig.load(args.config)
        args.video = args.video or cfg.video
        args.out = args.out or cfg.path("data", "tracks.pkl")
    device = args.device
    if device is None:
        import torch
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        res = model.track(frame, persist=True, imgsz=args.imgsz, conf=0.25,
                          tracker="bytetrack.yaml", device=device, verbose=False)[0]
        people = []
        if res.boxes is not None and len(res.boxes):
            ids = res.boxes.id
            boxes = res.boxes.xyxy.cpu().numpy()
            kp = res.keypoints.data.cpu().numpy()
            for j in range(len(boxes)):
                people.append({"id": int(ids[j]) if ids is not None else -1, "box": boxes[j], "kpts": kp[j]})
        frames.append(people)
        i += 1
        if i % 300 == 0:
            print(f"{i}/{total}", file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"fps": fps, "size": size, "frames": frames}, f)
    print(f"done: {i} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
