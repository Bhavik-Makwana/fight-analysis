"""Contact sheet of the longest person tracks, for labelling who is who once per track.

When colour alone can't separate the fighters from the referee and corners (shirts that
match skin tones, bare-chested fighters, a camera that zooms), label the tracks instead:
each sheet row shows a track at its start, middle and end. Write the answers to
<workdir>/labels/track_labels.json, e.g.

    {"3": "BLUE", "565": "KHAKI", "63": "ignore"}

analyze.py then uses them: a labelled track is that fighter (or never a fighter, for
"ignore"); unlabelled fragments still go through the colour scoring.

    uv run python make_track_sheet.py --top 60
"""
import argparse
import json
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np

import fightconfig

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
fightconfig.add_arg(ap)
ap.add_argument("--top", type=int, default=60, help="how many of the longest tracks")
ap.add_argument("--min-height", type=float, default=90, help="ignore tracks of small (far-away) people")
ap.add_argument("--per-sheet", type=int, default=20)
args = ap.parse_args()
cfg = fightconfig.load(args.config)
d = pickle.load(open(cfg.path("data", "tracks.pkl"), "rb"))
frames = d["frames"]

boxes = defaultdict(list)  # id -> [(t, box)]
for t, fr in enumerate(frames):
    for p in fr:
        if p["id"] >= 0:
            boxes[p["id"]].append((t, p["box"]))
tracks = [(k, v) for k, v in boxes.items() if np.median([b[3] - b[1] for _, b in v]) >= args.min_height]
tracks.sort(key=lambda kv: -len(kv[1]))
tracks = tracks[: args.top]
existing = {}
lab_path = cfg.path("labels", "track_labels.json")
if os.path.exists(lab_path):
    existing = json.load(open(lab_path))

cap = cv2.VideoCapture(cfg.video)
H0 = int(cap.get(4))
TH = 150  # thumbnail height
rows = []
for k, v in tracks:
    picks = [v[0], v[len(v) // 2], v[-1]]
    thumbs = []
    for t, (x1, y1, x2, y2) in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        pad = 0.15 * (y2 - y1)
        c = img[int(max(0, y1 - pad)):int(min(H0, y2 + pad)), int(max(0, x1 - pad)):int(x2 + pad)]
        c = cv2.resize(c, (int(TH * c.shape[1] / max(1, c.shape[0])), TH))
        c = cv2.copyMakeBorder(c, 0, 0, 0, max(0, 110 - c.shape[1]), cv2.BORDER_CONSTANT)[:, :110]
        cv2.putText(c, f"{t / 30:.0f}s", (2, TH - 4), 0, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(c)
    label = np.zeros((TH, 150, 3), np.uint8)
    cv2.putText(label, f"id {k}", (6, 30), 0, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(label, f"{len(v) / 30:.0f}s of video", (6, 60), 0, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(label, f"{v[0][0] / 30:.0f}-{v[-1][0] / 30:.0f}s", (6, 85), 0, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    if str(k) in existing:
        cv2.putText(label, existing[str(k)], (6, 120), 0, 0.6, (0, 255, 120), 2, cv2.LINE_AA)
    rows.append(np.hstack([label, *thumbs]))

os.makedirs(cfg.path("out"), exist_ok=True)
for s in range(0, len(rows), args.per_sheet):
    chunk = rows[s:s + args.per_sheet]
    half = (len(chunk) + 1) // 2
    left, right = chunk[:half], chunk[half:]
    right += [np.zeros_like(chunk[0])] * (len(left) - len(right))
    sheet = np.hstack([np.vstack(left), np.full((TH * len(left), 12, 3), 40, np.uint8), np.vstack(right)])
    out = cfg.path("out", f"tracks_{s // args.per_sheet + 1}.jpg")
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)
covered = sum(len(v) for _, v in tracks)
print(f"{len(tracks)} tracks shown; label them in {lab_path}")
