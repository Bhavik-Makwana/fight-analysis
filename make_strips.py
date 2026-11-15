"""Render dense, overlapping labelling windows: 4x4 frame sheets cropped to the fighters.

Each window covers 1.6 s at 10 fps (16 frames), windows start every 1.2 s so a strike
on a boundary appears whole in at least one window. A manifest maps window -> frames.
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np

import analyze as A
import fightconfig

ap = argparse.ArgumentParser(description=__doc__)
fightconfig.add_arg(ap)
ap.add_argument("--round", type=int, required=True, help="1-based")
ap.add_argument("--outdir", default=None, help="default: <workdir>/labels/strips")
ap.add_argument("--win", type=float, default=1.6)
ap.add_argument("--stride", type=float, default=1.2)
ap.add_argument("--fps", type=float, default=10)
ap.add_argument("--placeholder", action="store_true", help="mannequins instead of real footage")
args = ap.parse_args()
cfg = A.configure(fightconfig.load(args.config))
args.outdir = args.outdir or cfg.path("labels", "strips")

K = pickle.load(open(cfg.path("data", "K.pkl"), "rb"))
fps = 30.0
r0, r1 = A.ROUNDS[args.round - 1]
os.makedirs(args.outdir, exist_ok=True)
cap = cv2.VideoCapture(cfg.video)
W0, H0 = int(cap.get(3)), int(cap.get(4))
TILE = 300
src_time = cfg.src_time
if args.placeholder:
    import placeholder as ph


manifest = []
last_crop = None
starts = np.arange(r0, r1 - 0.4, args.stride)
for wi, ws in enumerate(starts):
    frames = [int(round((ws + k / args.fps) * fps)) for k in range(int(args.win * args.fps))]
    # crop = union of both fighters' keypoints across the window, padded, square-ish
    pts = np.concatenate([K[n][frames, :, :2].reshape(-1, 2) for n in A.NAMES])
    pts = pts[~np.isnan(pts).any(1)]
    if len(pts) > 10:
        x0, y0 = np.percentile(pts, 1, axis=0)
        x1, y1 = np.percentile(pts, 99, axis=0)
        pad = 0.25 * max(x1 - x0, y1 - y0) + 20
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 + pad
        half = float(np.clip(half, 90, 180))
        crop = [cx - half, cy - half, cx + half, cy + half]
        last_crop = crop
    else:
        crop = last_crop or [0, 0, W0, H0]
    x0, y0, x1, y1 = crop
    # shift into frame
    w, h = x1 - x0, y1 - y0
    x0 = float(np.clip(x0, 0, max(0, W0 - w)))
    y0 = float(np.clip(y0, 0, max(0, H0 - h)))
    x1, y1 = min(W0, x0 + w), min(H0, y0 + h)
    tiles = []
    for k, t in enumerate(frames):
        if args.placeholder:
            img = ph.frame(K, t, W0, H0, A.COL)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((H0, W0, 3), np.uint8)
        c = img[int(y0):int(y1), int(x0):int(x1)]
        c = cv2.resize(c, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        lab = f"{k:02d} {src_time(t)}"
        cv2.rectangle(c, (0, 0), (150, 22), (0, 0, 0), -1)
        cv2.putText(c, lab, (4, 16), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(c)
    sheet = cv2.vconcat([cv2.hconcat(tiles[i:i + 4]) for i in range(0, 16, 4)])
    name = f"r{args.round}_w{wi:03d}.jpg"
    cv2.imwrite(os.path.join(args.outdir, name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    manifest.append({"window": f"r{args.round}_w{wi:03d}", "file": name, "frames": frames,
                     "src_start": src_time(frames[0]), "src_end": src_time(frames[-1])})
json.dump(manifest, open(os.path.join(args.outdir, f"manifest_r{args.round}.json"), "w"), indent=1)
print(f"{len(manifest)} windows -> {args.outdir}")
