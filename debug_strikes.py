"""Render a contact sheet of sampled strike events for eyeballing precision."""
import argparse
import json
import pickle

import cv2
import numpy as np

import analyze as A
import fightconfig

ap = argparse.ArgumentParser()
fightconfig.add_arg(ap)
ap.add_argument("--json", default=None, help="default: <workdir>/out/stats.json")
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--landed", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="default: <workdir>/out/strikes_sheet.jpg")
args = ap.parse_args()
cfg = A.configure(fightconfig.load(args.config))
args.json = args.json or cfg.path("out", "stats.json")
args.out = args.out or cfg.path("out", "strikes_sheet.jpg")

K = pickle.load(open(cfg.path("data", "K.pkl"), "rb"))
ev = [e for e in json.load(open(args.json))["strikes"] if e["landed"] == bool(args.landed)]
rng = np.random.default_rng(args.seed)
pick = sorted(rng.choice(len(ev), min(args.n, len(ev)), replace=False))
cap = cv2.VideoCapture(cfg.video)
tiles = []
for k in pick:
    e = ev[k]
    t = e["t"]
    row = []
    for dt in (-4, 0, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t + dt)
        ok, img = cap.read()
        for n, c in A.COL.items():
            for a, b in A.SKELETON:
                pa, pb = K[n][t + dt, a, :2], K[n][t + dt, b, :2]
                if not (np.isnan(pa).any() or np.isnan(pb).any()):
                    cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), c, 1)
        if dt == 0:
            cv2.circle(img, (int(e["x"]), int(e["y"])), 7, (0, 255, 255), 2)
        cx, cy = int(e["x"]), int(e["y"])
        x0, y0 = int(np.clip(cx - 90, 0, 640 - 180)), int(np.clip(cy - 80, 0, 360 - 160))
        row.append(cv2.resize(img[y0:y0 + 160, x0:x0 + 180], (225, 200)))
    tile = cv2.hconcat(row)
    lab = f"#{k} {e['src_time']} {e['by'][0]} {e['kind']}>{e.get('target', '-')}" + (" snap" if e.get("snap") else "")
    cv2.putText(tile, lab, (4, 16), 0, 0.5, (0, 0, 0), 3)
    cv2.putText(tile, lab, (4, 16), 0, 0.5, (255, 255, 255), 1)
    tiles.append(tile)
while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(args.out, cv2.vconcat([cv2.hconcat(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
print(args.out)
