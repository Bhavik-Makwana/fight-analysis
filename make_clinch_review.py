"""Review page for the labelled clinches: who controlled each one?

"Save" writes <workdir>/labels/review/clinch_answers_<name>.json, which
analyze.py reads with --clinch-review <workdir>/review_clinch_<name>:<answers file>.
"""
import argparse
import json
import os
import pickle

import cv2

import analyze as A
import fightconfig
import review_ui

ap = argparse.ArgumentParser(description=__doc__)
fightconfig.add_arg(ap)
ap.add_argument("--name", default="v1")
ap.add_argument("--n", type=int, default=12, help="how many of the longest clinches")
ap.add_argument("--outdir", default=None)
ap.add_argument("--placeholder", action="store_true", help="mannequin clips instead of real footage")
args = ap.parse_args()
cfg = A.configure(fightconfig.load(args.config))
outdir = args.outdir or cfg.path(f"review_clinch_{args.name}")
fps = 30.0
K = pickle.load(open(cfg.path("data", "K.pkl"), "rb"))
F = len(next(iter(K.values())))
label_files = [cfg.path("labels", f"r{r}_events.json") for r in range(1, len(cfg.rounds) + 1)]
clinch, segs, ctrl = A.clinch_from_labels([p for p in label_files if os.path.exists(p)], F, fps)
segs = sorted(sorted(segs, key=lambda s: s[1] - s[0], reverse=True)[: args.n])
os.makedirs(os.path.join(outdir, "clips"), exist_ok=True)

cap = cv2.VideoCapture(cfg.video)
W0, H0 = int(cap.get(3)), int(cap.get(4))
ph = (K, A.COL) if args.placeholder else None
items = []
for i, (a, b) in enumerate(segs):
    a0, b0 = max(0, a - 15), min(F - 1, b + 15)
    box = review_ui.crop_box(K, A.NAMES, a0, b0, W0, H0, pad=50, lo=120)
    name = f"clinch_{i:02d}.mp4"
    review_ui.write_clip(cfg.video, os.path.join(outdir, "clips", name), a0, b0, box, 1.33, placeholder=ph)
    items.append({"id": i, "clip": f"clips/{name}", "poster": f"clips/{name[:-4]}.jpg",
                  "start": cfg.src_time(a), "end": cfg.src_time(b), "t0": a, "t1": b,
                  "seconds": round((b - a) / fps, 1), "round": A.in_round(a, fps) + 1,
                  **{f"label_{n}_s": round(ctrl[n][a:b].sum() / fps, 1) for n in A.NAMES}})
json.dump(items, open(os.path.join(outdir, "items.json"), "w"), indent=1)
save_path = f"labels/review/clinch_answers_{args.name}.json"
with open(os.path.join(outdir, "index.html"), "w") as f:
    f.write(review_ui.clinch_page(items, A.NAMES, "Clinch review", save_path))
print(f"{len(items)} clinch clips -> {outdir}/index.html (answers save to <workdir>/{save_path})")
