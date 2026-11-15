"""Pick labelled strikes worth a human look and build a review page for them.

Each pick gets a short cropped clip at 0.4x speed. Reviewers mark the outcome, fix the
attacker / strike / target if the label is wrong, and add strikes the labels missed;
"Save" writes <workdir>/labels/review/answers_<name>.json for apply_review.py.

    uv run python make_review.py --name v1                          # top-weighted picks
    uv run python make_review.py --name v2 --random --outcome unclear --exclude <workdir>/review_v1/items.json
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np

import analyze as A
import fightconfig
import review_ui

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
fightconfig.add_arg(ap)
ap.add_argument("--name", default="v1", help="review set name; output in <workdir>/review_<name>/")
ap.add_argument("--per-round", type=int, default=8)
ap.add_argument("--outdir", default=None, help="override the output folder")
ap.add_argument("--outcome", default="", help="only sample strikes with this labeller outcome")
ap.add_argument("--exclude", default="", help="items.json of an earlier review; skip strikes near those")
ap.add_argument("--random", action="store_true", help="random sample (unbiased calibration) not top-weighted")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--title", default="Strike review")
ap.add_argument("--placeholder", action="store_true", help="mannequin clips instead of real footage")
args = ap.parse_args()
cfg = A.configure(fightconfig.load(args.config))
outdir = args.outdir or cfg.path(f"review_{args.name}")
fps = 30.0
rng = np.random.default_rng(args.seed)


def item_frame(it):
    if "t" in it:
        return it["t"]
    return round((fightconfig.parse_time(it["src_time"]) - cfg.time_offset) * fps)


excluded = [item_frame(it) for it in json.load(open(args.exclude))] if args.exclude else []
K = pickle.load(open(cfg.path("data", "K.pkl"), "rb"))
os.makedirs(os.path.join(outdir, "clips"), exist_ok=True)


def weight(e):
    w = e["conf"]
    w *= {"clean": 1.6, "blocked": 1.3, "unclear": 1.0, "missed": 0.5}.get(e["outcome"], 1)
    w *= {"kick": 1.4, "knee": 1.1, "teep": 1.1, "elbow": 1.2}.get(e["kind"], 1.0)
    w *= 1.3 if e["target"] == "body" else 1.0
    return w


def cat(e):
    if e["kind"] == "punch":
        return "punch-" + ("body" if e["target"] == "body" else "head")
    return "kick" if e["kind"] in ("kick", "teep") else e["kind"]


picks = []
for r in range(1, len(cfg.rounds) + 1):
    path = cfg.path("labels", f"r{r}_events.json")
    if not os.path.exists(path):
        continue
    ev = [e for e in json.load(open(path))["strikes"] if e["conf"] >= 0.4
          and (not args.outcome or e["outcome"] == args.outcome)
          and all(abs(e["t"] - x) > 30 for x in excluded)]
    queues = {}
    for n in A.NAMES:  # per fighter: one of each category first, then by weight (or random)
        mine = [e for e in ev if e["by"] == n]
        mine = [mine[i] for i in rng.permutation(len(mine))] if args.random else sorted(mine, key=weight, reverse=True)
        first = []
        for c in ("punch-head", "punch-body", "kick", "knee"):
            first += [e for e in mine if cat(e) == c][:1]
        queues[n] = first + [e for e in mine if e not in first]
    kept, quota = [], {n: args.per_round // 2 for n in A.NAMES}
    while any(quota.values()) and any(queues.values()):
        for n in A.NAMES:  # alternate fighters; skip candidates within 1.5 s of a kept clip
            while quota[n] and queues[n]:
                e = queues[n].pop(0)
                if all(abs(e["t"] - k["t"]) > 45 for k in kept):
                    kept.append(e)
                    quota[n] -= 1
                    break
    picks += sorted(kept, key=lambda e: e["t"])

cap = cv2.VideoCapture(cfg.video)
W0, H0 = int(cap.get(3)), int(cap.get(4))
ph = (K, A.COL) if args.placeholder else None
items = []
for i, e in enumerate(picks):
    t = e["t"]
    box = review_ui.crop_box(K, A.NAMES, max(0, t - 15), t + 15, W0, H0)
    name = f"clip_{i:02d}.mp4"
    review_ui.write_clip(cfg.video, os.path.join(outdir, "clips", name), t - 21, t + 21, box, 2.5,
                         placeholder=ph)
    items.append({"id": i, "t": t, "clip": f"clips/{name}", "poster": f"clips/{name[:-4]}.jpg",
                  "src_time": cfg.src_time(t), "round": e["round"] + 1, "by": e["by"], "kind": e["kind"],
                  "target": e["target"], "label_outcome": e["outcome"], "label_conf": e["conf"],
                  "note": e.get("note", "")})

json.dump(items, open(os.path.join(outdir, "items.json"), "w"), indent=1)
save_path = f"labels/review/answers_{args.name}.json"
with open(os.path.join(outdir, "index.html"), "w") as f:
    f.write(review_ui.strike_page(items, A.NAMES, A.COL, args.title, save_path))
print(f"{len(items)} clips -> {outdir}/index.html (answers save to <workdir>/{save_path})")
