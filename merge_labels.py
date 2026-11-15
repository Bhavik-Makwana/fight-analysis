"""Merge vision-model labels into one deduplicated event list per round, attach an
overlay position from the pose tracks, and compare against the pose-only detector."""
import argparse
import glob
import json
import os
import pickle
from collections import Counter

import numpy as np

import analyze as A
import fightconfig

ap = argparse.ArgumentParser()
fightconfig.add_arg(ap)
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--labels-dir", default=None,
                help="where the r<N>_part*.json labeller files are (default: <workdir>/labels/out; "
                     "or an API run folder from label_strips.py)")
ap.add_argument("--promote", action="store_true",
                help="write the merged labels as the main <workdir>/labels/r<N>_events.json even for an API run")
args = ap.parse_args()
cfg = A.configure(fightconfig.load(args.config))
R = args.round
main_dir = cfg.path("labels", "out")
labels_dir = args.labels_dir or main_dir
default_out = cfg.path("labels", f"r{R}_events.json") if (labels_dir == main_dir or args.promote) \
    else os.path.join(labels_dir, f"r{R}_events.json")  # API runs don't overwrite reviewed labels by default
out_path = args.out or default_out

man = {m["window"]: m for m in json.load(open(cfg.path("labels", "strips", f"manifest_r{R}.json")))}
raw = []
for f in sorted(glob.glob(os.path.join(labels_dir, f"r{R}_part*.json"))):
    raw += json.load(open(f))
covered = {x["window"] for x in raw}
missing = sorted(set(man) - covered)

K = pickle.load(open(cfg.path("data", "K.pkl"), "rb"))
OUTCOMES = ("clean", "blocked", "missed", "unclear")
STRIKES = ("punch", "kick", "teep", "knee", "elbow")


def frame_of(x):
    tile = int(str(x.get("tile", 0)).lstrip("0") or 0)
    return man[x["window"]]["frames"][min(tile, 15)]


strikes, events = [], []
for x in raw:
    s = str(x.get("strike", "")).lower()
    if x.get("window") not in man or s in ("none", ""):
        continue
    if s == "event":
        events.append({**x, "t": frame_of(x)})
        continue
    if s not in STRIKES or str(x.get("attacker", "")).upper() not in A.NAMES:
        continue
    strikes.append({"t": frame_of(x), "by": x["attacker"].upper(), "kind": s,
                    "target": str(x.get("target", "body")).lower(),
                    "outcome": str(x.get("outcome", "unclear")).lower(),
                    "conf": float(x.get("confidence", 0.5)), "note": x.get("note", ""),
                    "window": x["window"]})

# dedupe: same attacker + strike type within 0.35 s (overlapping windows report twice)
strikes.sort(key=lambda e: e["t"])
merged = []
for e in strikes:
    dup = next((m for m in reversed(merged[-6:]) if m["by"] == e["by"] and m["kind"] == e["kind"]
                and abs(m["t"] - e["t"]) <= 10 and m["window"] != e["window"]), None)
    if dup:
        # keep the more confident reading; average the time
        if e["conf"] > dup["conf"]:
            dup.update({k: e[k] for k in ("target", "outcome", "conf", "note")})
        dup["t"] = (dup["t"] + e["t"]) // 2
        dup["seen"] = dup.get("seen", 1) + 1
    else:
        merged.append(e)

# overlay position: attacker limb nearest the defender's target at that frame
LIMB = {"punch": [A.L_WR, A.R_WR], "elbow": [A.L_EL, A.R_EL], "knee": [A.L_KN, A.R_KN],
        "kick": [A.L_AN, A.R_AN, A.L_KN, A.R_KN], "teep": [A.L_AN, A.R_AN]}
for e in merged:
    t, a = e["t"], K[e["by"]]
    d = K[A.NAMES[1 - A.NAMES.index(e["by"])]]
    tgt = {"head": np.nanmean(d[t, :5, :2], 0),
           "body": np.nanmean(d[t, [5, 6, 11, 12], :2], 0),
           "leg": np.nanmean(d[t, [13, 14], :2], 0)}.get(e["target"])
    best = None
    if tgt is not None and not np.isnan(tgt).any():
        for j in LIMB[e["kind"]]:
            p = a[t, j, :2]
            if not np.isnan(p).any():
                dd = np.linalg.norm(p - tgt)
                if best is None or dd < best[0]:
                    best = (dd, p)
        pos = best[1] if best else tgt
    else:
        both = np.nanmean(np.concatenate([a[t, :, :2], d[t, :, :2]]), 0)
        pos = both if not np.isnan(both).any() else np.array([320.0, 150.0])
    e["x"], e["y"] = float(pos[0]), float(pos[1])
    e["landed"] = e["outcome"] == "clean"
    e["round"] = R - 1

json.dump({"strikes": merged, "events": events, "missing_windows": missing},
          open(out_path, "w"), indent=1)

# ---- report
print(f"round {R}: {len(raw)} raw rows, {len(strikes)} strike rows -> {len(merged)} after dedupe;"
      f" {len(missing)} windows missing labels {missing[:5]}")
for n in A.NAMES:
    ev = [e for e in merged if e["by"] == n]
    oc = Counter(e["outcome"] for e in ev)
    cl = [e for e in ev if e["outcome"] == "clean"]
    print(f"\n{n}: {len(ev)} attempts  " + "  ".join(f"{o} {oc[o]}" for o in OUTCOMES))
    print("   clean by type:  " + "  ".join(f"{k} {sum(e['kind'] == k for e in cl)}" for k in STRIKES))
    print("   clean by target: " + "  ".join(f"{k} {sum(e['target'] == k for e in cl)}" for k in ("head", "body", "leg")))
    print("   all attempts by type/target: " + ", ".join(
        f"{k}>{t} {c}" for (k, t), c in Counter((e["kind"], e["target"]) for e in ev).most_common()))
print("\nevents:")
for x in events:
    print("  ", man[x["window"]]["src_start"], x.get("attacker", ""), x.get("note", ""))

# compare with the pose-only detector (out/stats.json)
try:
    cv = [e for e in json.load(open(cfg.path("out", "stats.json")))["strikes"] if e["round"] == R - 1]
    lab = [e for e in merged if e["outcome"] in ("clean", "blocked")]
    hit = sum(any(c["by"] == e["by"] and abs(c["t"] - e["t"]) <= 9 for c in cv) for e in lab)
    hit_l = sum(any(c["by"] == e["by"] and c["landed"] and abs(c["t"] - e["t"]) <= 9 for c in cv)
                for e in lab if e["outcome"] == "clean")
    n_clean = sum(e["outcome"] == "clean" for e in lab)
    print(f"\npose detector vs labels: found {hit}/{len(lab)} contacting strikes (any detection), "
          f"marked {hit_l}/{n_clean} clean ones as landed")
    for n in A.NAMES:
        for tg in ("head", "body", "leg"):
            L = [e for e in lab if e["by"] == n and e["target"] == tg]
            h = sum(any(c["by"] == n and abs(c["t"] - e["t"]) <= 9 for c in cv) for e in L)
            if L:
                print(f"   {n} -> {tg}: detector found {h}/{len(L)}")
except FileNotFoundError:
    pass
