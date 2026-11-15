"""Apply human review to the labelled strikes and calibrate the rest.

    uv run python apply_review.py --review <workdir>/review_v1:<workdir>/labels/review/answers_v1.json ...

Each --review pair is <review folder (with items.json)>:<answers file>. The answers file is
either what the review page saves (a list of {"id", "outcome", "attacker"?, "kind"?,
"target"?, "add": [...], "note"?}) or a hand-written corrections file (a dict keyed by item
id: {"set": {...}, "remove": true, "add": [...], "why": ...}).

For every pair:
  * reviewed strikes get the human verdict (outcome / attacker / type / target), or are removed
  * strikes the reviewer added are inserted
  * the reviewed set gives P(landed | labeller outcome, fighter), Laplace-smoothed
Every unreviewed strike then gets p_land from that table; reviewed ones get 1 or 0.
Writes <workdir>/labels/r<N>_events_corrected.json.
"""
import argparse
import json
from collections import defaultdict

import analyze as A
import fightconfig

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
fightconfig.add_arg(ap)
ap.add_argument("--review", nargs="+", required=True, help="pairs <review dir>:<answers or corrections json>")
args = ap.parse_args()
cfg = A.configure(fightconfig.load(args.config))
FPS = 30.0


def item_frame(it):
    if "t" in it:
        return it["t"]
    return round((fightconfig.parse_time(it["src_time"]) - cfg.time_offset) * FPS)


def to_corrections(answers):
    """Review-page answers -> the corrections format. Outcome-less answers with no fixes are skipped."""
    out = {}
    for a in answers:
        if "id" not in a:
            continue  # the page's general-comment entry
        c = {"set": {}, "add": a.get("add") or [], "why": a.get("note", "")}
        if a.get("outcome") == "not a strike":
            c["remove"] = True
        elif a.get("outcome") in ("clean", "blocked", "missed"):
            c["set"]["outcome"] = a["outcome"]
        for k_in, k_out in (("attacker", "by"), ("kind", "kind"), ("target", "target")):
            if a.get(k_in):
                c["set"][k_out] = a[k_in]
        if c.get("remove") or c["set"] or c["add"]:
            c["judged"] = bool(c.get("remove") or "outcome" in c["set"])
            out[str(a["id"])] = c
    return out


n_rounds = len(cfg.rounds)
rounds = {r: json.load(open(cfg.path("labels", f"r{r}_events.json"))) for r in range(1, n_rounds + 1)}
for r, d in rounds.items():
    for e in d["strikes"]:
        e["labelled_outcome"] = e["outcome"]

tally = defaultdict(lambda: [0, 0])  # (fighter, labelled outcome) -> [landed, n]
n_rev = 0
for pair in args.review:
    rdir, path = pair.rsplit(":", 1)
    items = json.load(open(f"{rdir}/items.json"))
    raw = json.load(open(path))
    corr = raw if isinstance(raw, dict) else to_corrections(raw)
    for it in items:
        c = corr.get(str(it["id"]))
        if c is None:
            continue
        evs = rounds[it["round"]]["strikes"]
        t = item_frame(it)
        match = min((e for e in evs if e["by"] == it["by"] and e["kind"] == it["kind"]
                     and e["target"] == it["target"] and not e.get("verified")),
                    key=lambda e: abs(e["t"] - t), default=None)
        if match is None or abs(match["t"] - t) > 6:
            print("no match for review item", it["id"], it.get("src_time"))
            continue
        judged = c.get("judged", True)  # hand-written corrections always carry a verdict
        if judged:
            n_rev += 1
            landed = (not c.get("remove")) and c.get("set", {}).get("outcome") == "clean"
            k = (match["by"], match["labelled_outcome"])
            tally[k][0] += int(landed)
            tally[k][1] += 1
        if c.get("remove"):
            evs.remove(match)
        else:
            match.update(c.get("set", {}))
            if judged:
                match["verified"] = True
            match["review_note"] = c.get("why", "")
        for a in c.get("add", []):
            evs.append({"t": match["t"] + round(a["dt"] * FPS), "by": a["by"], "kind": a["kind"],
                        "target": a["target"], "outcome": a["outcome"], "conf": 1.0,
                        "labelled_outcome": "added", "verified": True, "round": it["round"] - 1,
                        "x": match["x"], "y": match["y"], "review_note": c.get("why", "")})

P = {}
print(f"calibration from {n_rev} reviewed strikes  P(landed | labeller outcome, fighter):")
for n in A.NAMES:
    for o in ("clean", "unclear", "blocked", "missed"):
        l, m = tally[(n, o)]
        # Laplace smoothing; a "missed" strike starts from a 10% prior instead of 50%
        P[(n, o)] = (l + 0.1) / (m + 1) if o == "missed" else (l + 1) / (m + 2)
        print(f"  {n:5} {o:8} {l}/{m}  -> {P[(n, o)]:.2f}")

for r, d in rounds.items():
    for e in d["strikes"]:
        if e.get("verified"):
            e["p_land"] = 1.0 if e["outcome"] == "clean" else 0.0
        elif e["conf"] < 0.4:
            e["p_land"] = 0.25 * P[(e["by"], e["labelled_outcome"])]  # unvalidated low-confidence
        else:
            e["p_land"] = P[(e["by"], e["labelled_outcome"])]
        e["landed"] = e["p_land"] >= 0.5
    d["strikes"].sort(key=lambda e: e["t"])
    json.dump(d, open(cfg.path("labels", f"r{r}_events_corrected.json"), "w"), indent=1)

print("\nestimated landed (sum of p_land)       verified clean")
for r, d in rounds.items():
    row = []
    for n in A.NAMES:
        ev = [e for e in d["strikes"] if e["by"] == n]
        est = sum(e["p_land"] for e in ev)
        body = sum(e["p_land"] for e in ev if e["target"] == "body")
        ver = sum(1 for e in ev if e.get("verified") and e["outcome"] == "clean")
        row.append(f"{n[0]} {est:5.1f} (body {body:4.1f}, verified {ver})")
    print(f"R{r}: " + "   ".join(row))
