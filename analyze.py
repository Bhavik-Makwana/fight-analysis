"""Stage 2: turn raw pose tracks into fight events + an annotated video.

Pipeline
  1. camera-motion estimate (phase correlation on the top band of the frame, usually
     ceiling/lights/crowd) so motion numbers aren't polluted by the camera panning
  2. fighter identity per frame (colour signatures from the fight config + track-level
     colour + size + continuity)
  3. per-limb speed peaks -> strike candidates -> landed / thrown, target, head snap
  4. clinch segments + who controls them, forward pressure, falls / dumps
  5. per-round stats (JSON) and an overlay video with a HUD and timeline
"""
import argparse
import json
import os
import pickle
import re
import subprocess
from collections import defaultdict

import cv2
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
SKELETON = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
            (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KN), (L_KN, L_AN),
            (R_HIP, R_KN), (R_KN, R_AN)]

KP_CONF = 0.3

# Fight-specific values, set by configure(cfg) from the fight config (fightconfig.py)
CFG = None
TIME_OFFSET = 0.0   # source time of the prepared clip's t=0 (display only)
ROUNDS = []         # [(start_s, end_s), ...] in the prepared clip
NAMES = ("A", "B")  # the two fighter ids
F0, F1 = NAMES
COL = {}            # fighter id -> BGR overlay colour
CHOSEN_IDS = {}     # fighter id -> per-frame track id picked by assign_fighters (for track_report)
FRAME_W = 0         # clip width in pixels (set in main)


def configure(cfg):
    """Install a fight config into this module's globals (call once, before anything else)."""
    global CFG, TIME_OFFSET, ROUNDS, NAMES, F0, F1, COL, CTRL_RE
    CFG = cfg
    TIME_OFFSET = cfg.time_offset
    ROUNDS = list(cfg.rounds)
    NAMES = cfg.names
    F0, F1 = NAMES
    COL = dict(cfg.colours)
    ids = "|".join(re.escape(n.lower()) for n in NAMES)
    CTRL_RE = re.compile(r"\b(" + ids + r")(?:'s)?\s+(?:"
                         r"controls?|controlling|drives|turns|pushes|pins|pulls|gets both arms|wraps|ties up|"
                         r"(?:hands?|arms?)\s+(?:on|over|around|behind|up around))")
    return cfg


def src_stamp(t, fps=30.0):
    s = TIME_OFFSET + int(t / fps)
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"

LIMBS = [(L_WR, "punch", L_SH), (R_WR, "punch", R_SH),
         (L_AN, "kick", L_HIP), (R_AN, "kick", R_HIP),
         (L_KN, "knee", L_HIP), (R_KN, "knee", R_HIP)]
SPEED_THR = {"punch": 0.20, "kick": 0.24, "knee": 0.18}  # torso-lengths per frame


# ---------------------------------------------------------------- helpers
def nan_interp(a, max_gap):
    """Linear-interpolate NaN runs up to max_gap along axis 0 (1-D array)."""
    a = a.copy()
    n = len(a)
    isn = np.isnan(a)
    if isn.all():
        return a
    i = 0
    while i < n:
        if isn[i]:
            j = i
            while j < n and isn[j]:
                j += 1
            if i > 0 and j < n and j - i <= max_gap:
                a[i:j] = np.linspace(a[i - 1], a[j], j - i + 2)[1:-1]
            i = j
        else:
            i += 1
    return a


def smooth(a, w):
    """Centered moving average that tolerates NaN (1-D)."""
    if w <= 1:
        return a
    v = np.where(np.isnan(a), 0, a)
    m = (~np.isnan(a)).astype(float)
    k = np.ones(w)
    num = np.convolve(v, k, "same")
    den = np.convolve(m, k, "same")
    out = num / np.maximum(den, 1e-9)
    out[den < w / 2] = np.nan
    return out


def mid(K, a, b):
    return (K[:, a, :2] + K[:, b, :2]) / 2


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def runs(mask, min_len=1, merge_gap=0):
    """Return [(start, end_exclusive)] of True runs, merging small gaps."""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for r in out:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged if r[1] - r[0] >= min_len]


# ---------------------------------------------------------------- 1. camera
def camera_motion(video, n):
    cap = cv2.VideoCapture(video)
    prev, win = None, None
    shift = np.zeros((n, 2))
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f[:36], cv2.COLOR_BGR2GRAY).astype(np.float32)  # top band (ceiling/lights)
        if win is None:
            win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
            if resp > 0.15 and abs(dx) < 40 and abs(dy) < 20:
                shift[i] = (dx, dy)
        prev = g
    return np.cumsum(shift, axis=0)  # content offset; subtract to stabilise


# ---------------------------------------------------------------- 2. identity
def hsv_fraction(px, ranges):
    """Fraction of HSV pixels inside any of the config's ranges (inclusive bounds)."""
    if not ranges:
        return 0.0
    h, s, v = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    m = np.zeros(len(px), bool)
    for r in ranges:
        m |= ((h >= r["h"][0]) & (h <= r["h"][1]) & (s >= r["s"][0]) & (s <= r["s"][1])
              & (v >= r["v"][0]) & (v <= r["v"][1]))
    return float(m.mean())


def appearance(video, frames, cache):
    """Score each detection against every fighter's colour signature, from its box
    (robust to side-on poses): the config's `torso` ranges on the upper body, `shorts`
    ranges on the hips/thighs (the useful one when both fighters are bare-chested) and
    `legs` ranges (e.g. shin guards) on the lower legs."""
    try:
        return pickle.load(open(cache, "rb"))
    except (FileNotFoundError, EOFError):
        pass
    cap = cv2.VideoCapture(video)
    feats = []
    for fr in frames:
        ok, img = cap.read()
        if not ok:
            feats.append([None] * len(fr))
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, W = hsv.shape[:2]
        row = []
        for p in fr:
            x1, y1, x2, y2 = p["box"]
            bw, bh = x2 - x1, y2 - y1

            def crop(fx1, fy1, fx2, fy2):
                a, b = int(np.clip(x1 + fx1 * bw, 0, W)), int(np.clip(x1 + fx2 * bw, 0, W))
                c, d = int(np.clip(y1 + fy1 * bh, 0, H)), int(np.clip(y1 + fy2 * bh, 0, H))
                return hsv[c:d, a:b].reshape(-1, 3)

            up, lo = crop(0.3, 0.18, 0.7, 0.45), crop(0.15, 0.72, 0.85, 0.95)
            mid = crop(0.25, 0.44, 0.75, 0.62)
            if len(up) < 10:
                row.append(None)
                continue
            row.append({
                "torso": {f["id"]: hsv_fraction(up, f.get("torso")) for f in CFG.fighters},
                "legs": {f["id"]: hsv_fraction(lo, f.get("legs")) if len(lo) >= 10 else 0.0
                         for f in CFG.fighters},
                "shorts": {f["id"]: hsv_fraction(mid, f.get("shorts")) if len(mid) >= 10 else 0.0
                           for f in CFG.fighters},
                "h": float(bh)})
        feats.append(row)
    pickle.dump(feats, open(cache, "wb"))
    return feats


def track_label_fn(labels, fps=30.0):
    """labels: {"<track id>": "FIGHTER" | "ignore" | [["FIGHTER"|"ignore", from_s, to_s], ...]}
    (from make_track_sheet.py). Returns f(track_id, frame) -> label or None."""
    def f(tid, t):
        v = labels.get(str(tid))
        if v is None or isinstance(v, str):
            return v
        sec = t / fps
        for lab, a, b in v:
            if a <= sec <= b:
                return lab
        return None
    return f


def assign_fighters(frames, feats, track_labels=None):
    """Pick the two fighters per frame by jointly scoring every candidate pair on
    colour signature, apparent size, tracker-ID continuity and distance moved."""
    F = len(frames)
    for fr, fs in zip(frames, feats):
        for p, f in zip(fr, fs):
            p["app"] = f

    weights = {f["id"]: {"torso": 1.0, "legs": 1.5, "shorts": 1.5, "other_torso": 1.0, "other_shorts": 1.0,
                         **f.get("weights", {})}
               for f in CFG.fighters}

    def colour(p):
        """Per-fighter colour score: own torso + legs signature, minus looking like the other."""
        if p["app"] is None:
            return 0.0, 0.0
        t, l = p["app"]["torso"], p["app"]["legs"]
        sh = p["app"].get("shorts", {n: 0.0 for n in NAMES})
        out = []
        for n in NAMES:
            w = weights[n]
            other_t = sum(t[m] for m in NAMES if m != n)
            other_s = sum(sh[m] for m in NAMES if m != n)
            out.append(w["torso"] * t[n] + w["legs"] * l[n] + w["shorts"] * sh[n]
                       - w["other_torso"] * other_t - w["other_shorts"] * other_s)
        return tuple(out)

    # track-level colour averages smooth out bad single frames
    acc = defaultdict(lambda: np.zeros(3))
    for fr in frames:
        for p in fr:
            if p["id"] >= 0 and p["app"] is not None:
                acc[p["id"]] += (*colour(p), 1)
    tcol = {k: (v[0] / v[2], v[1] / v[2]) for k, v in acc.items() if v[2] >= 5}

    def centre(p):
        b = p["box"]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    ring = (CFG.data.get("ring") or {})
    ring_poly = np.array(ring["polygon"], np.float32) if ring.get("polygon") else None
    ring_margin = float(ring.get("margin", 15))

    def in_ring(p):
        """Optional: only people whose feet are on the canvas (config `ring.polygon`, clip pixels)
        can be fighters - keeps corner teams and front-row spectators out."""
        if ring_poly is None:
            return True
        x1, y1, x2, y2 = p["box"]
        return cv2.pointPolygonTest(ring_poly, (float((x1 + x2) / 2), float(y2)), True) > -ring_margin

    tlab = track_label_fn(track_labels or {})
    global CHOSEN_IDS
    CHOSEN_IDS = {n: np.full(F, -1, int) for n in NAMES}  # which track each fighter came from, per frame
    K = {n: np.full((F, 17, 3), np.nan) for n in NAMES}
    last = {n: None for n in NAMES}  # (id, centre, frame, height)
    heights = [180.0]
    for i, fr in enumerate(frames):
        ref_h = float(np.median(heights[-90:]))
        cands = [p for p in fr if (p["box"][3] - p["box"][1]) >= 0.6 * ref_h and in_ring(p)
                 and tlab(p["id"], i) != "ignore"]

        def fscore(p, n):
            lab = tlab(p["id"], i)
            if lab is not None:  # a human-labelled track decides it
                return 10.0 if lab == n else -1.0
            c = colour(p)
            tc = tcol.get(p["id"], c)
            col = 0.5 * (c[0] + tc[0]) if n == F0 else 0.5 * (c[1] + tc[1])
            sc = col
            lp = last[n]
            if lp is not None and i - lp[2] < 30:
                if p["id"] == lp[0]:
                    sc += 0.35
                elif col < 0.3:
                    return -1.0  # don't hand the label to a different, weakly-coloured person
                sc -= 0.25 * np.linalg.norm(centre(p) - lp[1]) / lp[3]
            return sc

        best, bs = (None, None), 0.0
        opts = [None] + list(range(len(cands)))
        for gi in opts:
            for wi in opts:
                if gi is not None and gi == wi:
                    continue
                s = 0.0
                if gi is not None:
                    v = fscore(cands[gi], F0)
                    if v < 0.15:
                        continue
                    s += v
                if wi is not None:
                    v = fscore(cands[wi], F1)
                    if v < 0.15:
                        continue
                    s += v
                if s > bs:
                    best, bs = (gi, wi), s
        for n, idx in zip(NAMES, best):
            if idx is None:
                continue
            p = cands[idx]
            h = p["box"][3] - p["box"][1]
            K[n][i] = p["kpts"]
            CHOSEN_IDS[n][i] = p["id"]
            last[n] = (p["id"], centre(p), i, h)
            heights.append(h)
    for n in NAMES:
        k = K[n]
        k[k[..., 2] < KP_CONF] = np.nan
        for j in range(17):
            for c in range(2):
                k[:, j, c] = nan_interp(k[:, j, c], 6)
    return K


# ---------------------------------------------------------------- 3-4. events
class Body:
    """Per-frame derived geometry for one fighter."""

    def __init__(self, K, cam):
        self.K = K
        self.Kc = K.copy()
        self.Kc[..., 0] -= cam[:, None, 0]
        self.Kc[..., 1] -= cam[:, None, 1]
        sh, hip = mid(K, L_SH, R_SH), mid(K, L_HIP, R_HIP)
        s = np.linalg.norm(sh - hip, axis=1)
        med = np.nanmedian(s)
        s = np.array([np.nanmedian(s[max(0, i - 15):i + 16]) if not np.isnan(s[max(0, i - 15):i + 16]).all()
                      else med for i in range(len(s))])
        self.s = np.where(np.isnan(s), med, s)
        self.neck, self.hip = sh, hip
        head = np.nanmean(K[:, :5, :2], axis=1)
        fallback = sh + (sh - hip) * 0.45
        self.head = np.where(np.isnan(head), fallback, head)
        self.centre = (sh + hip) / 2
        cc = (mid(self.Kc, L_SH, R_SH) + mid(self.Kc, L_HIP, R_HIP)) / 2
        self.vel = np.vstack([np.zeros((1, 2)), np.diff(
            np.column_stack([smooth(cc[:, 0], 9), smooth(cc[:, 1], 9)]), axis=0)])
        self.present = ~np.isnan(self.centre).any(axis=1)

    def torso_quad(self, t):
        q = self.K[t, [L_SH, R_SH, R_HIP, L_HIP], :2]
        return None if np.isnan(q).any() else q


def in_round(t, fps):
    sec = t / fps
    for r, (a, b) in enumerate(ROUNDS):
        if a <= sec < b:
            return r
    return None


def hit_test(att, dfn, p, t, kind):
    """Return (target, normalised distance) if point p touches defender at t."""
    s = dfn.s[t]
    best = None
    hd = np.linalg.norm(p - dfn.head[t]) / s
    if not np.isnan(hd) and hd < 0.5:
        best = ("head", hd)
    if kind != "punch" or best is None:
        q = dfn.torso_quad(t)
        if q is not None:
            c = q.mean(0)
            qe = (c + (q - c) * 1.15).astype(np.float32)
            d = cv2.pointPolygonTest(qe, (float(p[0]), float(p[1])), True)
            if d > -0.1 * s and best is None:
                best = ("body", max(0.0, -d / s))
    if kind == "kick" and best is None:
        K = dfn.K[t]
        for a, b in ((L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN)):
            if np.isnan(K[[a, b], :2]).any():
                continue
            d = seg_dist(p, K[a, :2], K[b, :2]) / s
            if d < 0.22:
                best = ("leg", d)
                break
    return best


def detect_strikes(A, D, an, fps, F, clinch):
    events = []
    last_limb = defaultdict(lambda: -99)
    for j, kind, root in LIMBS:
        P = A.Kc[:, j, :2]
        px, py = smooth(P[:, 0], 3), smooth(P[:, 1], 3)
        v = np.hypot(np.diff(px, prepend=np.nan), np.diff(py, prepend=np.nan)) / A.s
        thr = SPEED_THR[kind]
        for t in range(4, F - 6):
            if not (v[t] > thr and v[t] == np.nanmax(v[t - 3:t + 4])):
                continue
            r = in_round(t, fps)
            if r is None or not (A.present[t] and D.present[t]):
                continue
            if t - last_limb[j] < 8:
                continue
            s = A.s[t]
            gap = np.linalg.norm(A.centre[t] - D.centre[t]) / s
            if gap > 3.5:
                continue
            # depth sanity: similar apparent size = similar distance from camera
            if abs(A.s[t] - D.s[t]) / max(A.s[t], D.s[t]) > 0.35:
                continue
            # from a single camera, a fighter standing in front of the other "touches"
            # them everywhere in 2D. Only knees (clinch) are allowed when stacked.
            # in the clinch, arms wrap the head; only knees (and kicks) score there
            if kind == "punch" and clinch[max(0, t - 3):t + 4].any():
                continue
            stacked = abs(A.centre[t, 0] - D.centre[t, 0]) / s < 0.6
            if stacked and kind != "knee":
                continue
            if kind == "punch" and gap > 2.6:
                continue
            toward = D.centre[t] - A.centre[t]
            mv = A.K[t, j, :2] - A.K[max(0, t - 3), j, :2]
            ok_shape = False
            for w in range(t - 2, t + 5):
                K = A.K[w]
                if kind == "punch":
                    ext = np.linalg.norm(K[j, :2] - K[root, :2]) / A.s[w]
                    ok_shape |= ext > 0.75 and np.dot(mv, toward) > 0
                elif kind == "kick":
                    feet_y = np.nanmax(K[[L_AN, R_AN], 1])
                    ok_shape |= (feet_y - K[j, 1]) / A.s[w] > 0.45
                else:
                    ok_shape |= (K[j, 1] - K[root, 1]) / A.s[w] < 0.45
            if not ok_shape:
                continue
            landed = None
            for w in range(t - 2, t + 5):
                p = A.K[w, j, :2]
                if np.isnan(p).any():
                    continue
                h = hit_test(A, D, p, w, kind)
                if h and (landed is None or h[1] < landed[2]):
                    landed = (w, h[0], h[1], p.copy())
                    if h[0] == "head":
                        break
            ev = {"t": t, "round": r, "by": an, "kind": kind, "speed": float(v[t]),
                  "landed": landed is not None, "side": "L" if j in (L_WR, L_AN, L_KN) else "R"}
            if landed:
                w, target, _, p = landed
                ev.update(t=w, target=target, x=float(p[0]), y=float(p[1]))
                if target == "head":
                    rel = D.head - D.hip
                    seg = rel[w:w + 7]
                    if not np.isnan(seg).all():
                        disp = np.nanmax(np.linalg.norm(seg - rel[w], axis=1)) / D.s[w]
                        ev["snap"] = bool(disp > 0.35)
            else:
                p = A.K[t, j, :2]
                ev.update(x=float(p[0]), y=float(p[1]))
            last_limb[j] = t
            events.append(ev)
    events.sort(key=lambda e: e["t"])
    # the knee and ankle of one leg move together: keep one event per leg motion
    # (prefer the one that landed, then the kick - a raised ankle means a kick)
    legs, keep = [e for e in events if e["kind"] != "punch"], []
    used = set()
    for a, e in enumerate(legs):
        if a in used:
            continue
        group = [e] + [f for b, f in enumerate(legs) if b > a and b not in used
                       and f["side"] == e["side"] and abs(f["t"] - e["t"]) <= 5]
        used.update(b for b, f in enumerate(legs) if f in group)
        keep.append(max(group, key=lambda f: (f["landed"], f["kind"] == "kick")))
    events = sorted([e for e in events if e["kind"] == "punch"] + keep, key=lambda e: e["t"])
    # one landed strike per attacker per 5 frames (different limbs can fire together)
    out, last_t = [], -99
    for e in events:
        if e["landed"]:
            if e["t"] - last_t < 5:
                continue
            last_t = e["t"]
        out.append(e)
    return out


CLINCH_DIST = 1.2   # torso centres within this many torso-lengths
CLINCH_MIN_S = 1.0  # a tie-up must last this long to count as a clinch
MERGE_DIST = 0.8    # torso centres this close just before one fighter vanishes = pose merge, not a fall


def clinch_and_pressure(B, fps, F):
    G, W = B[F0], B[F1]
    both = G.present & W.present
    s = (G.s + W.s) / 2
    dist = np.linalg.norm(G.centre - W.centre, axis=1) / s
    ctrl = {n: np.zeros(F) for n in NAMES}
    tied = np.zeros(F, bool)
    for t in range(F):
        if not both[t] or dist[t] > CLINCH_DIST:
            continue
        for a, d in ((F0, F1), (F1, F0)):
            A, D = B[a], B[d]
            n = 0
            for j in (L_WR, R_WR):
                p = A.K[t, j, :2]
                if np.isnan(p).any():
                    continue
                if min(np.linalg.norm(p - D.head[t]), np.linalg.norm(p - D.neck[t])) / s[t] < 0.6:
                    n += 1
            # opponent bent over: head close to hip height relative to torso. Only counts
            # if A actually has a hand on D - otherwise D is just ducking or crouching.
            post_d = (D.hip[t, 1] - D.head[t, 1]) / D.s[t]
            post_a = (A.hip[t, 1] - A.head[t, 1]) / A.s[t]
            bent = n > 0 and post_d < 1.0 and post_a > 1.2
            ctrl[a][t] = n + (1.5 if bent else 0)
            tied[t] |= n > 0
    clinch = np.zeros(F, bool)
    segs = [(a, b) for a, b in runs(tied, min_len=int(CLINCH_MIN_S * fps), merge_gap=8)
            if in_round(a, fps) is not None]
    for a, b in segs:
        clinch[a:b] = True

    # forward pressure outside the clinch, camera-compensated
    fwd = {n: np.zeros(F) for n in NAMES}
    for t in range(F):
        if not both[t] or clinch[t] or in_round(t, fps) is None or dist[t] > 4:
            continue
        u = W.centre[t] - G.centre[t]
        u /= max(np.linalg.norm(u), 1e-9)
        fwd[F0][t] = max(0.0, np.dot(G.vel[t], u)) / s[t]
        fwd[F1][t] = max(0.0, np.dot(W.vel[t], -u)) / s[t]
    return clinch, segs, ctrl, fwd, dist


def detect_falls(B, strikes, clinch, dist, fps, F):
    """A fighter is 'down' if (a) the pose says so (hips near feet / head at hip
    height), or (b) they vanish mid-round and come back crouched - the pose model
    loses people lying on the canvas, so the get-up is the reliable signal."""
    down = {}
    for n in NAMES:
        A = B[n]
        feet = np.nanmax(A.K[:, [L_AN, R_AN], 1], axis=1)
        leg = (feet - A.hip[:, 1]) / A.s
        head_low = (A.head[:, 1] - A.hip[:, 1]) / A.s > -0.25
        # hips at foot level = on the canvas anywhere; head at hip height only counts when the
        # fighters are apart (bent over in a clinch is not down)
        apart = np.nan_to_num(dist, nan=9.0) > 1.5 * MERGE_DIST
        m = ((leg < 0.65) | (head_low & apart)) & A.present
        rr = [tuple(r) for r in runs(m, min_len=6, merge_gap=5)]
        upright = (A.hip[:, 1] - A.neck[:, 1]) / A.s  # ~1.0 standing, small when crouched/lying
        xs, ys = A.K[:, :, 0], A.K[:, :, 1]
        with np.errstate(all="ignore"):
            aspect = (np.nanmax(xs, 1) - np.nanmin(xs, 1)) / (np.nanmax(ys, 1) - np.nanmin(ys, 1))
        for g0, g1 in runs(~A.present, min_len=12):
            if g1 - g0 > 5 * fps or g0 < 10 or g1 >= F - 6 or in_round(g0, fps) is None:
                continue
            # vanishing inside a clinch, or while tight against the other fighter, means the pose
            # model merged the two of them - not a fall
            if clinch[max(0, g0 - 15):min(F, g1 + 15)].any():
                continue
            near = dist[max(0, g0 - 30):g0]  # last second they were both visible
            if not np.isnan(near).all() and np.nanmin(near) < MERGE_DIST:
                continue
            # vanishing at the side of the frame means leaving the shot, not going down
            xs_before = A.K[max(0, g0 - 8):g0, :, 0]
            if not np.isnan(xs_before).all() and FRAME_W:
                x_last = np.nanmean(xs_before)
                if x_last < 0.08 * FRAME_W or x_last > 0.92 * FRAME_W:
                    continue
            # going down: pose spreads out (horizontal) or hips drop, just before vanishing
            pre_asp = np.nanmax(aspect[g0 - 8:g0]) if not np.isnan(aspect[g0 - 8:g0]).all() else 0
            pre_drop = (np.nanmax(A.hip[g0 - 8:g0, 1]) - np.nanmin(A.hip[g0 - 8:g0, 1])) / A.s[g0 - 1] \
                if not np.isnan(A.hip[g0 - 8:g0, 1]).all() else 0
            # getting up: clearly crouched for the first few frames back
            back = upright[g1:g1 + 4]
            low_back = (np.nanmedian(back) < 0.6) if not np.isnan(back).all() else False
            if low_back and (pre_asp > 0.5 or pre_drop > 0.4):
                rr.append((g0, g1))
        def at_edge(a):  # half out of frame at the side: legs cut off look like crouching
            xs = A.K[max(0, a - 8):a + 4, :, 0]
            if np.isnan(xs).all() or not FRAME_W:
                return False
            x = np.nanmean(xs)
            return x < 0.08 * FRAME_W or x > 0.92 * FRAME_W

        down[n] = sorted(r for r in rr if in_round(r[0], fps) is not None and not at_edge(r[0]))
    falls = []
    for n in NAMES:
        other = NAMES[1 - NAMES.index(n)]
        for a, b in down[n]:
            both = any(abs(x - a) < 5 * fps or (x < b and a < y) for x, y in down[other])
            w0 = max(0, a - int(1.5 * fps))
            hit_by = [e for e in strikes if e["by"] == other and e["landed"] and a - fps <= e["t"] <= a]
            own_kick = any(e["by"] == n and e["kind"] == "kick" and w0 <= e["t"] <= a for e in strikes)
            close = np.nanmin(dist[w0:a + 1]) < 1.6 if not np.isnan(dist[w0:a + 1]).all() else False
            if both:
                cause, credit = "both went down", None
            elif clinch[w0:a + 1].any() or (own_kick and close):
                cause, credit = "dump/sweep", other
            elif hit_by:
                cause, credit = "knockdown?", other
            elif close:
                cause, credit = "dump/sweep", other
            else:
                cause, credit = "slip", None
            falls.append({"t": int(a), "end": int(b), "who": n, "cause": cause, "credit": credit,
                          "round": in_round(a, fps)})
    falls.sort(key=lambda f: f["t"])
    # merge repeated detections of the same fall (same fighter within 2s, or one tangle)
    out = []
    for f in falls:
        if out and f["t"] - out[-1]["t"] < 2 * fps and (
                f["who"] == out[-1]["who"] or f["cause"] == out[-1]["cause"] == "both went down"):
            out[-1]["end"] = max(out[-1]["end"], f["end"])
            continue
        out.append(f)
    return out




# ---------------------------------------------------------------- clinch from labels
CLINCH_WORDS = ("clinch", "tie-up", "tie up", "ties up", "neck", "wrap", "body-lock", "body lock")
END_WORDS = ("end", "break", "separate", "split")
EVEN_WORDS = ("even", "mutual", "unclear", "no clear", "neither", "ducks under")
CTRL_RE = None  # built from the fighter ids in configure()


def clinch_from_labels(label_files, F, fps):
    """Clinch time and control from the labellers' event notes.

    Events within 4 s of each other form one clinch, which ends at an 'end/break/
    separate' note. Clinches shorter than CLINCH_MIN_S are dropped. Between two notes,
    control belongs to whoever the earlier note says has hands/arms on the other or is
    driving/turning them; 'mutual/even/unclear' notes give control to nobody.
    """
    clinch = np.zeros(F, bool)
    ctrl = {n: np.zeros(F) for n in NAMES}
    segs = []
    for lf in label_files:
        evs = []
        for e in json.load(open(lf)).get("events", []):
            note = (e.get("note") or "").lower()
            if not any(w in note for w in CLINCH_WORDS):
                continue
            end = any(w in note for w in END_WORDS) and "continues" not in note
            who = None
            if not any(w in note for w in EVEN_WORDS):
                m = CTRL_RE.search(note)
                if m:
                    who = m.group(1).upper()
                elif str(e.get("attacker", "")).upper() in NAMES and "control" in note:
                    who = str(e["attacker"]).upper()
            joins = end or "continu" in note  # an end/continues note belongs to the open clinch
            evs.append((int(e["t"]), end, who, joins))
        evs.sort(key=lambda x: x[0])
        cur = []
        for t, end, who, joins in evs:
            if cur and t - cur[-1][0] > (10 if joins else 4) * fps:
                segs.append(cur)
                cur = []
            cur.append((t, end, who))
            if end:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
    out = []
    for seg in segs:
        a = seg[0][0]
        b = seg[-1][0] if seg[-1][1] else seg[-1][0] + int(fps)
        if b - a < CLINCH_MIN_S * fps or in_round(a, fps) is None:
            continue
        clinch[a:b] = True
        for (t0, _, who), nxt in zip(seg, seg[1:] + [(b, True, None)]):
            if who:
                ctrl[who][t0:nxt[0]] = 1.0
        out.append((a, b))
    return clinch, out, ctrl


KD_POS = re.compile(r"\bknock(?:ed)?[ -]?down\b")
KD_NEG = re.compile(r"\b(?:no|not|without)\s+(?:a\s+)?(?:\w+\s+)?knock(?:ed)?[ -]?down|no fall")


def merge_labelled_knockdowns(falls, label_files, fps):
    """Knockdowns the labellers saw become falls credited to the fighter they name; any
    pose-detected fall within 4 s of one is replaced by it (the labels saw the cause)."""
    kds = []
    for lf in label_files:
        for e in json.load(open(lf)).get("events", []):
            note = (e.get("note") or "").lower()
            att = str(e.get("attacker", "")).upper()
            if att not in NAMES or KD_NEG.search(note):
                continue
            if KD_POS.search(note):
                if any(abs(e["t"] - k["t"]) < 4 * fps for k in kds):
                    continue
                kds.append({"t": int(e["t"]), "end": int(e["t"] + fps), "who": NAMES[1 - NAMES.index(att)],
                            "cause": "knockdown", "credit": att, "round": in_round(e["t"], fps)})
    kept = [f for f in falls if all(abs(f["t"] - k["t"]) >= 4 * fps for k in kds)]
    return sorted(kept + kds, key=lambda f: f["t"])


# ---------------------------------------------------------------- 5. stats
KIND = {"teep": "kick", "elbow": "punch"}  # fold rarer strikes into the HUD's three columns


def plan(e):
    """Probability a strike landed: calibrated p_land if present, else the hard label."""
    return float(e.get("p_land", 1.0 if e["landed"] else 0.0))


def summarise(strikes, falls, clinch, segs, ctrl, fwd, fps):
    rounds = []
    for r, (a, b) in enumerate(ROUNDS):
        fa, fb = int(a * fps), int(b * fps)
        rs = {"round": r + 1, "clinch_seconds": round(float(clinch[fa:fb].sum() / fps), 1)}
        cg, cw = ctrl[F0][fa:fb][clinch[fa:fb]], ctrl[F1][fa:fb][clinch[fa:fb]]
        tot = cg.sum() + cw.sum()
        for n, c in ((F0, cg), (F1, cw)):
            ev = [e for e in strikes if e["round"] == r and e["by"] == n]
            ld = [e for e in ev if e["landed"]]
            pl = {id(e): plan(e) for e in ev}
            fw_tot = fwd[F0][fa:fb].sum() + fwd[F1][fa:fb].sum()
            rs[n] = {
                "thrown": len(ev),
                "landed": round(sum(pl.values()), 1),
                "verified_clean": sum(1 for e in ev if e.get("verified") and e["outcome"] == "clean"),
                "accuracy": round(sum(pl.values()) / len(ev), 2) if ev else None,
                "landed_by_kind": {k: round(sum(pl[id(e)] for e in ev if KIND.get(e["kind"], e["kind"]) == k), 1)
                                   for k in ("punch", "kick", "knee")},
                "landed_by_target": {k: round(sum(pl[id(e)] for e in ev if e.get("target") == k), 1)
                                     for k in ("head", "body", "leg")},
                "head_snaps": sum(bool(e.get("snap")) for e in ld),
                "clinch_control_share": round(float(c.sum() / tot), 2) if tot else None,
                "forward_pressure_share": round(float(fwd[n][fa:fb].sum() / fw_tot), 2) if fw_tot else None,
                "dumps_scored": sum(1 for f in falls if f["round"] == r and f["credit"] == n
                                    and f["cause"] != "knockdown"),
                "knockdowns_scored": sum(1 for f in falls if f["round"] == r and f["credit"] == n
                                         and f["cause"] == "knockdown"),
                "times_down": sum(1 for f in falls if f["round"] == r and f["who"] == n),
            }
        rounds.append(rs)
    return rounds


def score_round(rs):
    """Rough Muay Thai weighting: kicks/knees > punches, effect, dumps, control."""
    pts = {}
    for n in NAMES:
        x = rs[n]
        k = x["landed_by_kind"]
        p = k["punch"] * 1.0 + k["kick"] * 1.6 + k["knee"] * 1.4 + x["head_snaps"] * 1.5
        p += x["dumps_scored"] * 3.0 + x.get("knockdowns_scored", 0) * 6.0
        p += 4.0 * (x["clinch_control_share"] or 0.5) * min(rs["clinch_seconds"], 40) / 40
        p += 3.0 * (x["forward_pressure_share"] or 0.5)
        pts[n] = round(p, 1)
    return pts


# ---------------------------------------------------------------- render
def txt(img, s, org, scale=0.6, col=(255, 255, 255), th=1, font=cv2.FONT_HERSHEY_DUPLEX):
    """Text with a dark outline. The outline is the same text at the same thickness, offset by a
    couple of pixels (a thicker stroke changes glyph spacing in OpenCV and leaves a 'ghost')."""
    x, y = org
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(img, s, (x + dx, y + dy), font, scale, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, col, th, cv2.LINE_AA)


def fmt(sec):
    sec = max(0, int(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def render(video, out, B, strikes, falls, clinch, ctrl, fwd, fps, F, rounds_summary, round_pts,
           frame_range=None, placeholder=False):
    S = 2  # upscale
    cap = cv2.VideoCapture(video)
    w0, h0 = int(cap.get(3)), int(cap.get(4))
    W, H, PH = w0 * S, h0 * S, 170
    tmp = out.replace(".mp4", "_noaudio.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + PH))
    landed = [e for e in strikes if plan(e) > 0]
    cum = {n: defaultdict(float) for n in NAMES}
    ptr = 0
    cg = np.cumsum(ctrl[F0] * clinch)
    cw = np.cumsum(ctrl[F1] * clinch)
    fg, fw = np.cumsum(fwd[F0]), np.cumsum(fwd[F1])
    trail = 8
    tl_x0, tl_x1 = 20, W - 20
    tl_y = H + PH - 28

    def tx(t):
        return int(tl_x0 + (tl_x1 - tl_x0) * t / F)

    # static timeline background
    tl = np.zeros((PH, W, 3), np.uint8)
    tl[:] = (24, 24, 28)
    for r, (a, b) in enumerate(ROUNDS):
        cv2.rectangle(tl, (tx(a * fps), tl_y - H - 16), (tx(b * fps), tl_y - H + 16), (55, 55, 62), -1)
        txt(tl, f"R{r + 1}", (tx(a * fps) + 4, tl_y - H - 20), 0.45, (180, 180, 180))
    for e in landed:
        x = tx(e["t"])
        n = e["by"]
        y0, y1 = (tl_y - H - 14, tl_y - H - 2) if n == F0 else (tl_y - H + 2, tl_y - H + 14)
        cv2.line(tl, (x, y0), (x, y1), COL[n], 1)
    for f in falls:
        x = tx(f["t"])
        c = COL[f["credit"]] if f["credit"] else (200, 200, 200)
        cv2.drawMarker(tl, (x, tl_y - H), c, cv2.MARKER_TRIANGLE_DOWN, 12, 2)

    f_lo, f_hi = frame_range or (0, F)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    thrown = {n: 0 for n in NAMES}
    for i in range(f_lo, min(f_hi, F)):
        ok, fr = cap.read()
        if not ok:
            break
        if placeholder:
            import placeholder as ph
            fr = ph.frame({n: B[n].K for n in NAMES}, i, w0, h0, COL)
        img = cv2.resize(fr, (W, H), interpolation=cv2.INTER_CUBIC)
        r = in_round(i, fps)
        sec = i / fps
        # skeletons + trails
        for n in NAMES:
            K = B[n].K
            if not B[n].present[i]:
                continue
            c = COL[n]
            for a, b in SKELETON:
                pa, pb = K[i, a, :2], K[i, b, :2]
                if not (np.isnan(pa).any() or np.isnan(pb).any()):
                    cv2.line(img, tuple((pa * S).astype(int)), tuple((pb * S).astype(int)), c, 3, cv2.LINE_AA)
            for j in (L_WR, R_WR, L_AN, R_AN, L_KN, R_KN):
                pts = [K[k, j, :2] for k in range(max(0, i - trail), i + 1)]
                pts = [p for p in pts if not np.isnan(p).any()]
                for k in range(1, len(pts)):
                    th = 1 + 3 * k // len(pts)
                    cv2.line(img, tuple((pts[k - 1] * S).astype(int)), tuple((pts[k] * S).astype(int)),
                             c, th, cv2.LINE_AA)
            hp = B[n].head[i]
            if not np.isnan(hp).any():
                txt(img, n, (int(hp[0] * S) - 30, int(hp[1] * S) - 28), 0.5, c)
        # clinch link
        if clinch[i]:
            a, b = B[F0].neck[i], B[F1].neck[i]
            if not (np.isnan(a).any() or np.isnan(b).any()):
                cv2.line(img, tuple((a * S).astype(int)), tuple((b * S).astype(int)), (0, 220, 255), 2)
                m = ((a + b) / 2 * S).astype(int)
                who = F0 if ctrl[F0][i] > ctrl[F1][i] else (
                    F1 if ctrl[F1][i] > ctrl[F0][i] else None)
                txt(img, "CLINCH" + (f" - {who} control" if who else ""), (m[0] - 70, m[1] - 50), 0.55,
                    COL[who] if who else (0, 220, 255))
        # strike effects
        for e in strikes:
            age = i - e["t"]
            if age < 0 or age > 20:
                continue
            p = (int(e["x"] * S), int(e["y"] * S))
            c = COL[e["by"]]
            ver = e.get("verified")
            if e["landed"]:
                rad = 8 + age * 3
                ov = img.copy()
                cv2.circle(ov, p, rad, c, 4 if ver else 2, cv2.LINE_AA)
                cv2.circle(ov, p, 7 if ver else 4, (255, 255, 255), -1, cv2.LINE_AA)
                a = max(0.0, 1 - age / 20) * (1.0 if ver else 0.35 + 0.5 * plan(e))
                cv2.addWeighted(ov, a, img, 1 - a, 0, img)
                if age < 18:
                    lab = f"{e['kind'].upper()} > {e['target'].upper()}" + ("  SNAP!" if e.get("snap") else "")
                    lab += "  [reviewed]" if ver else (f"  ~{int(100 * plan(e))}%" if "p_land" in e else "")
                    txt(img, lab, (p[0] + 12, p[1] - 10 - age * 2), 0.6 if ver else 0.5, c, 2 if ver else 1)
            elif e.get("outcome") == "blocked" and age < 14:
                cv2.circle(img, p, 14, c, 2, cv2.LINE_AA)
                cv2.line(img, (p[0] - 10, p[1] - 10), (p[0] + 10, p[1] + 10), c, 2, cv2.LINE_AA)
                if age < 12:
                    txt(img, f"{e['kind'].upper()} > {e['target'].upper()}  BLOCKED", (p[0] + 12, p[1] - 10 - age),
                        0.5, (200, 200, 200), 1)
            elif age < 8:
                cv2.circle(img, p, 10, (150, 150, 150), 1, cv2.LINE_AA)
        # fall banners
        for f in falls:
            age = i - f["t"]
            if 0 <= age < 50:
                msg = {"dump/sweep": f"DUMP / SWEEP  -  {f['credit']} scores",
                       "knockdown": f"KNOCKDOWN  -  {f['credit']} scores",
                       "knockdown?": f"KNOCKDOWN?  -  {f['credit']}",
                       "both went down": "BOTH FIGHTERS DOWN",
                       "slip": f"{f['who']} SLIPS"}[f["cause"]]
                c = COL[f["credit"]] if f["credit"] else (230, 230, 230)
                cv2.rectangle(img, (W // 2 - 260, 60), (W // 2 + 260, 104), (0, 0, 0), -1)
                txt(img, msg, (W // 2 - 240, 92), 0.85, c, 2)
        # top bar
        if r is not None:
            st = f"ROUND {r + 1}   {fmt(sec - ROUNDS[r][0])}"
        elif sec < ROUNDS[0][0]:
            st = "PRE-FIGHT"
        elif sec >= ROUNDS[-1][1]:
            st = "DECISION"
        else:
            st = "BREAK"
        cv2.rectangle(img, (0, 0), (W, 40), (0, 0, 0), -1)
        txt(img, st, (16, 28), 0.8, (255, 255, 255), 2)
        yt = int(TIME_OFFSET + sec)
        txt(img, f"src {yt // 3600}:{(yt % 3600) // 60:02d}:{yt % 60:02d}", (W - 170, 28), 0.6, (180, 180, 180))

        # cumulative counters
        while ptr < len(landed) and landed[ptr]["t"] <= i:
            e = landed[ptr]
            q = plan(e)
            cum[e["by"]][KIND.get(e["kind"], e["kind"])] += q
            cum[e["by"]][e["target"]] += q
            cum[e["by"]]["snap"] += int(bool(e.get("snap")) and e["landed"])
            cum[e["by"]]["all"] += q
            cum[e["by"]]["ver"] += int(bool(e.get("verified")) and e["outcome"] == "clean")
            ptr += 1
        if i % 15 == 0 or i == f_lo:
            thrown = {n: sum(1 for e in strikes if e["by"] == n and e["t"] <= i) for n in NAMES}
        panel = tl.copy()
        for col_i, n in enumerate(NAMES):
            x0 = 20 if col_i == 0 else W // 2 + 170
            c = COL[n]
            d = cum[n]
            txt(panel, f"{n}", (x0, 30), 0.75, c, 2)
            txt(panel, f"landed ~{d['all']:.0f}/{thrown[n]}   punch {d['punch']:.0f}  kick {d['kick']:.0f}  "
                       f"knee {d['knee']:.0f}", (x0, 58), 0.5)
            txt(panel, f"head {d['head']:.0f}  body {d['body']:.0f}  leg {d['leg']:.0f}", (x0, 82), 0.5)
            dn = sum(1 for f in falls if f["credit"] == n and f["t"] <= i)
            txt(panel, f"reviewed clean {int(d['ver'])}   dumps {dn}", (x0, 106), 0.5)
        # centre bars: clinch control + pressure
        cx = W // 2 - 140
        for k, (lab, a, b) in enumerate((("clinch control", cg[i], cw[i]), ("forward pressure", fg[i], fw[i]))):
            y = 30 + k * 44
            tot = a + b
            frac = a / tot if tot > 0 else 0.5
            cv2.rectangle(panel, (cx, y), (cx + 300, y + 14), COL[F1], -1)
            cv2.rectangle(panel, (cx, y), (cx + int(300 * frac), y + 14), COL[F0], -1)
            txt(panel, f"{lab}  {int(frac * 100)}% / {int((1 - frac) * 100)}%", (cx, y + 32), 0.45)
        cv2.line(panel, (tx(i), tl_y - H - 18), (tx(i), tl_y - H + 18), (255, 255, 255), 2)
        frame = np.vstack([img, panel])

        # final card
        if sec >= ROUNDS[-1][1] + 2:
            ov = frame.copy()
            cv2.rectangle(ov, (80, 70), (W - 80, H - 40), (10, 10, 10), -1)
            cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
            title = "BY THE NUMBERS (labelled + your review)" if any("p_land" in e for e in strikes) \
                else "BY THE NUMBERS (computer vision estimate)"
            txt(frame, title, (110, 115), 0.8, (255, 255, 255), 2)
            y = 160
            for rs, pts in zip(rounds_summary, round_pts):
                g, wv = rs[F0], rs[F1]
                win = max(pts, key=pts.get) if abs(pts[F0] - pts[F1]) >= 1 else "EVEN"
                a0, a1 = F0[0], F1[0]
                line = (f"R{rs['round']}:  landed~ {a0} {g['landed']:.0f} - {a1} {wv['landed']:.0f}   "
                        f"body {a0} {g['landed_by_target']['body']:.0f} - {a1} {wv['landed_by_target']['body']:.0f}   "
                        f"dumps {a0} {g['dumps_scored']} - {a1} {wv['dumps_scored']}   "
                        f"clinch ctl {a0} {int(100 * (g['clinch_control_share'] or 0))}%   ->  {win}")
                txt(frame, line, (110, y), 0.6, COL.get(win, (230, 230, 230)))
                y += 40
            tg = sum(p[F0] for p in round_pts)
            tw = sum(p[F1] for p in round_pts)
            txt(frame, f"weighted score  {F0} {tg:.1f}  -  {F1} {tw:.1f}", (110, y + 20), 0.7, (255, 255, 255), 2)
            if CFG.official_result:
                txt(frame, f"Official result: {CFG.official_result}", (110, y + 70), 0.6, (200, 200, 200))
        vw.write(frame)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-i", video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "libx264", "-crf", "21", "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", out], check=True)
    os.remove(tmp)
    # poster frame for the web UI (a third of the way into what was rendered)
    mid = ((f_hi if frame_range else F) - f_lo) / fps / 3
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{mid:.2f}", "-i", out, "-frames:v", "1",
                    out[:-4] + ".jpg"], check=False)


# ---------------------------------------------------------------- main
def main():
    import fightconfig

    ap = argparse.ArgumentParser(description="Fight analysis: identity, events, stats and overlay video.")
    fightconfig.add_arg(ap)
    ap.add_argument("video", nargs="?", help="prepared clip (default: <workdir>/data/fight.mp4)")
    ap.add_argument("tracks", nargs="?", help="track.py output (default: <workdir>/data/tracks.pkl)")
    ap.add_argument("--out", default=None, help="overlay video (default: <workdir>/out/analysed.mp4)")
    ap.add_argument("--json", default=None, help="stats (default: <workdir>/out/stats.json)")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--labels", nargs="*", default=[],
                    help="<workdir>/labels/rN_events*.json files; replace detector strikes for those rounds")
    ap.add_argument("--range", default="", help="render only start:end seconds of the clip")
    ap.add_argument("--clinch-review", default="",
                    help="<review dir>:<answers.json> from make_clinch_review.py")
    ap.add_argument("--placeholder", action="store_true",
                    help="render skeleton mannequins on a drawn ring instead of the real footage")
    args = ap.parse_args()
    cfg = configure(fightconfig.load(args.config))
    args.video = args.video or cfg.video
    args.tracks = args.tracks or cfg.path("data", "tracks.pkl")
    args.out = args.out or cfg.path("out", "analysed.mp4")
    args.json = args.json or cfg.path("out", "stats.json")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)

    d = pickle.load(open(args.tracks, "rb"))
    frames, fps = d["frames"], d["fps"]
    F = len(frames)
    print(f"{F} frames @ {fps:.0f} fps")
    cam = camera_motion(args.video, F)
    print(f"camera drift range x {cam[:, 0].min():.0f}..{cam[:, 0].max():.0f}px")
    app_cache = args.tracks.replace(".pkl", f"_app_{cfg.appearance_key()}.pkl")
    lab_path = cfg.path("labels", "track_labels.json")
    track_labels = json.load(open(lab_path)) if os.path.exists(lab_path) else None
    if track_labels:
        print(f"using {len(track_labels)} labelled tracks from {lab_path}")
    K = assign_fighters(frames, appearance(args.video, frames, app_cache), track_labels)
    pickle.dump(K, open(os.path.join(os.path.dirname(args.tracks), "K.pkl"), "wb"))  # used by the label/review tools
    B = {n: Body(K[n], cam) for n in NAMES}
    for n in NAMES:
        inr = np.array([in_round(t, fps) is not None for t in range(F)])
        print(f"{n}: present {B[n].present[inr].mean():.0%} of in-round frames")

    clinch, segs, ctrl, fwd, dist = clinch_and_pressure(B, fps, F)
    strikes = detect_strikes(B[F0], B[F1], F0, fps, F, clinch) + \
        detect_strikes(B[F1], B[F0], F1, fps, F, clinch)
    strikes.sort(key=lambda e: e["t"])
    for lf in args.labels:
        lab = json.load(open(lf))["strikes"]
        rds = {e["round"] for e in lab}
        strikes = [e for e in strikes if e["round"] not in rds] + lab
        strikes.sort(key=lambda e: e["t"])
        print(f"using {len(lab)} labelled strikes for round(s) {sorted(r + 1 for r in rds)}")
    if args.labels:
        # labelled clinch replaces the pose clinch (pose loses fighters when they merge)
        lab_clinch, lab_segs, lab_ctrl = clinch_from_labels(args.labels, F, fps)
        print(f"clinch from labels: {len(lab_segs)} clinches, {lab_clinch.sum() / fps:.0f}s "
              f"(pose had {clinch.sum() / fps:.0f}s)")
        clinch, segs, ctrl = lab_clinch, lab_segs, lab_ctrl
        if args.clinch_review:
            # labeller control calls matched the human review only 2/9 times, so unreviewed
            # clinches get no control credit; reviewed ones get the human verdict.
            rdir, ans_path = args.clinch_review.rsplit(":", 1)
            items = {it["id"]: it for it in json.load(open(f"{rdir}/items.json"))}
            for n in NAMES:
                ctrl[n][:] = 0.0
            for a in json.load(open(ans_path)):
                if "id" not in a:
                    continue  # general comment
                it = items[a["id"]]
                verdict = a.get("control", a.get("you"))  # review page / older hand-pasted format
                t0, t1 = it["t0"], it["t1"]
                if verdict == "not really a clinch":
                    clinch[t0:t1] = False
                elif verdict in NAMES:
                    ctrl[verdict][t0:t1] = 1.0
                elif verdict == "even":
                    for n in NAMES:
                        ctrl[n][t0:t1] = 0.5
            segs = [(a, b) for a, b in segs if clinch[a:b].any()]
            print(f"clinch review applied: {clinch.sum() / fps:.0f}s of clinch, control now from your answers only")
        for n in NAMES:  # no forward pressure while clinched
            fwd[n][clinch] = 0.0
    global FRAME_W
    FRAME_W = (d.get("size") or (0, 0))[0] or int(cv2.VideoCapture(args.video).get(3))
    falls = detect_falls(B, strikes, clinch, dist, fps, F)
    if args.labels:
        falls = merge_labelled_knockdowns(falls, args.labels, fps)
    summary = summarise(strikes, falls, clinch, segs, ctrl, fwd, fps)
    pts = [score_round(rs) for rs in summary]

    def stamp(t):
        return src_stamp(t, fps)

    for rs, p in zip(summary, pts):
        print(f"\nROUND {rs['round']}  clinch {rs['clinch_seconds']}s  weighted {p}")
        for n in NAMES:
            print(f"  {n}: {json.dumps(rs[n])}")
    print("\nFALLS:")
    for f in falls:
        print(f"  {stamp(f['t'])} {f['who']} down ({f['cause']}, credit {f['credit']})")
    json.dump({"rounds": summary, "weighted": pts,
               "falls": [{**f, "src_time": stamp(f["t"])} for f in falls],
               "strikes": [{**e, "src_time": stamp(e["t"])} for e in strikes]},
              open(args.json, "w"), indent=1, default=float)

    if not args.no_render:
        fr_rng = None
        if args.range:
            a, b = (float(x) for x in args.range.split(":"))
            fr_rng = (int(a * fps), int(b * fps))
        render(args.video, args.out, B, strikes, falls, clinch, ctrl, fwd, fps, F, summary, pts, fr_rng,
               placeholder=args.placeholder)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
