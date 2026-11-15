"""Regenerate the README images in docs/ from a real fight, without showing real footage.

    uv run python docs/make_screenshots.py                       # uses private/fight.json
    uv run python docs/make_screenshots.py --config private/fight.json --overlay-at 392.4

What it does:
  1. builds a demo workdir (default private/demo) next to the fight: links to its pose
     data, copies of its labels and clinch answers, and a config with a neutral timecode
  2. renders everything there in --placeholder mode (tracked skeletons drawn as mannequins
     on a drawn ring): review pages, an overlay clip, one labelling sheet
  3. starts the pipeline UI on the demo workdir, runs one job so the log has output, and
     screenshots the UI and both review pages with headless Chrome

Needs: the fight's tracks + keypoints (track.py, analyze.py) and labels (merge_labels.py);
Chrome or Chromium ($CHROME to point at it).
"""
import argparse
import glob
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fightconfig  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
PY = sys.executable


def sh(cmd, env=None):
    print("+", " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) else c for c in cmd))
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})}, check=True,
                   stdout=subprocess.DEVNULL)


def find_chrome():
    for c in (os.environ.get("CHROME"),
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium",
              shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chromium-browser")):
        if c and os.path.exists(c):
            return c
    sys.exit("Chrome/Chromium not found; set $CHROME")


def build_demo(src, demo_dir):
    """A workdir that reuses the fight's data but only ever holds placeholder media."""
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    for d in ("data", "labels/review", "labels/strips", "out"):
        os.makedirs(os.path.join(demo_dir, d))
    for f in ["fight.mp4", "tracks.pkl", "K.pkl"] + [os.path.basename(p) for p in
                                                     glob.glob(src.path("data", "tracks_app_*.pkl"))]:
        target = os.path.abspath(src.path("data", f))
        if os.path.exists(target):
            os.symlink(target, os.path.join(demo_dir, "data", f))
    for p in glob.glob(src.path("labels", "r*_events*.json")):
        shutil.copy(p, os.path.join(demo_dir, "labels"))
    for p in glob.glob(src.path("labels", "strips", "manifest_r*.json")):
        shutil.copy(p, os.path.join(demo_dir, "labels", "strips"))
    data = json.loads(json.dumps(src.data))
    data.update(name="demo (placeholder media)", workdir=os.path.relpath(demo_dir, ROOT), official_result=None)
    data["video"] = {"source": "path/to/full-event.mp4", "start": "0:00:00"}  # neutral timecode
    cfg_path = os.path.join(demo_dir, "fight.json")
    json.dump(data, open(cfg_path, "w"), indent=2)
    return cfg_path


def both_tracked(K, t):
    import numpy as np

    return all(t < len(K[n]) and not np.isnan(K[n][t, [5, 6, 11, 12], 0]).any() for n in K)


def pick_moment(cfg, K):
    """A clean strike (reviewed, if there was a review) that reads well in one frame: both
    fighters tracked for as long as possible around it, and no other strike right next to it."""
    cands = []
    for r in range(1, len(cfg.rounds) + 1):
        for name in (f"r{r}_events_corrected.json", f"r{r}_events.json"):
            p = cfg.path("labels", name)
            if not os.path.exists(p):
                continue
            ev = json.load(open(p))["strikes"]
            for e in ev:
                if e["outcome"] != "clean" or ("p_land" in e and not e.get("verified")):
                    continue
                span = sum(both_tracked(K, t) for t in range(e["t"] - 45, e["t"] + 45))
                crowd = sum(1 for o in ev if o is not e and abs(o["t"] - e["t"]) <= 12)
                cands.append((span - 30 * crowd, r, e["t"] / 30.0))
            break
    if not cands:
        sys.exit("no clean strike found to centre the overlay on; pass --overlay-at")
    _, r, t = max(cands)
    return r, t


def pick_window(manifest, K, at_frame):
    """The labelling window near a moment with both fighters visible in the most tiles."""
    near = [m for m in manifest if abs((m["frames"][0] + m["frames"][-1]) / 2 - at_frame) <= 60]
    return max(near or manifest, key=lambda m: (sum(both_tracked(K, t) for t in m["frames"]),
                                                -abs(m["frames"][0] - at_frame)))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def api(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None)
    return json.load(urllib.request.urlopen(req, timeout=10))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fightconfig.add_arg(ap)
    ap.add_argument("--demo-workdir", default="private/demo")
    ap.add_argument("--overlay-at", type=float, default=None,
                    help="clip second for the overlay image (default: a reviewed clean strike + 0.4 s)")
    args = ap.parse_args()
    src = fightconfig.load(args.config)
    demo_dir = os.path.join(ROOT, args.demo_workdir)
    cfg_path = build_demo(src, demo_dir)
    env = {"FIGHT_CONFIG": cfg_path}
    demo = fightconfig.load(cfg_path)
    K = pickle.load(open(src.path("data", "K.pkl"), "rb"))

    # review pages (placeholder clips); the clinch review reuses the fight's answers if any
    sh([PY, "make_review.py", "--name", "v1", "--placeholder"], env)
    sh([PY, "make_clinch_review.py", "--name", "v1", "--placeholder"], env)
    ans = src.path("labels", "review", "clinch_answers_v1.json")
    clinch = []
    if os.path.exists(ans):
        shutil.copy(ans, demo.path("labels", "review"))
        clinch = ["--clinch-review", f"{demo.path('review_clinch_v1')}:{demo.path('labels', 'review', 'clinch_answers_v1.json')}"]

    # overlay: a 10 s placeholder render around a strike, one frame saved
    if args.overlay_at is None:
        rnd, t = pick_moment(src, K)
        at = t + 0.4
    else:
        at = args.overlay_at
        rnd = next((i + 1 for i, (a, b) in enumerate(src.rounds) if a <= at < b), len(src.rounds))
    start = max(0.0, at - 4.4)
    labels = []
    for r in range(1, len(src.rounds) + 1):
        for name in (f"r{r}_events_corrected.json", f"r{r}_events.json"):
            if os.path.exists(demo.path("labels", name)):
                labels.append(demo.path("labels", name))
                break
    out = demo.path("out", "overlay_placeholder.mp4")
    sh([PY, "analyze.py", "--placeholder", "--range", f"{start}:{start + 10}", "--out", out,
        *(["--labels", *labels] if labels else []), *clinch], env)
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{at - start:.2f}", "-i", out, "-frames:v", "1",
        os.path.join(DOCS, "overlay.png")])
    sh(["ffmpeg", "-v", "error", "-y", "-ss", f"{at - start:.2f}", "-i", out, "-frames:v", "1", out[:-4] + ".jpg"])

    # one labelling sheet: the window containing that moment
    strips = demo.path("strips_placeholder")
    sh([PY, "make_strips.py", "--round", str(rnd), "--placeholder", "--outdir", strips], env)
    import cv2

    manifest = json.load(open(os.path.join(strips, f"manifest_r{rnd}.json")))
    win = pick_window(manifest, K, round(at * 30))
    sheet = cv2.imread(os.path.join(strips, win["file"]))
    cv2.imwrite(os.path.join(DOCS, "labelling-sheet.jpg"), cv2.resize(sheet, (900, 900)),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    # the UI and the review pages, served from the demo workdir
    port = free_port()
    server = subprocess.Popen([PY, "web/server.py", "--port", str(port)], cwd=ROOT,
                              env={**os.environ, **env}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                api(port, "/api/state")
                break
            except OSError:
                time.sleep(0.5)
        job = api(port, "/api/run", {"step": "clinch_review", "params": {"name": "v1", "placeholder": True}})["id"]
        for _ in range(300):
            if api(port, f"/api/job/{job}")["status"] not in ("queued", "running"):
                break
            time.sleep(1)
        chrome = find_chrome()
        for page, name, size in (("/", "pipeline-ui.png", "1440,1250"),
                                 ("/files/review_v1/index.html", "strike-review.png", "1200,1400"),
                                 ("/files/review_clinch_v1/index.html", "clinch-review.png", "1200,1000")):
            sh([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={size}",
                "--virtual-time-budget=8000", f"--screenshot={os.path.join(DOCS, name)}",
                f"http://127.0.0.1:{port}{page}"])
    finally:
        server.terminate()
    print("updated:", ", ".join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(DOCS, "*.[pj][np]g")))))


if __name__ == "__main__":
    main()
