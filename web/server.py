"""Local web UI for running the whole pipeline end to end.

    uv run python web/server.py                      # http://127.0.0.1:8765
    uv run python web/server.py --config private/fight.json --port 9000

Edit the fight config, run each stage as a background job (jobs run one at a time, in
the order queued), open review pages (their Save button writes answers into the fight's
workdir), and look at the stats and rendered videos. Binds to localhost only.
"""
import argparse
import glob
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fightconfig  # noqa: E402

CONFIG_PATH = None
JOBS = {}            # id -> job dict
QUEUE = deque()
LOCK = threading.Lock()


def cfg():
    return fightconfig.load(CONFIG_PATH)


def wd(*p):
    c = cfg()
    base = c.workdir if os.path.isabs(c.workdir) else os.path.join(ROOT, c.workdir)
    return os.path.join(base, *p)


# ---------------------------------------------------------------- jobs
def commands(step, p):
    """Map a UI step + params to one or more script invocations."""
    c = cfg()
    py = [sys.executable]
    base = ["--config", CONFIG_PATH]
    rounds = [int(r) for r in p.get("rounds") or range(1, len(c.rounds) + 1)]
    if step == "prepare":
        return [py + ["prepare_video.py", *base] + (["--force"] if p.get("force") else [])]
    if step == "track":
        return [py + ["track.py", *base]]
    if step == "analyze":
        return [py + ["analyze.py", *base, "--no-render"]]
    if step == "strips":
        return [py + ["make_strips.py", *base, "--round", str(r)] for r in rounds]
    if step == "label":
        extra = ["--provider", p.get("provider", "anthropic")]
        for k in ("model", "base_url", "api_key_env", "effort", "json_mode", "windows"):
            if p.get(k):
                extra += ["--" + k.replace("_", "-"), str(p[k])]
        if p.get("run"):
            extra += ["--run", p["run"]]
        return [py + ["label_strips.py", *base, "--round", str(r), *extra] for r in rounds]
    if step == "merge":
        extra = []
        if p.get("labels_dir"):
            extra += ["--labels-dir", wd("labels", "api", p["labels_dir"])]
            if p.get("promote"):
                extra += ["--promote"]
        return [py + ["merge_labels.py", *base, "--round", str(r), *extra] for r in rounds]
    if step == "review":
        extra = ["--name", p.get("name", "v1")]
        if p.get("random"):
            extra += ["--random", "--outcome", "unclear"]
        if p.get("exclude"):
            extra += ["--exclude", wd(f"review_{p['exclude']}", "items.json")]
        if p.get("placeholder"):
            extra += ["--placeholder"]
        return [py + ["make_review.py", *base, *extra]]
    if step == "clinch_review":
        extra = ["--name", p.get("name", "v1")] + (["--placeholder"] if p.get("placeholder") else [])
        return [py + ["make_clinch_review.py", *base, *extra]]
    if step == "apply":
        pairs = review_pairs()
        if not pairs:
            raise ValueError("no saved strike-review answers yet")
        return [py + ["apply_review.py", *base, "--review", *pairs]]
    if step == "render":
        out = wd("out", p.get("out") or "final.mp4")
        cmd = py + ["analyze.py", *base, "--out", out, "--json", wd("out", "stats.json")]
        labels = [f for r in range(1, len(c.rounds) + 1)
                  for f in [wd("labels", f"r{r}_events_corrected.json"), wd("labels", f"r{r}_events.json")]
                  if os.path.exists(f)]
        per_round = {}
        for f in labels:  # prefer corrected over raw, per round
            per_round.setdefault(os.path.basename(f).split("_")[0], f)
        if per_round:
            cmd += ["--labels", *per_round.values()]
        cl = clinch_pair()
        if cl:
            cmd += ["--clinch-review", cl]
        if p.get("placeholder"):
            cmd += ["--placeholder"]
        if p.get("range"):
            cmd += ["--range", p["range"]]
        return [cmd]
    raise ValueError(f"unknown step {step}")


def review_pairs():
    pairs = []
    for d in sorted(glob.glob(wd("review_*", "items.json"))):
        name = os.path.basename(os.path.dirname(d))
        if name.startswith("review_clinch"):
            continue
        tag = name[len("review_"):]
        for f in (wd("labels", "review", f"answers_{tag}.json"), wd("labels", "review", f"corrections_{tag}.json")):
            if os.path.exists(f):
                pairs.append(f"{os.path.dirname(d)}:{f}")
                break
    return pairs


def clinch_pair():
    for d in sorted(glob.glob(wd("review_clinch_*", "items.json")), reverse=True):
        tag = os.path.basename(os.path.dirname(d))[len("review_clinch_"):]
        f = wd("labels", "review", f"clinch_answers_{tag}.json")
        if os.path.exists(f):
            return f"{os.path.dirname(d)}:{f}"
    return None


def worker():
    while True:
        with LOCK:
            job = QUEUE.popleft() if QUEUE else None
        if job is None:
            time.sleep(0.3)
            continue
        job["status"], job["started"] = "running", time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            for cmd in job["cmds"]:
                job["log"].append("$ " + " ".join(os.path.relpath(x, ROOT) if os.path.isabs(x) else x for x in cmd[1:]))
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1)
                job["proc"] = proc
                for line in proc.stdout:
                    job["log"].append(line.rstrip())
                proc.wait()
                if proc.returncode != 0:
                    job["status"] = f"failed ({proc.returncode})"
                    break
            else:
                job["status"] = "done"
        except Exception as e:  # noqa: BLE001 - surface anything in the job log
            job["log"].append(f"error: {e}")
            job["status"] = "failed"
        job["ended"] = time.time()
        job.pop("proc", None)


def status():
    c = cfg()
    n = len(c.rounds)
    ex = lambda *p: os.path.exists(wd(*p))  # noqa: E731
    reviews = []
    for d in sorted(glob.glob(wd("review_*", "index.html"))):
        name = os.path.basename(os.path.dirname(d))
        kind = "clinch" if name.startswith("review_clinch") else "strikes"
        tag = name.split("_")[-1]
        ans = wd("labels", "review", ("clinch_answers_" if kind == "clinch" else "answers_") + f"{tag}.json")
        legacy = wd("labels", "review", f"corrections_{tag}.json")
        reviews.append({"name": name, "kind": kind, "url": f"/files/{name}/index.html",
                        "answers": os.path.exists(ans) or (kind == "strikes" and os.path.exists(legacy))})
    stats = None
    if ex("out", "stats.json"):
        try:
            s = json.load(open(wd("out", "stats.json")))
            stats = {"rounds": s["rounds"], "weighted": s["weighted"]}
        except (ValueError, KeyError):
            pass
    return {
        "config_path": os.path.relpath(CONFIG_PATH, ROOT) if os.path.isabs(CONFIG_PATH) else CONFIG_PATH,
        "names": c.names, "rounds": c.rounds, "workdir": c.workdir,
        "have": {
            "video": ex("data", "fight.mp4"), "tracks": ex("data", "tracks.pkl"), "keypoints": ex("data", "K.pkl"),
            "strips": [ex("labels", "strips", f"manifest_r{r}.json") for r in range(1, n + 1)],
            "labels": [ex("labels", f"r{r}_events.json") for r in range(1, n + 1)],
            "corrected": [ex("labels", f"r{r}_events_corrected.json") for r in range(1, n + 1)],
        },
        "api_runs": sorted(os.path.basename(p) for p in glob.glob(wd("labels", "api", "*"))),
        "reviews": reviews,
        "videos": sorted(os.path.relpath(p, wd()) for p in glob.glob(wd("out", "*.mp4"))
                         if not p.endswith("_noaudio.mp4")),
        "stats": stats,
        "jobs": [{k: j[k] for k in ("id", "step", "status", "queued")} for j in
                 sorted(JOBS.values(), key=lambda j: j["queued"], reverse=True)[:12]],
    }


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.send_file(os.path.join(ROOT, "web", "index.html"))
        if path == "/api/state":
            return self.send_json(status())
        if path == "/api/config":
            return self.send_json({"path": CONFIG_PATH, "text": open(CONFIG_PATH).read()})
        if path.startswith("/api/job/"):
            j = JOBS.get(path.rsplit("/", 1)[-1])
            if not j:
                return self.send_json({"error": "no such job"}, 404)
            return self.send_json({k: (list(j[k])[-400:] if k == "log" else j[k])
                                   for k in ("id", "step", "status", "log")})
        if path.startswith("/files/"):
            full = os.path.realpath(wd(path[len("/files/"):]))
            if not full.startswith(os.path.realpath(wd()) + os.sep):
                return self.send_json({"error": "outside workdir"}, 403)
            return self.send_file(full)
        self.send_json({"error": "not found"}, 404)

    def send_file(self, full):
        if not os.path.isfile(full):
            return self.send_json({"error": "not found"}, 404)
        size = os.path.getsize(full)
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):  # video seeking
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = min(int(b), size - 1) if b else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(full, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self.body()
        except ValueError:
            return self.send_json({"error": "bad json"}, 400)
        if path == "/api/config":
            try:
                parsed = json.loads(data["text"])
                fightconfig.FightConfig(parsed, CONFIG_PATH)  # validates
            except (ValueError, KeyError, SystemExit) as e:
                return self.send_json({"error": f"invalid config: {e}"}, 400)
            with open(CONFIG_PATH, "w") as f:
                f.write(data["text"])
            return self.send_json({"ok": True})
        if path == "/api/run":
            try:
                cmds = commands(data["step"], data.get("params") or {})
            except (ValueError, KeyError) as e:
                return self.send_json({"error": str(e)}, 400)
            job = {"id": uuid.uuid4().hex[:8], "step": data["step"], "status": "queued", "queued": time.time(),
                   "cmds": cmds, "log": deque(maxlen=4000)}
            with LOCK:
                JOBS[job["id"]] = job
                QUEUE.append(job)
            return self.send_json({"id": job["id"]})
        if path == "/api/cancel":
            j = JOBS.get(data.get("id", ""))
            if j and j.get("proc"):
                j["proc"].terminate()
            elif j and j["status"] == "queued":
                with LOCK:
                    if j in QUEUE:
                        QUEUE.remove(j)
                j["status"] = "cancelled"
            return self.send_json({"ok": True})
        if path == "/api/answers":
            rel = os.path.normpath(data.get("path", ""))
            if not (rel.startswith("labels/review/") and rel.endswith(".json")) or ".." in rel:
                return self.send_json({"error": "answers must go under labels/review/*.json"}, 400)
            full = wd(rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                json.dump(data["answers"], f, indent=1)
            return self.send_json({"ok": True, "path": rel})
        self.send_json({"error": "not found"}, 404)


def main():
    global CONFIG_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fightconfig.add_arg(ap)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    CONFIG_PATH = os.path.abspath(fightconfig.load(args.config).path_)
    os.chdir(ROOT)
    threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"pipeline UI on http://127.0.0.1:{args.port}  (config {os.path.relpath(CONFIG_PATH, ROOT)})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
