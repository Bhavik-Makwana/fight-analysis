"""Offline test of label_strips.py: both backends against a local fake API server.

Checks the request each SDK actually sends (images, schema, system prompt, fallback
flag) and that replies are parsed, filtered, written as JSONL and resumable.
Self-contained: builds a temporary fight config and dummy sheets. No API key, no network.

    uv run python tests/test_label_strips.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUESTS = []
ITEMS = [  # one good item, one for a window that wasn't sent (must be dropped)
    {"window": "r1_w001", "tile": 5, "attacker": "BLUE", "strike": "kick", "target": "body",
     "outcome": "clean", "confidence": 0.6, "note": "body kick"},
    {"window": "r1_w099", "tile": 1, "attacker": "RED", "strike": "punch", "target": "head",
     "outcome": "unclear", "confidence": 0.3, "note": "hallucinated window"},
]


def make_fixture(tmp):
    """A throwaway fight config + three dummy labelling sheets for round 1."""
    work = os.path.join(tmp, "work")
    strips = os.path.join(work, "labels", "strips")
    os.makedirs(strips)
    manifest = []
    for i in range(3):
        name = f"r1_w{i:03d}.jpg"
        cv2.imwrite(os.path.join(strips, name), np.full((120, 120, 3), 80, np.uint8))
        manifest.append({"window": f"r1_w{i:03d}", "file": name, "frames": list(range(i * 36, i * 36 + 16)),
                         "src_start": f"0:0{i}.0", "src_end": f"0:0{i}.9"})
    json.dump(manifest, open(os.path.join(strips, "manifest_r1.json"), "w"))
    cfg = {"name": "test", "workdir": work, "sport": "Muay Thai", "video": {"source": "x.mp4"},
           "camera": "single fixed camera, elevated, ringside", "rounds": [[0, 10]],
           "fighters": [{"id": "RED", "description": "red shorts", "torso": []},
                        {"id": "BLUE", "description": "blue shorts", "torso": []}],
           "ignore": "Ignore the referee."}
    path = os.path.join(tmp, "fight.json")
    json.dump(cfg, open(path, "w"))
    return path, work


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"path": self.path, "headers": dict(self.headers), "body": body})
        text = json.dumps({"items": ITEMS})
        if self.path.startswith("/v1/messages"):
            resp = {"id": "msg_test", "type": "message", "role": "assistant", "model": body["model"],
                    "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                    "stop_sequence": None, "usage": {"input_tokens": 1000, "output_tokens": 50}}
        else:
            resp = {"id": "cc_test", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "```json\n" + text + "\n```"}}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}}
        out = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def run(args, env_extra):
    env = {**os.environ, **env_extra}
    return subprocess.run([sys.executable, "label_strips.py", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)


def main():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    tmp = tempfile.mkdtemp()
    cfg_path, work = make_fixture(tmp)
    runs = os.path.join(work, "labels", "api")

    # --- anthropic backend (official SDK)
    r = run(["--config", cfg_path, "--round", "1", "--windows", "0:3", "--run", "test-anthropic"],
            {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "test-key"})
    assert r.returncode == 0, r.stderr
    req = REQUESTS[-1]
    assert req["path"].startswith("/v1/messages"), req["path"]
    b = req["body"]
    assert b["model"] == "claude-opus-5-5"
    assert b["output_config"]["format"]["type"] == "json_schema"
    assert b["output_config"]["effort"] == "high"
    assert b["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert "RED: red shorts" in b["system"] and "single fixed camera" in b["system"]
    enum = b["output_config"]["format"]["schema"]["properties"]["items"]["items"]["properties"]["attacker"]["enum"]
    assert enum == ["RED", "BLUE", "NONE"], enum
    imgs = [c for c in b["messages"][0]["content"] if c["type"] == "image"]
    assert len(imgs) == 3 and imgs[0]["source"]["media_type"] == "image/jpeg"
    out = os.path.join(runs, "test-anthropic")
    part = json.load(open(os.path.join(out, "r1_part_api.json")))
    assert [p["window"] for p in part] == ["r1_w001"], part  # hallucinated window dropped
    assert part[0]["tile"] == "05"
    n_before = len(REQUESTS)
    r = run(["--config", cfg_path, "--round", "1", "--windows", "0:3", "--run", "test-anthropic"],
            {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": "test-key"})
    assert r.returncode == 0 and len(REQUESTS) == n_before, "resume should skip done windows"
    print("anthropic backend: ok")

    # --- OpenAI-compatible backend (openai SDK, any base URL)
    r = run(["--config", cfg_path, "--round", "1", "--windows", "0:3", "--run", "test-compatible",
             "--provider", "compatible",
             "--base-url", url + "/v1", "--api-key-env", "FAKE_KEY", "--model", "some-vlm"],
            {"FAKE_KEY": "x"})
    assert r.returncode == 0, r.stderr
    req = REQUESTS[-1]
    assert req["path"] == "/v1/chat/completions", req["path"]
    b = req["body"]
    assert b["model"] == "some-vlm"
    assert b["response_format"]["type"] == "json_schema" and b["response_format"]["json_schema"]["strict"]
    imgs = [c for c in b["messages"][1]["content"] if c["type"] == "image_url"]
    assert len(imgs) == 3 and imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    part = json.load(open(os.path.join(runs, "test-compatible", "r1_part_api.json")))
    assert [p["window"] for p in part] == ["r1_w001"]  # parsed through the ```json fence
    print("openai-compatible backend: ok")

    shutil.rmtree(tmp, ignore_errors=True)
    srv.shutdown()
    print("all good")


if __name__ == "__main__":
    main()
