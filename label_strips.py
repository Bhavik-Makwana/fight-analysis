"""Label strike attempts in the labelling sheets with any vision model, via its API.

This automates what we first did by hand with Claude Code subagents: each request
sends a few consecutive 4x4 frame sheets (from make_strips.py) and gets back one
JSON object per strike attempt / event, in the format merge_labels.py expects.

Backends (pick with --provider):
  anthropic   Claude via the official `anthropic` SDK (ANTHROPIC_API_KEY or `ant auth login`)
  openai      OpenAI via the `openai` SDK (OPENAI_API_KEY)
  gemini      Google Gemini through its OpenAI-compatible endpoint (GEMINI_API_KEY)
  openrouter  any model on OpenRouter (OPENROUTER_API_KEY)
  ollama      a local model served by Ollama (no key)
  compatible  any other OpenAI-compatible server: pass --base-url and --api-key-env

Output is one JSONL line per request in
<workdir>/labels/api/<run>/r<N>.jsonl (so runs can be resumed), plus
<workdir>/labels/api/<run>/r<N>_part_api.json in the merge_labels.py format.
The prompt (sport, camera framing, fighter descriptions) comes from the fight config.

    uv run python label_strips.py --round 1 --provider anthropic --dry-run
    uv run python label_strips.py --round 1 --provider anthropic
    uv run python label_strips.py --round 1 --provider gemini --model <gemini-vision-model>
    uv run python merge_labels.py --round 1 --labels-dir <workdir>/labels/api/<run>
"""
import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fightconfig

SYSTEM = """You label strikes in a {sport} bout from video frame sheets. \
Your labels are used as ground truth to count strikes, so be careful and honest.

CAMERA
{camera}. Fighters are often partly hidden by each other, the ropes or the referee.

FIGHTERS
{fighters}

INPUT
Each image is one window: a 4x4 grid of 16 frames, read left-to-right then top-to-bottom. \
Each tile is labelled with its index (00-15) and the source timestamp. Frames are 0.1 s \
apart, so a window covers 1.6 s. Consecutive windows overlap by about 0.4 s (tiles 12-15 of \
one window ~ tiles 00-03 of the next). If a strike appears in two windows, report it in both: \
deduplication happens later. The footage may be low resolution and blurry.

WHAT TO LABEL
Every strike attempt: punch (jab, cross, hook, uppercut), kick (body/leg/head roundhouse), \
teep (push kick), knee, elbow. For each strike give the window id, the tile of impact (or full \
extension if it missed), attacker, strike, target (head/body/leg), outcome and confidence:
- clean: visibly lands on the target - glove/shin/knee meets head/torso/leg and/or a reaction \
(head snaps, body folds, weight shifts)
- blocked: meets a guard, arm, check or raised shin
- missed: falls short or is slipped
- unclear: thrown, but the result can't be seen
Also record, with strike = "event": dumps or sweeps (who put whom down), knockdowns or falls, \
clinch start/end and who controls it (in the note, e.g. "clinch start, {a} arms around {b}'s \
neck, {a} controlling"), and clear slips or ducks. For events use target/outcome "none"; \
attacker is the fighter doing it or in control, or "NONE".
For a window with nothing to label, add one item with strike = "none".

GUIDANCE
- Body shots matter. Look for hooks/straights to the body and roundhouse kicks to the ribs; \
they are easy to miss when the fighters are close.
- Kicks are often mistaken for punches or knees: check whether a leg or an arm extends.
- When a clear reaction is visible, commit to "clean"; use "unclear" only when the contact \
point really can't be seen.
- Don't invent strikes: guard adjustments, feints, steps and clinch arm-wraps are not strikes. \
Include possible strikes with low confidence rather than dropping them.
- In the clinch, knees count; an arm around the neck is not a punch."""


def system_prompt(cfg, fighters_override=None):
    a, b = cfg.names
    return SYSTEM.format(sport=cfg.sport, camera=cfg.camera.rstrip("."),
                         fighters=fighters_override or cfg.fighters_text(), a=a, b=b)


def make_schema(names):
    item = {
        "type": "object",
        "properties": {
            "window": {"type": "string", "description": "window id, e.g. r2_w014"},
            "tile": {"type": "integer", "description": "0-15"},
            "attacker": {"type": "string", "enum": [*names, "NONE"]},
            "strike": {"type": "string", "enum": ["punch", "kick", "teep", "knee", "elbow", "event", "none"]},
            "target": {"type": "string", "enum": ["head", "body", "leg", "none"]},
            "outcome": {"type": "string", "enum": ["clean", "blocked", "missed", "unclear", "none"]},
            "confidence": {"type": "number", "description": "0.0-1.0"},
            "note": {"type": "string"},
        },
        "required": ["window", "tile", "attacker", "strike", "target", "outcome", "confidence", "note"],
        "additionalProperties": False,
    }
    return {"type": "object", "properties": {"items": {"type": "array", "items": item}},
            "required": ["items"], "additionalProperties": False}


SCHEMA = None      # set in main() from the fight config
STRIP_DIR = None


def user_content_parts(windows):
    """(kind, payload) parts shared by all backends: a text header then image per window."""
    parts = [("text", f"Label these {len(windows)} consecutive windows. "
                      "Return JSON matching the schema: {\"items\": [...]}.")]
    for w in windows:
        with open(os.path.join(STRIP_DIR, w["file"]), "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode("utf-8")
        parts.append(("text", f"Window {w['window']} ({w['src_start']} - {w['src_end']}):"))
        parts.append(("image", b64))
    return parts


def parse_items(text):
    """Pull the items list out of a model's JSON reply (tolerates code fences)."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)
    items = data["items"] if isinstance(data, dict) else data
    return [it for it in items if isinstance(it, dict)]


# ---------------------------------------------------------------- backends
class AnthropicBackend:
    """Claude through the official anthropic SDK, with schema-constrained JSON output."""

    def __init__(self, model, effort, fallback):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=4)  # env key or `ant auth login` profile
        self.model, self.effort, self.fallback = model, effort, fallback

    def label(self, system, parts):
        content = []
        for kind, payload in parts:
            if kind == "text":
                content.append({"type": "text", "text": payload})
            else:
                content.append({"type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": payload}})
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        if self.fallback:
            # On a policy decline, the API re-runs the request on a fallback model.
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused: {getattr(resp.stop_details, 'category', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("hit max_tokens before finishing the JSON")
        text = next(b.text for b in resp.content if b.type == "text")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                 "model": resp.model}
        return parse_items(text), usage

    def retryable(self, exc):
        a = self.anthropic
        return isinstance(exc, (a.RateLimitError, a.APIConnectionError, a.InternalServerError))


PRESETS = {  # provider -> (base_url, api key env var)
    "openai": (None, "OPENAI_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


class OpenAICompatibleBackend:
    """Any server speaking the OpenAI chat-completions API with image inputs."""

    def __init__(self, model, base_url, key_env, json_mode):
        import openai

        self.openai = openai
        key = os.environ.get(key_env, "") if key_env else "not-needed"
        if key_env and not key:
            sys.exit(f"set {key_env} for this provider")
        self.client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=4)
        self.model, self.json_mode = model, json_mode

    def label(self, system, parts):
        content = []
        for kind, payload in parts:
            if kind == "text":
                content.append({"type": "text", "text": payload})
            else:
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
        if self.json_mode == "schema":
            fmt = {"type": "json_schema", "json_schema": {"name": "labels", "schema": SCHEMA, "strict": True}}
        elif self.json_mode == "object":
            fmt = {"type": "json_object"}
        else:
            fmt = None
        sys_text = system if fmt and fmt["type"] == "json_schema" else (
            system + "\n\nReply with only a JSON object of this shape:\n" + json.dumps(SCHEMA))
        kwargs = dict(model=self.model,
                      messages=[{"role": "system", "content": sys_text}, {"role": "user", "content": content}])
        if fmt:
            kwargs["response_format"] = fmt
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("hit the output limit before finishing the JSON")
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", None),
                 "output_tokens": getattr(resp.usage, "completion_tokens", None), "model": resp.model}
        return parse_items(choice.message.content or ""), usage

    def retryable(self, exc):
        o = self.openai
        return isinstance(exc, (o.RateLimitError, o.APIConnectionError, o.InternalServerError))


def make_backend(args):
    if args.provider == "anthropic":
        return AnthropicBackend(args.model or "claude-opus-5-5", args.effort, not args.no_fallback)
    if args.provider in PRESETS:
        base_url, key_env = PRESETS[args.provider]
    elif args.provider == "compatible":
        if not args.base_url:
            sys.exit("--provider compatible needs --base-url")
        base_url, key_env = args.base_url, args.api_key_env
    else:
        sys.exit(f"unknown provider {args.provider}")
    if not args.model:
        sys.exit(f"--model is required for --provider {args.provider}")
    return OpenAICompatibleBackend(args.model, args.base_url or base_url, args.api_key_env or key_env,
                                   args.json_mode)


# ---------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fightconfig.add_arg(ap)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "gemini", "openrouter", "ollama", "compatible"])
    ap.add_argument("--model", default=None, help="default for anthropic: claude-opus-5-5")
    ap.add_argument("--base-url", default=None, help="override / set the OpenAI-compatible endpoint")
    ap.add_argument("--api-key-env", default=None, help="env var holding the key (compatible provider)")
    ap.add_argument("--json-mode", default="schema", choices=["schema", "object", "prompt"],
                    help="OpenAI-compatible only: strict json_schema, json_object, or prompt-only")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"],
                    help="anthropic only")
    ap.add_argument("--no-fallback", action="store_true",
                    help="anthropic only: don't re-run declined requests on a fallback model")
    ap.add_argument("--batch", type=int, default=3, help="windows (images) per request")
    ap.add_argument("--workers", type=int, default=4, help="concurrent requests")
    ap.add_argument("--windows", default="", help="subset, e.g. 0:20")
    ap.add_argument("--fighters-file", default=None,
                    help="override the fighter descriptions from the config (plain text)")
    ap.add_argument("--run", default=None, help="run name (default: <provider>-<model>)")
    ap.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    args = ap.parse_args()
    global SCHEMA, STRIP_DIR
    cfg = fightconfig.load(args.config)
    SCHEMA = make_schema(cfg.names)
    STRIP_DIR = cfg.path("labels", "strips")

    manifest = json.load(open(os.path.join(STRIP_DIR, f"manifest_r{args.round}.json")))
    if args.windows:
        a, b = (int(x) if x else None for x in args.windows.split(":"))
        manifest = manifest[a:b]
    batches = [manifest[i:i + args.batch] for i in range(0, len(manifest), args.batch)]
    model_name = args.model or ("claude-opus-5-5" if args.provider == "anthropic" else "model")
    run = args.run or re.sub(r"[^A-Za-z0-9._-]+", "_", f"{args.provider}-{model_name}")
    out_dir = cfg.path("labels", "api", run)
    jsonl = os.path.join(out_dir, f"r{args.round}.jsonl")
    done = set()
    if os.path.exists(jsonl):
        for line in open(jsonl):
            done.update(json.loads(line)["windows"])
    todo = [b for b in batches if not all(w["window"] in done for w in b)]
    system = system_prompt(cfg, open(args.fighters_file).read() if args.fighters_file else None)

    print(f"round {args.round}: {len(manifest)} windows -> {len(batches)} requests of {args.batch} "
          f"({len(todo)} still to do) | {args.provider} / {model_name} | output {out_dir}")
    if args.dry_run:
        print("\n--- system prompt ---\n" + system)
        print("\n--- first request ---")
        for kind, payload in user_content_parts(todo[0] if todo else batches[0]):
            print(f"[{kind}] " + (payload if kind == "text" else f"<jpeg, {len(payload) * 3 // 4 // 1024} KB>"))
        return

    backend = make_backend(args)
    os.makedirs(out_dir, exist_ok=True)

    def run_batch(b):
        for attempt in range(4):
            try:
                items, usage = backend.label(system, user_content_parts(b))
                ids = {w["window"] for w in b}
                items = [it for it in items if it.get("window") in ids]  # drop hallucinated ids
                return {"windows": sorted(ids), "items": items, "usage": usage}
            except (json.JSONDecodeError, KeyError, RuntimeError) as e:  # bad/partial reply: ask again
                err = e
            except Exception as e:  # SDK errors: retry only transient ones (SDK already retried)
                if not backend.retryable(e):
                    raise
                err = e
            time.sleep(2 ** attempt)
        return {"windows": sorted(w["window"] for w in b), "error": str(err)}

    tot_in = tot_out = n_items = n_err = 0
    with open(jsonl, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_batch, b): b for b in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if "error" in rec:
                n_err += 1
                print(f"  [{k}/{len(todo)}] {rec['windows'][0]}..: FAILED {rec['error']}", file=sys.stderr)
                continue  # not written, so a re-run retries it
            out.write(json.dumps(rec) + "\n")
            out.flush()
            u = rec["usage"]
            tot_in += u.get("input_tokens") or 0
            tot_out += u.get("output_tokens") or 0
            n_items += len(rec["items"])
            print(f"  [{k}/{len(todo)}] {rec['windows'][0]}..{rec['windows'][-1]}: {len(rec['items'])} items")

    # rebuild the merge_labels.py input from everything labelled so far (all runs of this round)
    items = []
    for line in open(jsonl):
        rec = json.loads(line)
        for it in rec["items"]:
            it = dict(it)
            it["tile"] = f"{int(it.get('tile', 0)):02d}"
            if it.get("attacker") == "NONE":
                it["attacker"] = None
            items.append(it)
    part = os.path.join(out_dir, f"r{args.round}_part_api.json")
    json.dump(items, open(part, "w"), indent=1)
    print(f"\n{n_items} items this run, {n_err} failed requests (re-run to retry); "
          f"tokens in {tot_in:,} / out {tot_out:,}\nwrote {part}")
    print(f"next: uv run python merge_labels.py --round {args.round} --labels-dir {out_dir}")


if __name__ == "__main__":
    main()
