"""Shared pieces of the review pages: clip cutting (real or placeholder) and the HTML.

Answers are structured, so apply_review.py can use them as-is:
  strike review: [{"id", "outcome", "attacker"?, "kind"?, "target"?, "add": [...], "note"?}, ...]
  clinch review: [{"id", "control", "note"?}, ...]
When the page is served by the pipeline UI (web/server.py) "Save" writes the answers file
straight into the fight's workdir; opened as a plain file it downloads answers.json instead.
"""
import html
import json
import os
import subprocess

import cv2
import numpy as np

STRIKES = ["punch", "kick", "teep", "knee", "elbow"]
TARGETS = ["head", "body", "leg"]
OUTCOMES = ["clean", "blocked", "missed", "not a strike", "can't tell"]


def hex_colour(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def crop_box(K, names, f0, f1, W0, H0, pad=40, lo=100, hi=180):
    """Square crop around both fighters over frames [f0, f1)."""
    pts = np.concatenate([K[n][f0:f1, :, :2].reshape(-1, 2) for n in names])
    pts = pts[~np.isnan(pts).any(1)]
    if len(pts) > 10:
        x0, y0 = np.percentile(pts, 2, axis=0)
        x1, y1 = np.percentile(pts, 98, axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = float(np.clip(max(x1 - x0, y1 - y0) / 2 + pad, lo, hi))
    else:
        cx, cy, half = W0 / 2, H0 / 2, min(W0, H0) / 2
    half = min(half, W0 / 2, H0 / 2)
    w = h = int(2 * half)
    return int(np.clip(cx - half, 0, W0 - w)), int(np.clip(cy - half, 0, H0 - h)), w, h


def write_clip(video, out_mp4, f0, f1, box, slow, fps=30.0, placeholder=None):
    """Cut [f0, f1) cropped to box at 1/slow speed, plus a poster jpg from the middle frame.
    placeholder: (K, colours) to draw mannequins instead of the real footage."""
    x, y, w, h = box
    poster = out_mp4.replace(".mp4", ".jpg")
    if placeholder is None:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{f0 / fps:.3f}", "-t", f"{(f1 - f0) / fps:.3f}",
                        "-i", video, "-an", "-vf",
                        f"crop={w}:{h}:{x}:{y},scale=540:540:flags=bicubic,setpts={slow}*PTS",
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out_mp4], check=True)
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, (f0 + f1) // 2)
        ok, img = cap.read()
        if ok:
            cv2.imwrite(poster, cv2.resize(img[y:y + h, x:x + w], (540, 540)))
        return
    import placeholder as ph

    K, colours = placeholder
    cap = cv2.VideoCapture(video)
    W0, H0 = int(cap.get(3)) or 640, int(cap.get(4)) or 360
    tmp = out_mp4.replace(".mp4", "_raw.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps / slow, (540, 540))
    for t in range(f0, f1):
        img = ph.frame(K, t, W0, H0, colours)[y:y + h, x:x + w]
        img = cv2.resize(img, (540, 540))
        vw.write(img)
        if t == (f0 + f1) // 2:
            cv2.imwrite(poster, img)
    vw.release()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", out_mp4], check=True)
    os.remove(tmp)


# ---------------------------------------------------------------- HTML
CSS = """
body { background:#111; color:#ddd; font:15px/1.4 -apple-system, system-ui, sans-serif; margin:24px; }
h1 { font-size:20px; margin:0 0 4px; }
p.sub { color:#999; margin:0 0 20px; max-width:820px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(520px, 1fr)); gap:18px; }
.card { background:#1c1c20; border-radius:8px; overflow:hidden; display:flex; flex-direction:column; }
video { width:100%; aspect-ratio:1; background:#000; object-fit:cover; }
.meta { padding:10px 14px 14px; }
.q { color:#888; font-size:13px; }
.claim { font-size:17px; margin:2px 0; }
.hint { color:#777; font-size:12px; margin-bottom:8px; }
.opts label { margin-right:12px; white-space:nowrap; cursor:pointer; }
.fix { margin-top:8px; font-size:13px; color:#aaa; }
.fix select, .add select { background:#121214; color:#ddd; border:1px solid #333; border-radius:4px; padding:2px 4px; margin-right:6px; }
.add { margin-top:6px; font-size:13px; }
.addbtn { background:none; color:#7fb8ff; border:0; cursor:pointer; padding:0; font-size:13px; margin-top:6px; }
textarea { width:100%; box-sizing:border-box; margin-top:10px; min-height:40px; background:#121214; color:#ddd;
  border:1px solid #333; border-radius:5px; padding:6px 8px; font:14px -apple-system, system-ui, sans-serif; resize:vertical; }
.bar { position:sticky; bottom:0; background:#111d; padding:14px 0; margin-top:20px; }
.bar button { padding:10px 18px; font-size:15px; background:#3cff8c; color:#111; border:0; border-radius:6px; cursor:pointer; margin-right:8px; }
.bar button.sec { background:#333; color:#ddd; }
#status { margin-left:6px; color:#aaa; }
"""

SAVE_JS = """
async function saveAnswers(data) {
  const status = document.getElementById('status');
  const body = JSON.stringify({path: SAVE_PATH, answers: data});
  if (location.protocol.startsWith('http')) {
    try {
      const r = await fetch('/api/answers', {method: 'POST', headers: {'Content-Type': 'application/json'}, body});
      if (r.ok) { status.textContent = 'Saved to ' + SAVE_PATH; return; }
      status.textContent = 'Save failed (' + r.status + '), downloading instead';
    } catch (e) { status.textContent = 'Server not reachable, downloading instead'; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], {type: 'application/json'}));
  a.download = SAVE_PATH.split('/').pop();
  a.click();
  status.textContent = 'Downloaded ' + a.download + ' - put it at <workdir>/' + SAVE_PATH;
}
function copyAnswers(data) {
  navigator.clipboard.writeText(JSON.stringify(data)).then(
    () => document.getElementById('status').textContent = 'Copied ' + data.length + ' answers');
}
"""


def _select(cls, options, selected):
    return f'<select class="{cls}">' + "".join(
        f'<option value="{html.escape(o)}"{" selected" if o == selected else ""}>{html.escape(o)}</option>'
        for o in options) + "</select>"


def page(title, intro, cards, script, save_path):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>
<h1>{html.escape(title)}: {len(cards)} clips</h1>
<p class="sub">{intro}</p>
<div class="grid">{''.join(cards)}</div>
<h2 style="font-size:16px;margin-top:28px">Anything else?</h2>
<textarea id="general" style="max-width:900px;min-height:70px" placeholder="general comments"></textarea>
<div class="bar"><button onclick="saveAnswers(collect())">Save answers</button>
<button class="sec" onclick="copyAnswers(collect())">Copy</button><span id="status"></span></div>
<script>const SAVE_PATH = {json.dumps(save_path)};
{SAVE_JS}
{script}
</script></body></html>"""


def strike_page(items, names, colours, title, save_path):
    """items: dicts with id, clip, poster, round, src_time, by, kind, target."""
    cards = []
    for it in items:
        col = hex_colour(colours[it["by"]])
        opts = "".join(f'<label><input type="radio" name="q{it["id"]}" value="{o}"> {o}</label>' for o in OUTCOMES)
        fix = (f'<div class="fix">If the label is wrong: {_select("f-by", names, it["by"])}'
               f'{_select("f-kind", STRIKES, it["kind"])}{_select("f-target", TARGETS, it["target"])}</div>')
        cards.append(f"""
<div class="card" data-id="{it['id']}">
  <video src="{it['clip']}" poster="{it['poster']}" autoplay loop muted playsinline controls></video>
  <div class="meta">
    <div class="q">#{it['id']} &middot; R{it['round']} &middot; {it['src_time']}</div>
    <div class="claim"><b style="color:{col}">{html.escape(it['by'])}</b> {it['kind']} &rarr; {it['target']}</div>
    <div class="hint">the strike lands mid-clip; judge whether it landed cleanly</div>
    <div class="opts">{opts}</div>
    {fix}
    <div class="adds"></div>
    <button class="addbtn" onclick="addRow(this)">+ add a strike seen in this clip</button>
    <textarea placeholder="optional notes"></textarea>
  </div>
</div>""")
    script = f"""
const NAMES = {json.dumps(list(names))}, STRIKES = {json.dumps(STRIKES)}, TARGETS = {json.dumps(TARGETS)};
const sel = (cls, opts) => '<select class="' + cls + '">' + opts.map(o => '<option>' + o + '</option>').join('') + '</select>';
function addRow(btn) {{
  const d = document.createElement('div'); d.className = 'add';
  d.innerHTML = sel('a-when', ['just before', 'just after', '1s after']) + sel('a-by', NAMES) + sel('a-kind', STRIKES)
    + sel('a-target', TARGETS) + sel('a-outcome', ['clean', 'blocked', 'missed']);
  btn.previousElementSibling.appendChild(d);
}}
const DT = {{'just before': -0.5, 'just after': 0.5, '1s after': 1.0}};
function collect() {{
  const out = [...document.querySelectorAll('.card')].map(c => {{
    const id = +c.dataset.id, v = c.querySelector('input[type=radio]:checked');
    const a = {{id, outcome: v ? v.value : null}};
    const orig = ITEMS[id];
    const by = c.querySelector('.f-by').value, kind = c.querySelector('.f-kind').value, target = c.querySelector('.f-target').value;
    if (by !== orig.by) a.attacker = by;
    if (kind !== orig.kind) a.kind = kind;
    if (target !== orig.target) a.target = target;
    a.add = [...c.querySelectorAll('.add')].map(r => ({{dt: DT[r.querySelector('.a-when').value],
      by: r.querySelector('.a-by').value, kind: r.querySelector('.a-kind').value,
      target: r.querySelector('.a-target').value, outcome: r.querySelector('.a-outcome').value}}));
    const note = c.querySelector('textarea').value.trim(); if (note) a.note = note;
    return a;
  }});
  const g = document.getElementById('general').value.trim();
  if (g) out.push({{general: g}});
  return out;
}}
const ITEMS = {json.dumps({it['id']: {k: it[k] for k in ('by', 'kind', 'target')} for it in items})};
"""
    intro = ("Each clip is a strike the vision model flagged. Mark what actually happened. If the attacker, "
             "strike or target is wrong, fix it with the dropdowns; add any strike the labels missed. "
             "Save writes the answers file that apply_review.py reads.")
    return page(title, intro, cards, script, save_path)


def clinch_page(items, names, title, save_path):
    """items: dicts with id, clip, poster, round, start, end, seconds."""
    cards = []
    for it in items:
        opts = "".join(f'<label><input type="radio" name="q{it["id"]}" value="{html.escape(v)}"> {html.escape(v)}</label>'
                       for v in (*names, "even", "not really a clinch"))
        cards.append(f"""
<div class="card" data-id="{it['id']}">
  <video src="{it['clip']}" poster="{it['poster']}" autoplay loop muted playsinline controls></video>
  <div class="meta">
    <div class="q">#{it['id']} &middot; R{it['round']} &middot; {it['start']} &ndash; {it['end']} ({it['seconds']}s)</div>
    <div class="claim">Who controlled this clinch?</div>
    <div class="hint">control = hands/arms on the head or neck, turning or driving him, bending him over</div>
    <div class="opts">{opts}</div>
    <textarea placeholder="optional: control changed partway, knees landed..."></textarea>
  </div>
</div>""")
    script = """
function collect() {
  const out = [...document.querySelectorAll('.card')].map(c => {
    const v = c.querySelector('input[type=radio]:checked');
    const a = {id: +c.dataset.id, control: v ? v.value : null};
    const note = c.querySelector('textarea').value.trim(); if (note) a.note = note;
    return a;
  });
  const g = document.getElementById('general').value.trim();
  if (g) out.push({general: g});
  return out;
}
"""
    intro = ("The longest clinches the labels found, with half a second of padding. Mark who controlled each; "
             "use the notes if it changed partway. Save writes the answers file analyze.py reads.")
    return page(title, intro, cards, script, save_path)
