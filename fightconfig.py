"""Per-fight configuration: everything that is specific to one video and one bout.

A fight config is a JSON file (see configs/example.fight.json):

{
  "name": "my-bout",
  "workdir": "private/my-bout",          # where video, tracks, labels, reviews, renders go
  "sport": "Muay Thai",
  "video": {
    "source": "path/to/video.mp4" | "https://...",   # local file or any yt-dlp URL
    "start": "1:02:10", "end": "1:11:30",            # optional trim of the source
    "timecode_offset": null                          # source time of the prepared clip's t=0
  },                                                 #   (defaults to "start", else 0)
  "camera": "single fixed camera, elevated ...",     # framing, used in the labelling prompt
  "rounds": [[4.0, 127.0], ...],                     # seconds in the prepared clip
  "fighters": [                                      # exactly two
    {"id": "RED", "description": "red shorts, ...",
     "draw_bgr": [70, 90, 255],                      # overlay colour
     "torso": [{"h": [0, 8], "s": [110, 255], "v": [60, 255]}],   # OpenCV HSV ranges (H 0-179)
     "legs":  [...],                                 # optional, e.g. shin guards
     "weights": {"torso": 1.0, "legs": 1.5, "other_torso": 1.0}}, # optional identity scoring
    ...
  ],
  "ignore": "The referee wears all black ...",       # people the labeller should ignore
  "official_result": "RED by decision"               # optional, shown on the final card
}

Resolution order for the config path: --config, $FIGHT_CONFIG, private/fight.json.
"""
import hashlib
import json
import os
import sys

DEFAULT_PATHS = ("private/fight.json",)


def parse_time(t):
    """'1:02:10' / '62:10' / 3730 / '3730.5' -> seconds (float)."""
    if t is None or t == "":
        return None
    if isinstance(t, (int, float)):
        return float(t)
    parts = [float(p) for p in str(t).split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def fmt_time(sec, decimals=0):
    sec = max(0.0, float(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    ss = f"{s:0{3 + decimals if decimals else 2}.{decimals}f}"
    return f"{int(h)}:{int(m):02d}:{ss}" if h >= 1 else f"{int(m)}:{ss}"


class FightConfig:
    def __init__(self, data, path):
        self.data, self.path_ = data, path
        self.name = data.get("name", "fight")
        self.workdir = data.get("workdir", "private")
        self.sport = data.get("sport", "Muay Thai")
        v = data.get("video", {})
        self.video_source = v.get("source")
        self.video_start = parse_time(v.get("start"))
        self.video_end = parse_time(v.get("end"))
        off = parse_time(v.get("timecode_offset"))
        self.time_offset = off if off is not None else (self.video_start or 0.0)
        self.camera = data.get("camera", "single fixed camera")
        self.rounds = [tuple(map(float, r)) for r in data.get("rounds", [])]
        self.fighters = data["fighters"]
        if len(self.fighters) != 2:
            sys.exit("a fight config needs exactly two fighters")
        self.names = tuple(f["id"] for f in self.fighters)
        self.colours = {f["id"]: tuple(f.get("draw_bgr", (255, 255, 255))) for f in self.fighters}
        self.ignore = data.get("ignore", "")
        self.official_result = data.get("official_result")

    def path(self, *parts):
        return os.path.join(self.workdir, *parts)

    @property
    def video(self):
        """The prepared clip every stage works on."""
        return self.path("data", "fight.mp4")

    def appearance_key(self):
        """Hash of the colour signatures, so cached colour features are rebuilt when they change."""
        sig = json.dumps([{k: f.get(k) for k in ("id", "torso", "legs")} for f in self.fighters],
                         sort_keys=True)
        return hashlib.sha1(sig.encode()).hexdigest()[:8]

    def fighters_text(self):
        lines = [f"- {f['id']}: {f.get('description', '')}" for f in self.fighters]
        if self.ignore:
            lines.append(f"- {self.ignore}")
        return "\n".join(lines)

    def src_time(self, frame, fps=30.0, decimals=1):
        """Display time in the original source for a frame of the prepared clip."""
        return fmt_time(self.time_offset + frame / fps, decimals)


def load(path=None):
    path = path or os.environ.get("FIGHT_CONFIG")
    if not path:
        path = next((p for p in DEFAULT_PATHS if os.path.exists(p)), None)
    if not path or not os.path.exists(path):
        sys.exit("no fight config: pass --config, set FIGHT_CONFIG, or copy "
                 "configs/example.fight.json to private/fight.json")
    with open(path) as f:
        return FightConfig(json.load(f), path)


def add_arg(ap):
    ap.add_argument("--config", default=None,
                    help="fight config JSON (default: $FIGHT_CONFIG or private/fight.json)")
