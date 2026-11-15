"""Stage 0: turn the fight config's video source into the clip every other stage reads.

    uv run python prepare_video.py                 # -> <workdir>/data/fight.mp4

The source can be a local file or any URL yt-dlp understands, optionally trimmed with
video.start / video.end. The clip is re-encoded to H.264 at a constant 30 fps (the
windowing and review timing assume 30 fps). The clip's t=0 is shown as video.start in
the overlay and review pages, so timestamps still match the original source.
"""
import argparse
import os
import shutil
import subprocess
import sys

import fightconfig


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fightconfig.add_arg(ap)
    ap.add_argument("--yt-dlp-format", default="bv*[height<=1080]+ba/b")
    ap.add_argument("--yt-dlp-args", default="", help="extra yt-dlp arguments, e.g. --cookies-from-browser chrome")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--force", action="store_true", help="overwrite an existing clip")
    args = ap.parse_args()
    cfg = fightconfig.load(args.config)
    src, out = cfg.video_source, cfg.video
    if not src:
        sys.exit("set video.source in the fight config")
    if os.path.exists(out) and not args.force:
        sys.exit(f"{out} exists (use --force to rebuild)")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    start, end = cfg.video_start, cfg.video_end
    trim_in = []

    if src.startswith(("http://", "https://")):
        if not shutil.which("yt-dlp"):
            sys.exit("yt-dlp is needed for URL sources (brew install yt-dlp / pipx install yt-dlp)")
        raw = os.path.join(os.path.dirname(out), "source_download.%(ext)s")
        cmd = ["yt-dlp", "-f", args.yt_dlp_format, "--merge-output-format", "mp4", "-o", raw]
        if start is not None or end is not None:
            cmd += ["--download-sections", f"*{start or 0}-{end if end is not None else 'inf'}",
                    "--force-keyframes-at-cuts"]
        cmd += args.yt_dlp_args.split() + [src]
        run(cmd)
        src = raw.replace("%(ext)s", "mp4")
    else:
        if not os.path.exists(src):
            sys.exit(f"video source not found: {src}")
        if start is not None:
            trim_in += ["-ss", str(start)]
        if end is not None:
            trim_in += ["-to", str(end)]

    run(["ffmpeg", "-v", "error", "-y", *trim_in, "-i", src, "-r", "30", "-c:v", "libx264", "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-c:a", "aac", out])
    if src.endswith("source_download.mp4"):
        os.remove(src)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", out],
                         capture_output=True, text=True).stdout.strip()
    print(f"wrote {out} ({float(dur or 0):.0f}s); timestamps shown from {fightconfig.fmt_time(cfg.time_offset)}")
    print("next: uv run python track.py, then set `rounds` in the config (seconds in this clip)")


if __name__ == "__main__":
    main()
