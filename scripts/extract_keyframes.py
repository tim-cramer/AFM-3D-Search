#!/usr/bin/env python3
"""Select sharp, novel keyframes from a video for the reconstruction pipeline.

SLAM-style keyframing without the SLAM: sample the video at a base rate,
drop motion-blurred frames (variance of Laplacian), and keep a frame only
when it sees enough new content vs. the last keyframe (ORB feature overlap),
with a forced keyframe after --force-gap seconds of visual standstill.
A hard cap bounds GPU memory for the reconstruction forward pass.

Usage:
    python scripts/extract_keyframes.py video.mp4 data/testing/<scene>/images
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--max-frames", type=int, default=80,
                    help="hard cap (GPU memory bound; ~150 fits half an A100)")
    ap.add_argument("--base-fps", type=float, default=4.0, help="candidate sampling rate")
    ap.add_argument("--overlap", type=float, default=0.55,
                    help="ORB match ratio below which a frame counts as novel")
    ap.add_argument("--min-sharpness", type=float, default=40.0,
                    help="variance-of-Laplacian threshold; blurrier frames are dropped")
    ap.add_argument("--force-gap", type=float, default=3.0,
                    help="max seconds between keyframes even without novelty")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / args.base_fps))

    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kept, last_desc, last_kept_t = [], None, -1e9
    n_candidates = n_blurry = 0
    idx = 0
    while cap.grab():
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / fps
        n_candidates += 1

        width = 640
        small = cv2.resize(frame, (width, max(1, round(frame.shape[0] * width / frame.shape[1]))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < args.min_sharpness:
            n_blurry += 1
            idx += 1
            continue

        _, desc = orb.detectAndCompute(gray, None)
        novel = True
        if last_desc is not None and desc is not None and len(desc) and len(last_desc):
            matches = matcher.match(last_desc, desc)
            good = [m for m in matches if m.distance < 50]
            overlap = len(good) / max(min(len(last_desc), len(desc)), 1)
            novel = overlap < args.overlap
        if novel or (t - last_kept_t) >= args.force_gap:
            kept.append(frame)
            last_desc, last_kept_t = desc, t
        idx += 1
    cap.release()

    if len(kept) > args.max_frames:
        sel = np.linspace(0, len(kept) - 1, args.max_frames).round().astype(int)
        kept = [kept[i] for i in sel]
        print(f"thinned to hard cap of {args.max_frames}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, frame in enumerate(kept):
        cv2.imwrite(str(args.out_dir / f"frame_{n:04d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = (kept[0].shape[:2]) if kept else (0, 0)
    print(f"{n_candidates} candidates ({n_blurry} blurry dropped) -> "
          f"{len(kept)} keyframes ({w}x{h}) in {args.out_dir}")


if __name__ == "__main__":
    main()
