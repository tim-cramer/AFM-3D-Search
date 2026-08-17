#!/usr/bin/env python3
"""Generate a synthetic featurized room scene for testing the web viewer.

Geometry is procedural, but the per-point CLIP features are real text
embeddings (label prompt + noise), so text search behaves like the real
pipeline. Output matches run_pipeline.py's format:

    data/completed/synthetic_room/point_cloud.ply
    data/completed/synthetic_room/clip_features.npy   (float16 to save disk)
    data/completed/synthetic_room/suggestions.json
    data/completed/synthetic_room/ground_truth.json   (for automated checks)

Usage (inside .rerun_env, from repo root):
    python webviewer/make_synthetic_scene.py
"""
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(42)
OUT = Path(__file__).parent.parent / "data" / "completed" / "synthetic_room"


def box_surface(n, center, size, jitter=0.004):
    """Sample n points on the surface of an axis-aligned box (y-up)."""
    sx, sy, sz = size
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy], float)
    face = rng.choice(6, n, p=areas / areas.sum())
    u, v = rng.uniform(-0.5, 0.5, (2, n))
    pts = np.zeros((n, 3))
    for f in range(6):
        m = face == f
        axis, sign = divmod(f, 2)
        w = [sx, sy, sz]
        pts[m, axis] = (0.5 if sign == 0 else -0.5) * w[axis]
        other = [a for a in range(3) if a != axis]
        pts[m, other[0]] = u[m] * w[other[0]]
        pts[m, other[1]] = v[m] * w[other[1]]
    return pts + np.asarray(center) + rng.normal(0, jitter, (n, 3))


def blob(n, center, radii, jitter=0.008):
    """Sample n points on an ellipsoid-ish blob."""
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 1 - np.abs(rng.normal(0, 0.12, n))  # cluster near the surface
    return d * r[:, None] * np.asarray(radii) + np.asarray(center) + rng.normal(0, jitter, (n, 3))


def plane(n, center, size_x, size_z, axis="y", jitter=0.003):
    pts = np.zeros((n, 3))
    a, b = rng.uniform(-0.5, 0.5, (2, n))
    if axis == "y":      # horizontal (floor)
        pts[:, 0], pts[:, 2] = a * size_x, b * size_z
    elif axis == "z":    # back wall
        pts[:, 0], pts[:, 1] = a * size_x, b * size_z
    else:                # side wall
        pts[:, 2], pts[:, 1] = a * size_x, b * size_z
    return pts + np.asarray(center) + rng.normal(0, jitter, (n, 3))


# label, prompt, base color, n_points, geometry  (room is 6 x 4 m, y-up)
OBJECTS = [
    ("floor",    "the wooden floor of a room", (168, 144, 118), 40000,
     lambda n: plane(n, (0, 0, 0), 6.0, 4.0, "y")),
    ("wall",     "a plain white wall",         (208, 205, 198), 36000,
     lambda n: np.vstack([plane(n // 2, (0, 1.25, -2.0), 6.0, 2.5, "z"),
                          plane(n - n // 2, (-3.0, 1.25, 0), 4.0, 2.5, "x")])),
    ("desk",     "a wooden desk",              (146, 100, 62), 16000,
     lambda n: np.vstack([box_surface(int(n*0.7), (-1.6, 0.74, -1.6), (1.5, 0.05, 0.7)),
                          box_surface(n - int(n*0.7), (-1.6, 0.37, -1.6), (1.4, 0.7, 0.6), 0.002)])),
    ("monitor",  "a computer monitor",         (28, 30, 38), 7000,
     lambda n: box_surface(n, (-1.6, 1.05, -1.85), (0.62, 0.38, 0.05))),
    ("chair",    "an office chair",            (52, 56, 66), 11000,
     lambda n: np.vstack([box_surface(n // 2, (-1.55, 0.45, -0.95), (0.45, 0.06, 0.45)),
                          box_surface(n - n // 2, (-1.55, 0.75, -0.72), (0.45, 0.55, 0.07))])),
    ("bed",      "a bed with a blue blanket",  (72, 108, 170), 26000,
     lambda n: box_surface(n, (1.8, 0.28, 1.0), (1.4, 0.55, 2.0))),
    ("backpack", "a red backpack",             (188, 44, 52), 9000,
     lambda n: blob(n, (0.4, 0.24, -1.5), (0.22, 0.26, 0.16))),
    ("plant",    "a green potted plant",       (58, 138, 62), 9000,
     lambda n: np.vstack([blob(int(n*0.75), (2.6, 0.85, -1.6), (0.28, 0.34, 0.28)),
                          box_surface(n - int(n*0.75), (2.6, 0.3, -1.6), (0.26, 0.3, 0.26))])),
]


def main():
    import clip
    import torch

    print("Loading CLIP ViT-B/32 (cpu, downloads ~340MB on first run)...")
    model, _ = clip.load("ViT-B/32", device="cpu")

    prompts = [o[1] for o in OBJECTS]
    with torch.no_grad():
        text = model.encode_text(clip.tokenize(prompts)).float().numpy()
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    dim = text.shape[1]

    all_pts, all_rgb, all_feat, ground_truth = [], [], [], {}
    cursor = 0
    for (label, _prompt, color, n, gen), emb in zip(OBJECTS, text):
        pts = gen(n)
        n = len(pts)
        rgb = np.asarray(color, float) + rng.normal(0, 10, (n, 3))
        feats = emb[None, :] + rng.normal(0, 0.016, (n, dim))
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        all_pts.append(pts)
        all_rgb.append(rgb.clip(0, 255))
        all_feat.append(feats.astype(np.float16))
        ground_truth[label] = [cursor, cursor + n]
        cursor += n
        print(f"  {label:9s} {n:6d} points")

    pts = np.vstack(all_pts).astype(np.float32)
    rgb = np.vstack(all_rgb).astype(np.uint8)
    feats = np.vstack(all_feat)

    OUT.mkdir(parents=True, exist_ok=True)
    from plyfile import PlyData, PlyElement

    vertex = np.zeros(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = pts.T
    vertex["red"], vertex["green"], vertex["blue"] = rgb.T
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(OUT / "point_cloud.ply"))
    np.save(OUT / "clip_features.npy", feats)
    (OUT / "suggestions.json").write_text(json.dumps(
        ["backpack", "plant", "bed", "monitor", "chair", "desk"]))
    (OUT / "ground_truth.json").write_text(json.dumps(ground_truth))

    mb = (OUT / "clip_features.npy").stat().st_size / 1e6
    print(f"\nWrote {len(pts):,} points to {OUT} (clip features {mb:.0f} MB)")


if __name__ == "__main__":
    main()
