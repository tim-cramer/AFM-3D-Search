#!/usr/bin/env python3
"""AFM 3D Search — browser-based interactive text-search viewer.

Serves a three.js point-cloud viewer with a live query box. Text queries are
CLIP-encoded on CPU and matched against the featurized point cloud, so the
whole thing runs without a GPU.

Usage (from repo root, inside .rerun_env):
    python webviewer/server.py data/completed/<scene> [--port 8090]

Accepts either pipeline output (point_cloud.ply + clip_features.npy
[+ dino_features.npy]) or legacy featurized .pt files.
"""
import argparse
import io
import json
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

STATIC_DIR = Path(__file__).parent / "static"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_scene(scene_dir: Path, max_points: int):
    """Load points, colors, and features from a scene directory."""
    pt_files = sorted(scene_dir.glob("*.pt"))
    if (scene_dir / "point_cloud.ply").exists():
        points, rgb = _load_ply(scene_dir / "point_cloud.ply")
        clip_feats = np.load(scene_dir / "clip_features.npy")
        dino_path = scene_dir / "dino_features.npy"
        dino_feats = np.load(dino_path) if dino_path.exists() else None
    elif pt_files:
        import torch  # noqa: PLC0415 — only needed for the legacy format

        preferred = [p for p in pt_files if "clip" in p.name.lower()] or pt_files
        data = torch.load(preferred[0], map_location="cpu", weights_only=False)
        to_np = lambda v: v.numpy() if hasattr(v, "numpy") else np.asarray(v)
        points = to_np(data["points"])
        rgb = to_np(data["rgb"])
        clip_feats = to_np(data["features_clip"])
        dino_feats = to_np(data["features_dino"]) if data.get("features_dino") is not None else None
    else:
        raise FileNotFoundError(
            f"No point_cloud.ply or *.pt found in {scene_dir}"
        )

    if rgb.max() > 1.0:
        rgb = rgb / 255.0

    n = len(points)
    if n != len(clip_feats):
        raise ValueError(f"points ({n}) and clip features ({len(clip_feats)}) misaligned")

    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        idx.sort()
        points, rgb, clip_feats = points[idx], rgb[idx], clip_feats[idx]
        if dino_feats is not None:
            dino_feats = dino_feats[idx]
        print(f"Subsampled {n} -> {max_points} points")

    clip_feats = clip_feats.astype(np.float32)
    norms = np.linalg.norm(clip_feats, axis=1, keepdims=True)
    clip_feats /= np.maximum(norms, 1e-8)

    return {
        "points": points.astype(np.float32),
        "rgb": (rgb * 255).clip(0, 255).astype(np.uint8),
        "clip": clip_feats,
        "dino": dino_feats,
    }


def _load_ply(path: Path):
    from plyfile import PlyData  # noqa: PLC0415

    ply = PlyData.read(str(path))
    v = ply["vertex"]
    points = np.stack([v["x"], v["y"], v["z"]], axis=1)
    names = v.data.dtype.names
    if "red" in names:
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
    else:
        rgb = np.full_like(points, 180.0)
    return points, rgb


# --------------------------------------------------------------------------- #
# CLIP text encoder (auto-detects model from feature dim, CPU is fine)
# --------------------------------------------------------------------------- #
class TextEncoder:
    def __init__(self, feature_dim: int):
        import clip  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self._clip, self._torch = clip, torch
        version = {512: "ViT-B/32", 768: "ViT-L/14"}.get(feature_dim, "ViT-B/32")
        print(f"Loading CLIP {version} for {feature_dim}-d features (cpu)...")
        self.model, _ = clip.load(version, device="cpu")
        self.version = version

    def encode(self, text: str) -> np.ndarray:
        tokens = self._clip.tokenize([text], truncate=True)
        with self._torch.no_grad():
            feat = self.model.encode_text(tokens).float().numpy()[0]
        return feat / max(np.linalg.norm(feat), 1e-8)


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    scene = None
    encoder = None
    encoder_lock = threading.Lock()
    meta = {}

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("  " + fmt % args + "\n")

    def _send(self, code, body: bytes, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), MIME[".html"])
        elif path.startswith("/static/"):
            f = (STATIC_DIR / path[len("/static/"):]).resolve()
            if STATIC_DIR.resolve() in f.parents and f.is_file():
                self._send(200, f.read_bytes(), MIME.get(f.suffix, "application/octet-stream"))
            else:
                self._send(404, b"{}")
        elif path == "/api/meta":
            self._send(200, json.dumps(self.meta).encode())
        elif path == "/api/pointcloud":
            s = self.scene
            buf = io.BytesIO()
            buf.write(struct.pack("<I", len(s["points"])))
            buf.write(s["points"].tobytes())
            buf.write(s["rgb"].tobytes())
            self._send(200, buf.getvalue(), "application/octet-stream")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        if self.path.split("?")[0] != "/api/search":
            self._send(404, b"{}")
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        query = (req.get("query") or "").strip()
        percentile = float(req.get("percentile", 99.0))
        if not query:
            self._send(400, b'{"error": "empty query"}')
            return

        with self.encoder_lock:
            text_feat = self.encoder.encode(query)
        sims = self.scene["clip"] @ text_feat

        threshold = float(np.percentile(sims, percentile))
        matches = int((sims >= threshold).sum())
        headers = {
            "X-Threshold": f"{threshold:.6f}",
            "X-Matches": str(matches),
            "X-Sim-Min": f"{float(sims.min()):.6f}",
            "X-Sim-Max": f"{float(sims.max()):.6f}",
        }
        print(f"  query='{query}' p{percentile:g} thr={threshold:.4f} matches={matches}")
        self._send(200, sims.astype(np.float32).tobytes(), "application/octet-stream", headers)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_dir", type=Path)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    args = ap.parse_args()

    scene = load_scene(args.scene_dir, args.max_points)
    print(f"Loaded scene '{args.scene_dir.name}': {len(scene['points']):,} points, "
          f"CLIP dim {scene['clip'].shape[1]}, DINO {'yes' if scene['dino'] is not None else 'no'}")

    encoder = TextEncoder(scene["clip"].shape[1])

    suggestions = []
    sf = args.scene_dir / "suggestions.json"
    if sf.exists():
        suggestions = json.loads(sf.read_text())

    Handler.scene = scene
    Handler.encoder = encoder
    Handler.meta = {
        "scene": args.scene_dir.name,
        "num_points": len(scene["points"]),
        "clip_model": encoder.version,
        "has_dino": scene["dino"] is not None,
        "suggestions": suggestions,
    }

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  AFM 3D Search viewer -> http://localhost:{args.port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
