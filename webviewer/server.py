#!/usr/bin/env python3
"""AFM 3D Search — browser-based interactive text-search viewer.

Serves a three.js point-cloud viewer with a live query box and a scene
switcher. Text queries are encoded on CPU (OpenAI CLIP or SigLIP2, chosen
per scene via encoder.json) and matched against the featurized point
cloud — no GPU required.

Usage (from repo root, inside .rerun_env):
    python webviewer/server.py data/completed/<scene> [--port 8090]

All sibling directories of <scene> that contain scene data appear in the
dashboard's scene dropdown. Accepts pipeline output (point_cloud.ply +
clip_features.npy [+ dino_features.npy]) or legacy featurized .pt files.
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
def has_scene_data(path: Path) -> bool:
    return path.is_dir() and ((path / "point_cloud.ply").exists() or any(path.glob("*.pt")))


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
        raise FileNotFoundError(f"No point_cloud.ply or *.pt found in {scene_dir}")

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
        "raw_norms": norms[:, 0],
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
# Text encoder (OpenAI CLIP or SigLIP/SigLIP2, chosen per scene)
# --------------------------------------------------------------------------- #
class TextEncoder:
    TEMPLATES = ["a photo of a {}", "a {} in a room", "{}"]

    def __init__(self, feature_dim: int, model_name: str = None):
        import torch  # noqa: PLC0415

        self._torch = torch
        if model_name is None:
            model_name = {512: "ViT-B/32", 768: "ViT-L/14",
                          1152: "google/siglip2-so400m-patch14-384"}.get(feature_dim, "ViT-B/32")
        self.version = model_name
        self.is_siglip = "siglip" in model_name.lower()
        # contrastive-softmax temperature; SigLIP cosine gaps are tighter than
        # CLIP's, but too steep saturates scores into massive ties at 0/0.5/1
        self.temperature = 20.0 if self.is_siglip else 10.0

        print(f"Loading text encoder {model_name} (cpu)...")
        if self.is_siglip:
            from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
            self.model = AutoModel.from_pretrained(model_name).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            import clip  # noqa: PLC0415
            self._clip = clip
            self.model, _ = clip.load(model_name, device="cpu")

    def encode(self, text: str) -> np.ndarray:
        """Prompt-ensembled embedding: average of normalized template embeddings."""
        prompts = [t.format(text) for t in self.TEMPLATES]
        with self._torch.no_grad():
            if self.is_siglip:
                inputs = self.tokenizer(prompts, padding="max_length", max_length=64,
                                        truncation=True, return_tensors="pt")
                out = self.model.get_text_features(**inputs)
                if not self._torch.is_tensor(out):  # transformers>=5 returns an output object
                    out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
                feats = out.float().numpy()
            else:
                tokens = self._clip.tokenize(prompts, truncate=True)
                feats = self.model.encode_text(tokens).float().numpy()
        feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
        feat = feats.mean(axis=0)
        return feat / max(np.linalg.norm(feat), 1e-8)


# --------------------------------------------------------------------------- #
# Search index: contrastive negatives + DINO-weighted kNN smoothing
# --------------------------------------------------------------------------- #
NEGATIVE_PROMPTS = ["object", "things", "stuff", "texture"]


class SearchIndex:
    """Precomputed structures that sharpen raw similarity when needed:

    - canonical negative embeddings for LERF-style pairwise-softmax relevancy
    - a 3D kNN graph with DINO-similarity edge weights (disk-cached per scene)
    - a validity mask excluding points whose features were never observed
    """

    def __init__(self, scene, encoder, cache_path: Path = None):
        self.negatives = np.stack([encoder.encode(p) for p in NEGATIVE_PROMPTS])
        self.temperature = encoder.temperature
        # points SAM never covered have (near-)zero features — unrankable noise
        self.valid = scene["raw_norms"] > 0.05
        n_invalid = int((~self.valid).sum())
        if n_invalid:
            print(f"Excluding {n_invalid} zero-feature points from ranking")

        self.nbr_idx = self.nbr_w = self.nbr_w_sum = None
        if scene["dino"] is None:
            return
        n = len(scene["points"])

        if cache_path and cache_path.exists():
            try:
                z = np.load(cache_path)
                if len(z["idx"]) == n:
                    self.nbr_idx = z["idx"]
                    self.nbr_w = z["w"].astype(np.float32)
                    self.nbr_w_sum = self.nbr_w.sum(axis=1)
                    print("kNN graph loaded from cache")
                    return
            except Exception as e:
                print(f"cache read failed ({e}), rebuilding")

        try:
            from scipy.spatial import cKDTree  # noqa: PLC0415
        except ImportError:
            print("scipy not available — DINO smoothing disabled")
            return

        print("Building 3D kNN graph...")
        points = scene["points"]
        tree = cKDTree(points)
        _, idx = tree.query(points, k=9, workers=-1)
        idx = idx[:, 1:].astype(np.int32)  # drop self-neighbor

        print("Computing DINO edge weights...")
        dino = scene["dino"].astype(np.float32)
        dino /= np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1e-8)
        w = np.empty(idx.shape, dtype=np.float32)
        chunk = 100_000
        for s in range(0, len(dino), chunk):
            e = min(s + chunk, len(dino))
            w[s:e] = np.clip(np.einsum("nd,nkd->nk", dino[s:e], dino[idx[s:e]]), 0.0, 1.0)
        del dino
        self.nbr_idx, self.nbr_w, self.nbr_w_sum = idx, w, w.sum(axis=1)
        if cache_path:
            np.savez(cache_path, idx=idx, w=w.astype(np.float16))
            print(f"kNN graph cached to {cache_path.name}")
        print(f"Search index ready ({n} points, k=8)")

    def relevancy(self, sims_query, clip_feats):
        """Pairwise softmax vs each negative prompt, take the minimum (LERF)."""
        result = None
        for neg in self.negatives:
            sims_neg = clip_feats @ neg
            p = 1.0 / (1.0 + np.exp(np.clip((sims_neg - sims_query) * self.temperature, -50, 50)))
            result = p if result is None else np.minimum(result, p)
        return result

    def smooth(self, scores):
        if self.nbr_idx is None:
            return scores
        neighbor_avg = (self.nbr_w * scores[self.nbr_idx]).sum(axis=1)
        return (scores + neighbor_avg) / (1.0 + self.nbr_w_sum)

    def coherence_mask(self, matched):
        """Keep matches with at least 2 matching neighbors (kills stray points)."""
        if self.nbr_idx is None:
            return matched
        support = matched[self.nbr_idx].sum(axis=1)
        return matched & (support >= 2)


# --------------------------------------------------------------------------- #
# Scene manager: hot-swap between scene directories
# --------------------------------------------------------------------------- #
class SceneManager:
    def __init__(self, root: Path, initial: str, max_points: int):
        self.root = root
        self.max_points = max_points
        self.lock = threading.Lock()
        self._encoders = {}
        self.scene = self.index = self.encoder = None
        self.name = None
        self.suggestions = []
        self.load(initial)

    def list_scenes(self):
        return sorted(d.name for d in self.root.iterdir() if has_scene_data(d))

    def load(self, name: str):
        path = self.root / name
        if not has_scene_data(path):
            raise FileNotFoundError(f"no scene data in {path}")
        print(f"\n=== Loading scene '{name}' ===")
        scene = load_scene(path, self.max_points)

        model_name = None
        encoder_info = path / "encoder.json"
        if encoder_info.exists():
            model_name = json.loads(encoder_info.read_text()).get("clip_model")
        dim = scene["clip"].shape[1]
        key = model_name or {512: "ViT-B/32", 768: "ViT-L/14",
                             1152: "google/siglip2-so400m-patch14-384"}.get(dim, "ViT-B/32")
        if key not in self._encoders:
            self._encoders[key] = TextEncoder(dim, key)
        encoder = self._encoders[key]

        index = SearchIndex(scene, encoder, cache_path=path / "search_cache.npz")
        suggestions = []
        sf = path / "suggestions.json"
        if sf.exists():
            suggestions = json.loads(sf.read_text())

        with self.lock:
            self.scene, self.index, self.encoder = scene, index, encoder
            self.name, self.suggestions = name, suggestions
        print(f"=== Scene '{name}' ready: {len(scene['points']):,} points, "
              f"{encoder.version}, DINO {'yes' if scene['dino'] is not None else 'no'} ===\n")

    def current(self):
        with self.lock:
            return self.scene, self.index, self.encoder

    def meta(self):
        with self.lock:
            return {
                "scene": self.name,
                "num_points": len(self.scene["points"]),
                "clip_model": self.encoder.version,
                "has_dino": self.scene["dino"] is not None,
                "suggestions": self.suggestions,
                "scenes": self.list_scenes(),
            }


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    mgr: SceneManager = None
    switch_lock = threading.Lock()

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
            self._send(200, json.dumps(self.mgr.meta()).encode())
        elif path == "/api/pointcloud":
            scene, _, _ = self.mgr.current()
            buf = io.BytesIO()
            buf.write(struct.pack("<I", len(scene["points"])))
            buf.write(scene["points"].tobytes())
            buf.write(scene["rgb"].tobytes())
            self._send(200, buf.getvalue(), "application/octet-stream")
        else:
            self._send(404, b"{}")

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")

        if path == "/api/scene":
            name = req.get("name", "")
            if name not in self.mgr.list_scenes():
                self._send(404, json.dumps({"error": f"unknown scene {name!r}"}).encode())
                return
            with self.switch_lock:
                if name != self.mgr.name:
                    try:
                        self.mgr.load(name)
                    except Exception as e:
                        self._send(500, json.dumps({"error": str(e)}).encode())
                        return
            self._send(200, json.dumps(self.mgr.meta()).encode())
            return

        if path != "/api/search":
            self._send(404, b"{}")
            return

        query = (req.get("query") or "").strip()
        percentile = float(req.get("percentile", 99.0))
        # both default off: with sharp SigLIP features + mask-exact crops the raw
        # similarities are already well separated; these help mushy features only
        contrastive = bool(req.get("contrastive", False))
        smoothing = bool(req.get("smooth", False))
        if not query:
            self._send(400, b'{"error": "empty query"}')
            return

        scene, index, encoder = self.mgr.current()
        text_feat = encoder.encode(query)
        sims = scene["clip"] @ text_feat

        scores = index.relevancy(sims, scene["clip"]) if contrastive else sims
        if smoothing:
            scores = index.smooth(scores)

        # break saturated-sigmoid ties with the raw similarity so top-k stays exact
        sims_span = float(sims.max() - sims.min()) or 1.0
        scores = scores + 0.002 * (sims - sims.min()) / sims_span
        scores[~index.valid] = float(scores.min()) - 1.0

        # exact top-k selection — percentile on a tied plateau overshoots badly
        k = max(1, int(round(len(scores) * (100.0 - percentile) / 100.0)))
        threshold = float(np.partition(scores, len(scores) - k)[len(scores) - k])
        matched = scores >= threshold
        kept = index.coherence_mask(matched)
        scores = scores.astype(np.float32)
        scores[matched & ~kept] = threshold - 1e-4  # drop stray hits below threshold
        matches = int(kept.sum())

        headers = {
            "X-Threshold": f"{threshold:.6f}",
            "X-Matches": str(matches),
            "X-Sim-Min": f"{float(scores.min()):.6f}",
            "X-Sim-Max": f"{float(scores.max()):.6f}",
        }
        print(f"  query='{query}' p{percentile:g} contrastive={contrastive} smooth={smoothing} "
              f"thr={threshold:.4f} matches={matches}")
        self._send(200, scores.tobytes(), "application/octet-stream", headers)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_dir", type=Path,
                    help="a scene directory; its siblings become switchable scenes")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-points", type=int, default=1_500_000)
    args = ap.parse_args()

    if has_scene_data(args.scene_dir):
        root, initial = args.scene_dir.parent, args.scene_dir.name
    else:
        root = args.scene_dir
        candidates = sorted(d.name for d in root.iterdir() if has_scene_data(d))
        if not candidates:
            raise SystemExit(f"no scenes found under {root}")
        initial = candidates[0]

    Handler.mgr = SceneManager(root, initial, args.max_points)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  AFM 3D Search viewer -> http://localhost:{args.port}\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
