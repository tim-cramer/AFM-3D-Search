# Web Viewer

Browser-based interactive text search on featurized point clouds. A small
dependency-free Python server (stdlib HTTP) loads the pipeline output,
CLIP-encodes queries on CPU, and streams per-point similarities to a
three.js front-end (vendored — works fully offline).

```bash
# one-time env (any Python >= 3.10)
uv venv .viewer_env --python 3.11
source .viewer_env/bin/activate
uv pip install -r webviewer/requirements.txt

# serve a processed scene
python webviewer/server.py data/completed/<scene_id> --port 8090
# open http://localhost:8090
```

Accepts pipeline output (`point_cloud.ply` + `clip_features.npy`
[+ `dino_features.npy`]) or legacy featurized `.pt` files. The CLIP text
model (ViT-B/32 or ViT-L/14) is auto-detected from the feature dimension.

No processed scene yet? Generate a synthetic test room with real CLIP text
embeddings:

```bash
python webviewer/make_synthetic_scene.py
python webviewer/server.py data/completed/synthetic_room
```
