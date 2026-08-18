# Demo Walkthrough

End-to-end: images → featurized point cloud (GPU) → interactive search in the browser (no GPU).

## 1. Process a scene (GPU machine)

Set up once — see [README](README.md#setup-pipeline-gpu-machine) or run `scripts/setup_gpu_instance.sh`. Then:

```bash
# images live in data/testing/<scene_id>/images/
python src/afm_3d_search/run_pipeline.py scene_id=bude
```

First run downloads model weights (VGGT-Ω ~4.6 GB, SAM ViT-H ~2.4 GB, CLIP ViT-L/14 ~0.9 GB, DINOv2-S). Expect ~20–40 min per ~50-frame scene on an A100 slice; SAM mask generation dominates.

Output lands in `data/completed/<scene_id>/`:

```
point_cloud.ply      # aggregated colored point cloud
clip_features.npy    # per-point CLIP features (search)
dino_features.npy    # per-point DINOv2 features (structural filtering)
```

## 2. Fetch results to your laptop

```bash
rsync -avz <gpu-host>:AFM-3D-Search/data/completed/bude data/completed/
```

## 3. Search in the browser (CPU only)

```bash
uv venv .viewer_env --python 3.11 && source .viewer_env/bin/activate
uv pip install -r webviewer/requirements.txt
python webviewer/server.py data/completed/bude --port 8090
```

Open http://localhost:8090 — type queries ("backpack", "red chair"), matched points highlight; sliders control highlight percentile and point size.

## Optional: API service

```bash
# Terminal 1 — accepts image uploads on :8000
uvicorn src.afm_3d_search.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — worker polls jobs and runs the pipeline
python src/afm_3d_search/worker.py
```

POST image files to `http://localhost:8000/v1/scenes` to queue a scene.
