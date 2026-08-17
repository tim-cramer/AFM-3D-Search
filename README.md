# AFM 3D Search — Natural-Language Object Search in 3D Scenes

Reconstruct a 3D scene from plain RGB images and search it with natural language ("red chair", "laptop on the desk") — training-free, built entirely from pretrained foundation models.

**Team:** Marco Lorenz ([@lorenz369](https://github.com/lorenz369)), Sami Haddouti ([@SamiHaddouti](https://github.com/SamiHaddouti)), Tim Cramer ([@tim-cramer](https://github.com/tim-cramer)) — developed in the Applied Foundation Models practical course at TUM.

## How it works

1. **Reconstruction** — [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) (default) or [VGGT](https://github.com/facebookresearch/vggt) turns RGB images into dense per-frame depth + camera poses, unprojected to a 3D point cloud (`src/afm_3d_search/pipeline/reconstruction.py`). The backbone is switchable via `models.recon.backbone=vggt_omega|vggt`.
2. **Featurization** — each point is enriched with CLIP semantics (SAM-mask-blended, ViT-L/14) and DINOv2 structural features (`src/afm_3d_search/pipeline/feature_extraction.py`).
3. **Search & visualization** — a browser-based viewer ([`webviewer/`](webviewer/)): type a query, it is CLIP-encoded on CPU and matched against the featurized cloud; hits light up in 3D. No GPU needed for this step.

A FastAPI service with a job queue (`src/afm_3d_search/api/`, `worker.py`) wraps the pipeline for end-to-end scene processing. See [DEMO.md](DEMO.md) for a step-by-step walkthrough.

## Setup (pipeline, GPU machine)

```bash
git clone --recursive https://github.com/tim-cramer/AFM-3D-Search.git
cd AFM-3D-Search
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install -e submodules/vggt                                        # only for backbone=vggt
uv pip install --no-deps "vggt-omega @ git+https://github.com/facebookresearch/vggt-omega.git"
```

Or run [`scripts/setup_gpu_instance.sh`](scripts/setup_gpu_instance.sh) on a fresh Ubuntu GPU box.

**GPU requirements:** peak memory is the reconstruction forward pass over all frames at once. With VGGT-Ω, ~50 frames fit comfortably in 20 GB (half A100); original VGGT needs roughly 3× that. SAM/CLIP/DINO run sequentially afterwards and fit in 8 GB.

## Running the pipeline

Images go in `data/testing/<scene_id>/images/`, artifacts come out in `data/completed/<scene_id>/` (`point_cloud.ply`, `clip_features.npy`, `dino_features.npy`):

```bash
python src/afm_3d_search/run_pipeline.py scene_id=bude
python src/afm_3d_search/run_pipeline.py scene_id=bude models.recon.backbone=vggt   # original VGGT
```

## Interactive search (no GPU)

```bash
python webviewer/server.py data/completed/bude --port 8090   # then open http://localhost:8090
```

See [webviewer/README.md](webviewer/README.md) for details and a synthetic test scene that works without any processed data.

## Data

Demo recordings and results: [Google Drive folder](https://drive.google.com/drive/folders/184vJEGNb4RQ5tb9fF1LaFxy98oRriyPi?usp=drive_link).

## History

Earlier iterations used [Rerun](https://rerun.io/) for visualization and MASt3R-SLAM / ARKit VIO as reconstruction sources — see git history (`rerun/` before this branch) and the `eval` branch for benchmark scripts.
