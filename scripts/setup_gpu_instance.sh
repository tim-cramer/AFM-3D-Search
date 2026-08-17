#!/bin/bash
# Set up a fresh Ubuntu 22/24 GPU instance for the AFM-3D-Search pipeline.
# Tested target: Jetstream2 g3.large (half A100, 20 GB) — VGGT-Omega fits
# ~50-frame scenes comfortably in that budget.
set -euo pipefail

BRANCH="${1:-main}"

# 0. GPU driver ships with the Jetstream2 featured images
nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA driver/GPU visible"; exit 1; }

# 1. uv
command -v uv >/dev/null || {
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
}

# 2. Repo + submodules
if [ ! -d AFM-3D-Search ]; then
    git clone --branch "$BRANCH" --recursive https://github.com/tim-cramer/AFM-3D-Search.git
fi
cd AFM-3D-Search

# 3. Environment
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
uv pip install -e submodules/vggt   # original VGGT backbone (optional fallback)

# 4. VGGT-Omega (default backbone). --no-deps keeps our torch pin; its
#    runtime deps (torch, einops, safetensors, numpy, Pillow, huggingface_hub)
#    are already in our environment.
uv pip install --no-deps "vggt-omega @ git+https://github.com/facebookresearch/vggt-omega.git"
python -c "from vggt_omega.models import VGGTOmega; print('vggt_omega import OK')" || {
    echo "WARN: vggt_omega import failed — installing with its own requirements"
    uv pip install "vggt-omega @ git+https://github.com/facebookresearch/vggt-omega.git"
}

# 5. Pre-download reconstruction weights (~4.6 GB; SAM/CLIP/DINO download on
#    first pipeline run). facebook/VGGT-Omega is GATED on Hugging Face: accept
#    the Fair Noncommercial Research License in the browser, then authenticate
#    here once with `huggingface-cli login` (or export HF_TOKEN).
python - <<'EOF' || echo "WARN: VGGT-Omega weights not cached yet — accept the license at https://huggingface.co/facebook/VGGT-Omega and run 'huggingface-cli login', then re-run this script."
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="facebook/VGGT-Omega", filename="vggt_omega_1b_512.pt")
print("VGGT-Omega weights cached")
EOF

echo
echo "Done. Upload images to data/testing/<scene>/images/ then run:"
echo "  source AFM-3D-Search/.venv/bin/activate"
echo "  python src/afm_3d_search/run_pipeline.py scene_id=<scene>"
