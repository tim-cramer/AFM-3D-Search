import hydra
from pathlib import Path
import torch
import os
from PIL import Image

# Import the refactored modules
from pipeline import reconstruction, feature_extraction, processing
from conf.schema import MainConfig

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: MainConfig) -> None: 
    """Orchestrates the entire ML pipeline for a given scene."""
    
    # --- 1. Setup ---
    print(f"--- Starting Pipeline for Scene: {cfg.scene_id} ---")
    
    data_root = Path(cfg.paths.data_root)
    image_dir = data_root / cfg.paths.raw_dir_name / cfg.scene_id / "images"
    output_dir = data_root / cfg.paths.completed_dir_name / cfg.scene_id
    
    print(f"Reading images from: {image_dir}")
    print(f"Saving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    image_paths = sorted([str(p) for p in image_dir.glob('*')])

    if not image_paths:
        raise FileNotFoundError(f"No images found in '{image_dir}'")

    print(f"Pre-loading {len(image_paths)} images into memory...")
    pil_images = [Image.open(p).convert("RGB") for p in image_paths]

    vggt_output_gpu = reconstruction.run(image_paths, pil_images, cfg, device, dtype)
    feature_paths = feature_extraction.run(pil_images, vggt_output_gpu, cfg, device, output_dir)
    final_data_cpu = processing.filter_and_aggregate(vggt_output_gpu, feature_paths, cfg.processing)
    processing.save_artifacts(output_dir, final_data_cpu)

    import shutil
    shutil.rmtree(output_dir / "temp_dino_features", ignore_errors=True)
    shutil.rmtree(output_dir / "temp_clip_features", ignore_errors=True)
    
    print(f"--- ✅ Successfully Processed Scene: {cfg.scene_id} ---")

if __name__ == "__main__":
    main()