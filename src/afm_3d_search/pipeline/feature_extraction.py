# src/afm_3d_search/pipeline/feature_extraction.py

import torch
from omegaconf import DictConfig
from typing import List, Dict
from PIL import Image
import gc
import os
import requests
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from pathlib import Path

# --- Model Imports ---
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# --- Helper Function for Downloads ---
def _download_file(url, destination: Path):
    print(f"📦 Downloading required model: {destination.name}...")
    parent_dir = destination.parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True, desc=destination.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

# --- Helper Classes for CLIP/SigLIP + SAM Blending ---
class _ClipEncoder:
    """OpenAI CLIP (e.g. 'ViT-L/14')."""

    def __init__(self, version, device):
        self.device = device
        self.model, self.preprocess = clip.load(version.replace("_", "/"), device=self.device, jit=False)
        self.feat_dim = self.model.visual.output_dim

    @torch.no_grad()
    def encode_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        return self.model.encode_image(processed_image).float()


class _SiglipEncoder:
    """SigLIP/SigLIP2 via HF transformers (e.g. 'google/siglip2-so400m-patch14-384')."""

    def __init__(self, version, device):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(version).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(version)
        self.feat_dim = self.model.config.vision_config.hidden_size

    @torch.no_grad()
    def encode_image(self, image: np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8))
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        if not torch.is_tensor(out):  # transformers>=5 returns an output object
            out = out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]
        return out.float()


def make_image_encoder(version: str, device: str):
    if "siglip" in version.lower():
        return _SiglipEncoder(version, device)
    return _ClipEncoder(version, device)

class _MaskEmbeddingFeatureImageGenerator:
    def __init__(self, mask_generator, image_text_encoder, device):
        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.device = device
        self.feat_dim = self.image_text_encoder.feat_dim

    @torch.no_grad()
    def generate_features(self, image_np: np.ndarray):
        masks = self.mask_generator.generate(image_np)
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))

        with torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
            global_feat = self.image_text_encoder.encode_image(image_np)
            global_feat = torch.nn.functional.normalize(global_feat, dim=-1)

        H, W = image_np.shape[0], image_np.shape[1]
        # float32 accumulators: overlapping masks are BLENDED per pixel instead of
        # overwriting each other (part/whole masks otherwise win arbitrarily and
        # the same surface ends up with patchy, view-inconsistent features)
        accum = torch.zeros(H, W, self.feat_dim, dtype=torch.float32, device=self.device)
        cover = torch.zeros(H, W, dtype=torch.float32, device=self.device)
        feat_per_roi, roi_nonzero_inds, similarity_scores = [], [], []
        mean_color = image_np.reshape(-1, 3).mean(axis=0).astype(image_np.dtype)

        for mask in masks:
            _x, _y, _w, _h = map(int, mask["bbox"])
            # Mask-exact crop: pad the bbox slightly for context, then neutralize
            # everything outside SAM's segmentation so the embedding describes
            # the object, not its rectangular surroundings.
            pad = max(4, int(0.1 * max(_w, _h)))
            y0, y1 = max(0, _y - pad), min(image_np.shape[0], _y + _h + pad)
            x0, x1 = max(0, _x - pad), min(image_np.shape[1], _x + _w + pad)
            img_roi = image_np[y0:y1, x0:x1].copy()
            seg_roi = mask["segmentation"][y0:y1, x0:x1]
            if img_roi.size == 0 or not seg_roi.any(): continue
            img_roi[~seg_roi] = mean_color
            roifeat = torch.nn.functional.normalize(self.image_text_encoder.encode_image(img_roi), dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(torch.from_numpy(mask["segmentation"]).to(self.device))
            similarity_scores.append(self.cosine_similarity(global_feat, roifeat))

        if feat_per_roi:
            softmax_scores = torch.nn.functional.softmax(torch.cat(similarity_scores), dim=0)
            for i, mask_seg in enumerate(roi_nonzero_inds):
                weighted_feat = torch.nn.functional.normalize(
                    softmax_scores[i] * global_feat + (1 - softmax_scores[i]) * feat_per_roi[i], dim=-1)
                accum[mask_seg] += weighted_feat.squeeze(0).float()
                cover[mask_seg] += 1.0

        covered = cover > 0
        if covered.any():
            accum[covered] = torch.nn.functional.normalize(accum[covered], dim=-1)
        # pixels no mask ever covered: weak global-image fallback (0.5x) so they
        # stay searchable instead of being zero-feature holes
        accum[~covered] = 0.5 * global_feat.squeeze(0).float()
        return accum.half()


# --- Main Feature Extraction Functions ---
def extract_dino_features_from_pil(pil_images, dino_version, target_height, target_width, device, batch_size, output_dir, sink=None):

    print(f"🦖 Initializing DINOv2 model ({dino_version})...")
    dinov2_model = torch.hub.load('facebookresearch/dinov2', dino_version, verbose=False).to(device).eval()

    temp_features_dir = output_dir / "temp_dino_features"
    temp_features_dir.mkdir(parents=True, exist_ok=True)

    # DINOv2 requires dims divisible by its patch size (14); the recon backbone
    # may produce e.g. 16-divisible dims. Run DINO on the nearest 14-divisible
    # size, then interpolate features back to the target grid below.
    dino_height = max(14, round(target_height / 14) * 14)
    dino_width = max(14, round(target_width / 14) * 14)
    dino_transforms = transforms.Compose([
        transforms.Resize((dino_height, dino_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feature_file_paths = []
    print(f"🦖 Extracting DINOv2 features from {len(pil_images)} images...")
    with torch.no_grad():
        for i in tqdm(range(0, len(pil_images), batch_size), desc="DINOv2 Batches"):
            image_batch = pil_images[i:i+batch_size]
            transformed_images = torch.stack([dino_transforms(p) for p in image_batch]).to(device)

            features_dict = dinov2_model.forward_features(transformed_images)
            patch_features = features_dict['x_norm_patchtokens']

            B, N, D = patch_features.shape
            H_patch = dino_height // 14
            W_patch = dino_width // 14

            feature_map_2d = patch_features.reshape(B, H_patch, W_patch, D).permute(0, 3, 1, 2)
            upsampled_features = torch.nn.functional.interpolate(
                feature_map_2d, size=(target_height, target_width), mode='bilinear', align_corners=False
            )
            if sink is not None:
                for j in range(upsampled_features.shape[0]):
                    sink(i + j, upsampled_features[j].permute(1, 2, 0))
            else:
                batch_filepath = temp_features_dir / f"batch_{i}.pt"
                # fp16 halves the on-disk footprint (~300MB/frame at full res otherwise)
                torch.save(upsampled_features.half().cpu(), batch_filepath)
                feature_file_paths.append(batch_filepath)
    del dinov2_model
    torch.cuda.empty_cache()

    return feature_file_paths 


def extract_clip_features_from_pil(pil_images: List[Image.Image], clip_version: str, sam_checkpoint_filename: str, target_height: int, target_width: int, device: str, output_dir: Path, sink=None) -> List[Path]:
    print("📎 Initializing SAM and CLIP models...")
    
    WEIGHTS_DIR = Path("weights")
    sam_checkpoint_path = WEIGHTS_DIR / sam_checkpoint_filename
    
    if not sam_checkpoint_path.exists():
        _download_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_checkpoint_path)

    sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path).to(device)

    # crop_n_layers=1: extra AMG pass on image crops for small-object coverage;
    # min_mask_region_area filters speckle masks (needs opencv)
    mask_generator = SamAutomaticMaskGenerator(sam_model, crop_n_layers=1, min_mask_region_area=100)
    clip_encoder = make_image_encoder(clip_version, device)
    feature_generator = _MaskEmbeddingFeatureImageGenerator(mask_generator, clip_encoder, device)

    # --- Create a temporary directory for CLIP feature batches ---
    temp_features_dir = output_dir / "temp_clip_features"
    temp_features_dir.mkdir(parents=True, exist_ok=True)

    feature_file_paths = []
    print(f"📎 Extracting CLIP (SAM-blended) features from {len(pil_images)} images...")
    with torch.no_grad():
        for i, image in enumerate(tqdm(pil_images, desc="CLIP Features")):
            resized_image = image.resize((target_width, target_height))
            # Generate the feature tensor for the single image
            feature_tensor = feature_generator.generate_features(np.array(resized_image))

            if sink is not None:
                sink(i, feature_tensor)
            else:
                image_filepath = temp_features_dir / f"image_{i}.pt"
                torch.save(feature_tensor.cpu(), image_filepath)
                feature_file_paths.append(image_filepath)

    del sam_model, mask_generator, clip_encoder, feature_generator
    gc.collect()
    torch.cuda.empty_cache()

    return feature_file_paths


# --- Main Run Function ---
def run(pil_images: List[Image.Image], vggt_output: Dict, cfg: DictConfig, device: str,  output_dir, aggregator=None) -> Dict:
    """Extracts all configured features. With an aggregator, features are
    voxel-accumulated on the fly instead of being written to temp files."""

    dino_paths = extract_dino_features_from_pil(
        pil_images, cfg.models.dino.version, vggt_output['height'], vggt_output['width'], device, cfg.processing.dino_batch_size, output_dir,
        sink=aggregator.add_dino if aggregator is not None else None
    )

    clip_paths = extract_clip_features_from_pil(
        pil_images, cfg.models.clip.version, cfg.models.sam.checkpoint, vggt_output['height'], vggt_output['width'], device, output_dir,
        sink=aggregator.add_clip if aggregator is not None else None
    )

    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "dino_paths": dino_paths,
        "clip_paths": clip_paths
    }