import torch
from PIL import Image
from typing import List, Dict
import torchvision.transforms.functional as TF
import numpy as np
import gc

VGGT_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
VGGT_OMEGA_WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-Omega/resolve/main/vggt_omega_1b_512.pt"


def run(image_paths: List[str], pil_images: List[Image.Image], cfg, device: str, dtype: torch.dtype) -> Dict:
    """Dispatch to the configured reconstruction backbone.

    Every backbone returns the same contract dict consumed by
    feature_extraction and processing:
        depth_tensor (1,S,H,W,1), confidence_tensor (1,S,H,W),
        images_tensor (1,S,3,H,W), extrinsic_tensor (1,S,3,4),
        intrinsic_tensor (1,S,3,3), height, width
    """
    backbone = cfg.models.recon.backbone
    if backbone == "vggt_omega":
        return run_vggt_omega(image_paths, cfg.models.recon.image_resolution, device, dtype)
    elif backbone == "vggt":
        return run_vggt(pil_images, device, dtype)
    raise ValueError(f"Unknown reconstruction backbone: {backbone!r} (expected 'vggt_omega' or 'vggt')")


def _with_batch_dim(t: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Ensure a leading batch dimension of 1 in front of the frame dimension."""
    return t.unsqueeze(0) if t.shape[0] == num_frames else t


def run_vggt_omega(image_paths: List[str], image_resolution: int, device: str, dtype: torch.dtype) -> Dict:
    """VGGT-Omega (CVPR 2026) — ~30% of VGGT's memory, 1.6x faster inference."""
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    print(f"🔄 Initializing VGGT-Omega on {device}...")
    model = VGGTOmega()
    state_dict = torch.hub.load_state_dict_from_url(VGGT_OMEGA_WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval().to(device)
    del state_dict

    images = load_and_preprocess_images(image_paths, image_resolution=image_resolution).to(device)
    num_frames = images.shape[0] if images.dim() == 4 else images.shape[1]

    print(f"🚀 Running VGGT-Omega inference on {num_frames} frames...")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda"):
        predictions = model(images)

    depth = _with_batch_dim(predictions["depth"], num_frames)
    if depth.shape[-1] != 1:
        depth = depth[..., None]
    confidence = _with_batch_dim(predictions["depth_conf"], num_frames)
    images_tensor = _with_batch_dim(predictions.get("images", images), num_frames)
    H, W = images_tensor.shape[-2:]

    extrinsic, intrinsic = encoding_to_camera(predictions["pose_enc"], (H, W))
    extrinsic = _with_batch_dim(extrinsic, num_frames)
    intrinsic = _with_batch_dim(intrinsic, num_frames)

    print("🧹 Cleaning up VGGT-Omega model from GPU memory...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "depth_tensor": depth,
        "confidence_tensor": confidence,
        "images_tensor": images_tensor,
        "extrinsic_tensor": extrinsic,
        "intrinsic_tensor": intrinsic,
        "height": H,
        "width": W,
    }


def _preprocess_single_image(img: Image.Image, mode: str = "crop", target_size: int = 518) -> torch.Tensor:
    """Applies the specific preprocessing steps from the original project to a single PIL image."""
    
    # If there's an alpha channel, blend onto a white background
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    
    # Calculate new dimensions, ensuring they are divisible by 14
    if mode == "pad":
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
    else:  # mode == "crop"
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_tensor = TF.to_tensor(img)  # Convert to tensor (0, 1)

    if mode == "crop" and new_height > target_size:
        img_tensor = TF.center_crop(img_tensor, [target_size, target_size])

    if mode == "pad":
        h_padding = target_size - img_tensor.shape[1]
        w_padding = target_size - img_tensor.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            img_tensor = torch.nn.functional.pad(
                img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
            
    return img_tensor


# --- Main function for the pipeline ---
def run_vggt(pil_images: List[Image.Image], device: str, dtype: torch.dtype) -> Dict:
    """
    Runs VGGT reconstruction and returns a dictionary of raw GPU tensors.
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    print(f"🔄 Initializing VGGT model on {device}...")
    vggt_model = VGGT()
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_WEIGHTS_URL, map_location=device))
    vggt_model.eval().to(device)

    # Preprocess all images using our new helper function
    print("Pre-processing images with original project logic...")
    processed_images = [_preprocess_single_image(img, mode="crop") for img in pil_images]

    # Stacking logic to handle potentially different shapes after processing
    shapes = {img.shape for img in processed_images}
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes after processing: {shapes}. Padding to match.")
        max_height = max(shape[1] for shape in shapes)
        max_width = max(shape[2] for shape in shapes)
        
        padded_images = []
        for img in processed_images:
            h, w = img.shape[1], img.shape[2]
            h_padding = max_height - h
            w_padding = max_width - w
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
            padded_images.append(img)
        processed_images = padded_images

    # Create the final batch tensor for the model
    images_tensor = torch.stack(processed_images).unsqueeze(0).to(device)

    print("🚀 Running VGGT Inference...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = vggt_model(images_tensor)

    print("✅ VGGT Inference complete.")
    B, S, C, H, W = predictions["images"].shape
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

    print("🧹 Cleaning up VGGT model from GPU memory...")
    del vggt_model, images_tensor
    gc.collect(); torch.cuda.empty_cache()

    return {
        "depth_tensor": predictions["depth"],
        "confidence_tensor": predictions["depth_conf"],
        "images_tensor": predictions["images"], 
        "extrinsic_tensor": extrinsic,
        "intrinsic_tensor": intrinsic,
        "height": H,
        "width": W
    }