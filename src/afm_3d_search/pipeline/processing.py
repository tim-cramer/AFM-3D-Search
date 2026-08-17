import torch
import trimesh
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

# Helper function from the original VGGT utilities for CPU processing
def depth_to_world_coords_points(depth, extr, intr):
    H, W = depth.shape
    ys, xs = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    cam_coords = np.stack([xs, ys, np.ones_like(xs)], axis=-1)
    cam_coords = cam_coords * depth[..., None]
    cam_coords = cam_coords @ np.linalg.inv(intr).T

    hom_cam_coords = np.concatenate([cam_coords, np.ones((H, W, 1))], axis=-1)
    
    bottom_row = np.array([[0.0, 0.0, 0.0, 1.0]])
    extr_hom = np.vstack((extr, bottom_row))
    world_coords = hom_cam_coords @ np.linalg.inv(extr_hom).T

    return world_coords[..., :3], world_coords, hom_cam_coords

# The original CPU-based unprojection function
def unproject_depth_map_to_point_map_numpy(depth_maps, extrinsics, intrinsics):
    world_points_list = []
    for i in range(depth_maps.shape[0]):
        world_points, _, _ = depth_to_world_coords_points(
            depth_maps[i].squeeze(-1), extrinsics[i], intrinsics[i]
        )
        world_points_list.append(world_points)
    return np.stack(world_points_list, axis=0)

# Vectorized CPU voxel aggregation (bincount-based; the previous per-point
# Python loop took tens of minutes and gigabytes of dict overhead at 10M+ points)
def aggregate_points_and_features_numpy(points, colors, features_dict, voxel_size):
    print(f"🧊 Voxelizing and aggregating points with voxel size {voxel_size}...")
    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    voxel_indices -= voxel_indices.min(axis=0)
    extents = voxel_indices.max(axis=0) + 1
    linear = (voxel_indices[:, 0] * extents[1] + voxel_indices[:, 1]) * extents[2] + voxel_indices[:, 2]

    _, inverse = np.unique(linear, return_inverse=True)
    num_voxels = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=num_voxels).astype(np.float64)

    def mean_by_voxel(values, desc):
        out = np.empty((num_voxels, values.shape[1]), dtype=np.float32)
        for c in tqdm(range(values.shape[1]), desc=desc, leave=False):
            out[:, c] = np.bincount(inverse, weights=values[:, c].astype(np.float64),
                                    minlength=num_voxels) / counts
        return out

    agg_points = mean_by_voxel(points, "Averaging positions")
    agg_colors = mean_by_voxel(colors, "Averaging colors")
    agg_dino = mean_by_voxel(features_dict['dino'], "Averaging DINO")
    agg_clip = mean_by_voxel(features_dict['clip'], "Averaging CLIP")

    print(f"✅ Aggregation complete. Original points: {len(points)}, Aggregated points: {num_voxels}")
    return {
        "points": agg_points,
        "colors": agg_colors,
        "dino_features": agg_dino,
        "clip_features": agg_clip,
    }

def filter_and_aggregate(vggt_output_gpu: dict, feature_paths: dict, proc_cfg: DictConfig) -> dict:
    """
    Moves data to CPU and processes it using NumPy, including voxel aggregation.
    """
    print("🚚 Moving reconstruction and feature data from GPU to CPU...")
    # --- The main data transfer step ---
    depth_np = vggt_output_gpu["depth_tensor"].squeeze(0).cpu().numpy()
    extr_np = vggt_output_gpu["extrinsic_tensor"].squeeze(0).cpu().numpy()
    intr_np = vggt_output_gpu["intrinsic_tensor"].squeeze(0).cpu().numpy()
    confidence_np = vggt_output_gpu["confidence_tensor"].squeeze(0).cpu().numpy()
    images_np = vggt_output_gpu["images_tensor"].squeeze(0).cpu().numpy()
    
     # --- Load and Correct Feature Shapes ---
    print("🚚 Loading feature batches from disk...")
    dino_batches = [torch.load(p) for p in tqdm(feature_paths['dino_paths'], desc="Loading DINO")]
    dino_features_np = torch.cat(dino_batches, dim=0).numpy()
    del dino_batches  # tens of GB — free before loading CLIP

    clip_batches = [torch.load(p) for p in tqdm(feature_paths['clip_paths'], desc="Loading CLIP")]
    clip_features_np = torch.cat(clip_batches, dim=0).numpy()
    del clip_batches

    # ✅ FIX 1: Transpose DINO from (N, C, H, W) to (N, H, W, C)
    # The new order is (0, 2, 3, 1) corresponding to the original indices
    dino_features_np = np.transpose(dino_features_np, (0, 2, 3, 1))

    # ✅ FIX 2: Reshape CLIP from its mangled shape back to (N, H, W, C)
    # Assuming N=13 and H=392 from the confidence shape
    num_images = confidence_np.shape[0]
    height = confidence_np.shape[1]
    clip_features_np = clip_features_np.reshape(num_images, height, -1, clip_features_np.shape[-1])

    # --- Unprojection on CPU ---
    print("🚀 Projecting depth to points on CPU...")
    world_points = unproject_depth_map_to_point_map_numpy(depth_np, extr_np, intr_np)

    # --- Flatten all NumPy arrays for processing ---
    points_flat = world_points.reshape(-1, 3)
    colors_flat = np.transpose(images_np, (0, 2, 3, 1)).reshape(-1, 3)
    confidence_flat = confidence_np.reshape(-1)
    
    # Now this reshape will work correctly
    dino_features_flat = dino_features_np.reshape(-1, dino_features_np.shape[-1])
    clip_features_flat = clip_features_np.reshape(-1, clip_features_np.shape[-1])
    del dino_features_np, clip_features_np
    
    
    
    # --- Filtering on CPU ---
    print(f"🔍 Applying confidence filter on CPU...")
    if proc_cfg.conf_percentile > 0:
        conf_threshold = np.percentile(confidence_flat, proc_cfg.conf_percentile)
        keep_mask = confidence_flat >= conf_threshold
        
        filtered_points = points_flat[keep_mask]
        filtered_colors = colors_flat[keep_mask]
        filtered_dino = dino_features_flat[keep_mask]
        filtered_clip = clip_features_flat[keep_mask]
        
        print(f"✅ Filtering complete. Filtered point count: {len(filtered_points)}")
    else:
        filtered_points, filtered_colors = points_flat, colors_flat
        filtered_dino, filtered_clip = dino_features_flat, clip_features_flat

    # --- Voxel Aggregation on CPU ---
    if proc_cfg.voxel_size > 0:
        features_dict_unaggregated = {'dino': filtered_dino, 'clip': filtered_clip}
        final_data = aggregate_points_and_features_numpy(
            filtered_points, filtered_colors, features_dict_unaggregated, proc_cfg.voxel_size
        )
    else:
        final_data = {
            "points": filtered_points,
            "colors": filtered_colors,
            "dino_features": filtered_dino,
            "clip_features": filtered_clip,
        }
    
    return final_data

def save_artifacts(output_dir: Path, final_data_cpu: dict):
    """Saves the final NumPy arrays to disk."""
    print(f"💾 Saving final outputs to {output_dir}...")
    
    points_cpu = final_data_cpu['points']
    colors_cpu = final_data_cpu['colors']

    ply_path = output_dir / "point_cloud.ply"
    if colors_cpu.max() <= 1.0:
        colors_cpu = (colors_cpu * 255).astype(np.uint8)
    pc = trimesh.PointCloud(vertices=points_cpu, colors=colors_cpu)
    pc.export(ply_path)
    print(f"✅ Point cloud saved to {ply_path}")

    np.save(output_dir / "dino_features.npy", final_data_cpu['dino_features'])
    print(f"✅ DINO features saved")
    
    np.save(output_dir / "clip_features.npy", final_data_cpu['clip_features'])
    print(f"✅ CLIP features saved")