# src/afm_3d_search/conf/schema.py

from dataclasses import dataclass, field
from typing import Optional

# Using nested dataclasses for clear organization
@dataclass
class ClipModelConfig:
    version: str

@dataclass
class SamModelConfig:
    checkpoint: str

@dataclass
class DinoModelConfig:
    version: str

@dataclass
class ReconModelConfig:
    backbone: str = "vggt_omega"
    image_resolution: int = 512

@dataclass
class ModelsConfig:
    clip: ClipModelConfig
    sam: SamModelConfig
    dino: DinoModelConfig
    recon: ReconModelConfig = field(default_factory=ReconModelConfig)

@dataclass
class PathsConfig:
    # Base directory for all data
    data_root: str = "/workspace/AFM-3D-Search/data"
    # Subdirectory for raw images
    raw_dir_name: str = "testing" 
    # Subdirectory for pipeline outputs
    completed_dir_name: str = "completed"
    
    # These will be derived in the code, not set here
    ply_filename: str = "point_cloud.ply"
    clip_features_filename: str = "clip_features.npy"
    dino_features_filename: str = "dino_features.npy"

@dataclass
class ProcessingConfig:
    conf_percentile: float
    voxel_size: float
    dino_batch_size: int
    clip_batch_size: int
    streaming: bool = True

@dataclass
class HighlightConfig:
    # Parameters for the highlight scripts
    text_query: str = "a bed"
    query_point_index: int = 10000
    top_k: int = 5000

@dataclass
class MainConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    
    scene_id: str = "???"