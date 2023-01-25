"""
Implementation of a nerf registration network (optimizer).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import torch
import yaml
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps


def load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Path:
    # TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")
    pipeline.load_pipeline(loaded_state["pipeline"])
    return load_path


def load_model_from_config(
    config_path: Path,
) -> Model:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test datset into memory
            'inference': does not load any dataset into memory


    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=None)
    assert isinstance(pipeline, Pipeline)

    # load checkpointed information
    _ = load_checkpoint(config, pipeline)
    model: Model = pipeline._model

    return model


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    # Normalize the quaternion
    quaternion = quaternion/torch.norm(quaternion)
    # Unpack the quaternion
    w, x, y, z = quaternion
    # Compute the rotation matrix
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*w*z
    r02 = 2*x*z + 2*w*y
    r10 = 2*x*y + 2*w*z
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*w*x
    r20 = 2*x*z - 2*w*y
    r21 = 2*y*z + 2*w*x
    r22 = 1 - 2*x*x - 2*y*y
    return torch.tensor([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=quaternion.dtype, device=quaternion.device)


@dataclass
class NerfRegistrationConfig(ModelConfig):
    """Nerf Registration Config"""

    _target: Type = field(default_factory=lambda: NerfRegistrationModel)
    main_method_config: Path = Path()
    """main model which defines the world coordinates"""
    main_method_name: Optional[str] = None
    """main model which defines the world coordinates"""
    main_load_dir: Optional[Path] = None
    """Specify a pre-trained model directory to load from."""
    main_load_step: Optional[Path] = None
    sub_method_config: Path = Path()
    """sub to align to the main model"""
    sub_method_name: Optional[str] = None
    """model to align to the main model"""
    sub_load_dir: Optional[Path] = None
    """Specify a pre-trained model directory to load from."""
    sub_load_step: Optional[Path] = None


class NerfRegistrationModel(Model):
    """Nerf Registration model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: NerfRegistrationConfig

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # safe parameters to be optimized
        self.translate = torch.nn.Parameter(torch.zeros(3))
        self.translate.requires_grad_(True)
        self.rotation = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        self.rotation.requires_grad_(True)
        self.scale = torch.nn.Parameter(torch.ones(3))
        self.scale.requires_grad_(True)

        # load and fix main model
        self.main_model = load_model_from_config(self.config.main_method_config)
        self.main_model.requires_grad_(False)
        self.main_model.

        # load and fix sub model
        self.sub_model = load_model_from_config(self.config.sub_method_config)
        self.sub_model.requires_grad_(False)

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # TODO:????
        # param_groups["translation"] = list(self.translate.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = {}

        # get main output
        with torch.no_grad():
            main_outputs = self.main_model(ray_bundle)

        # apply the current transformation to ray bundle
        transform = torch.eye(4, device=self.device)
        scale = torch.diag(self.scale)
        rotation = quaternion_to_rotation_matrix(self.rotation)
        scale_rotate = scale @ rotation
        transform[:3, :3] = scale_rotate
        transform[:3, 3] = self.translate

        ray_bundle.origins = torch.matmul(transform, torch.cat([ray_bundle.origins, torch.ones((*ray_bundle.origins.shape[:-1], 1), dtype=ray_bundle.origins.dtype, device=self.device)], 1).view(-1, 4, 1)).view(*ray_bundle.origins.shape[:-1], 4)[..., :3]
        ray_bundle.directions = torch.matmul(transform, torch.cat([ray_bundle.directions, torch.zeros((*ray_bundle.directions.shape[:-1], 1), dtype=ray_bundle.directions.dtype, device=self.device)], 1).view(-1, 4, 1)).view(*ray_bundle.origins.shape[:-1], 4)[..., :3]

        # get sub output
        with torch.no_grad():
            sub_outputs = self.sub_model(ray_bundle)

        outputs = {f"main_{key}": value for key, value in main_outputs.items()}
        outputs.update({f"sub_{key}": value for key, value in sub_outputs.items()})

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        loss_dict["rgb_loss"] = self.rgb_loss(outputs["main_rgb"], outputs["sub_rgb"])

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        main_rgb = outputs["main_rgb"]
        sub_rgb = outputs["sub_rgb"]
        main_acc = colormaps.apply_colormap(outputs["main_accumulation"])
        sub_acc = colormaps.apply_colormap(outputs["sub_accumulation"])
        main_depth = colormaps.apply_depth_colormap(
            outputs["main_depth"],
            accumulation=outputs["main_accumulation"],
        )
        sub_depth = colormaps.apply_depth_colormap(
            outputs["sub_depth"],
            accumulation=outputs["sub_accumulation"],
        )

        combined_rgb = torch.cat([main_rgb, sub_rgb], dim=1)
        combined_acc = torch.cat([main_acc, sub_acc], dim=1)
        combined_depth = torch.cat([main_depth, sub_depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        main_rgb = torch.moveaxis(main_rgb, -1, 0)[None, ...]
        sub_rgb = torch.moveaxis(sub_rgb, -1, 0)[None, ...]

        psnr = self.psnr(main_rgb, sub_rgb)
        ssim = self.ssim(main_rgb, sub_rgb)
        lpips = self.lpips(main_rgb, sub_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        metrics_dict["translation_x"] = float(self.translate[0])
        metrics_dict["translation_y"] = float(self.translate[1])
        metrics_dict["translation_z"] = float(self.translate[2])

        metrics_dict["rotation_w"] = float(self.rotation[0])
        metrics_dict["rotation_x"] = float(self.rotation[1])
        metrics_dict["rotation_y"] = float(self.rotation[2])
        metrics_dict["rotation_z"] = float(self.rotation[3])

        metrics_dict["scale_x"] = float(self.scale[0])
        metrics_dict["scale_y"] = float(self.scale[1])
        metrics_dict["scale_z"] = float(self.scale[2])

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict

