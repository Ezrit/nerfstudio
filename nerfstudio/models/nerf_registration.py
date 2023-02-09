"""
Implementation of a nerf registration network (optimizer).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import wandb
import yaml
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)

# from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components.losses import MSELoss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps


def load_checkpoint(config, pipeline: Pipeline) -> Path:
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
    # assert isinstance(config, TrainerConfig)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    assert isinstance(pipeline, Pipeline)
    pipeline.requires_grad_(False)

    # load checkpointed information
    _ = load_checkpoint(config, pipeline)
    model: Model = pipeline._model

    return model


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    # Normalize and unpack the quaternion
    qw, qx, qy, qz = quaternion / torch.norm(quaternion)

    # Compute the rotation matrix
    matrix = torch.zeros((3, 3), device=quaternion.device)

    matrix[0, 0] = 1. - 2. * qy ** 2 - 2. * qz ** 2
    matrix[1, 1] = 1. - 2. * qx ** 2 - 2. * qz ** 2
    matrix[2, 2] = 1. - 2. * qx ** 2 - 2. * qy ** 2

    matrix[0, 1] = 2. * qx * qy - 2. * qz * qw
    matrix[1, 0] = 2. * qx * qy + 2. * qz * qw

    matrix[0, 2] = 2. * qx * qz + 2 * qy * qw
    matrix[2, 0] = 2. * qx * qz - 2 * qy * qw

    matrix[1, 2] = 2. * qy * qz - 2. * qx * qw
    matrix[2, 1] = 2. * qy * qz + 2. * qx * qw
    return matrix


@dataclass
class NerfRegistrationConfig(ModelConfig):
    """Nerf Registration Config"""

    _target: Type = field(default_factory=lambda: NerfRegistrationModel)
    main_method_config: Path = Path()
    """main model which defines the world coordinates"""
    sub_method_config: Path = Path()
    """sub to align to the main model"""
    fake_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """fake translation to test to train"""
    fake_scaling: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """fake scaling to test to train"""
    fake_rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """fake rotation to test to train"""


class NerfRegistrationModel(Model):
    """Nerf Registration model

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: NerfRegistrationConfig

    def log_wand_3d_nerf_box(self, number_ray_vis: int, coordinate_system_length: float, ray_length: float,  step: int) -> None:
        boxes = []
        points = []
        vectors = []

        # create main nerf box
        main_box_points: TensorType[8, 4, 1] = torch.tensor([[[i], [j], [k], [1]] for i in [-1.0, 1.0] for j in [-1.0, 1.0] for k in [-1.0, 1.0]], device=self.device)
        main_box = {
            "corners": main_box_points[:, :3, 0].tolist(),
            # optionally customize each label
            "label": "main NeRF",
            "color": [0, 255, 0],
        }
        main_box_points_render = torch.clone(main_box_points[:, :, 0])
        main_box_points_render[:, 3] = 0
        points += main_box_points_render.tolist()
        boxes.append(main_box)

        # visualize coordinate system of main nerf box
        coordinate_origin_main: TensorType[1, 4, 1] = torch.tensor([[[0], [0], [0], [1]]], dtype=torch.float, device=self.device)
        coordinate_directions_main: TensorType[3, 4, 1] = torch.tensor([[[1], [0], [0], [0]], [[0], [1], [0], [0]], [[0], [0], [1], [0]]], dtype=torch.float, device=self.device)
        main_coord_vector = [{"start": coordinate_origin_main[0, :3, 0].tolist(), "end": (coordinate_origin_main[0, :3, 0] + coordinate_system_length * coordinate_directions_main[i, :3, 0]).tolist()} for i in range(3)]
        vectors += main_coord_vector

        # create sub nerf box
        # The transformation the network learns currently is a transformation from the main NeRF to the sub NeRF:
        #   Ray origins and directions are created in the main NeRF coordinate sytem and then transformed to the sub NeRF coordinate system
        #   Therefore, the translation is main -> sub!
        # Since this visualization is in the main NeRF coordinate system, we need to inverse the transformation and apply it on the box points for the sub NeRF!

        # compute inverse of fake transformation for computing the sub nerf box points
        inv_fake_transformation: TensorType[4, 4] = torch.eye(4, device=self.device)
        inv_fake_transformation[:3, :3] = ((1.0 / self.fake_scaling[0]) * torch.eye(3, device=self.device)) @ quaternion_to_rotation_matrix(self.fake_rotation).T
        inv_fake_transformation[:3, 3] = -inv_fake_transformation[:3, :3] @ self.fake_translation

        # # compute inverse of the (currently) learned transformation for computing the sub nerf box points
        # inv_transformation: TensorType[4, 4] = torch.eye(4, device=self.device)
        # inv_transformation[:3, :3] = ((1.0 / self.scale[0]) * torch.eye(3, device=self.device)) @ quaternion_to_rotation_matrix(self.rotation).T
        # inv_transformation[:3, 3] = -inv_transformation[:3, :3] @ self.translate

        # apply transformations
        sub_box_points = torch.matmul(inv_fake_transformation @ self.sub_to_main_transformation, main_box_points)
        sub_box = {
            "corners": sub_box_points[:, :3, 0].tolist(),
            # optionally customize each label
            "label": "sub NeRF",
            "color": [255, 255, 0],
        }
        sub_box_points_render = torch.clone(sub_box_points[:, :, 0])
        sub_box_points_render[:, 3] = 0
        points += sub_box_points_render.tolist()
        boxes.append(sub_box)

        # visualize coordinate system of sub nerf box
        coordinate_origin_sub: TensorType[1, 4, 1] = torch.matmul(inv_fake_transformation @ self.sub_to_main_transformation, coordinate_origin_main)
        coordinate_directions_sub: TensorType[3, 4, 1] = torch.matmul(inv_fake_transformation @ self.sub_to_main_transformation, coordinate_directions_main)
        sub_coord_vector = [{"start": coordinate_origin_sub[0, :3, 0].tolist(), "end": (coordinate_origin_sub[0, :3, 0] + coordinate_system_length * coordinate_directions_sub[i, :3, 0]).tolist()} for i in range(3)]
        vectors += sub_coord_vector

        # get the sub origin and assign a group value for rendering the points
        current_sub_origin: TensorType[1, 4] = coordinate_origin_sub[:, :, 0]
        current_sub_origin[0, 3] = 2
        if self.all_sub_origins is not None:
            self.all_sub_origins = torch.cat([self.all_sub_origins, current_sub_origin], 0)
        else:
            self.all_sub_origins = current_sub_origin
        points += self.all_sub_origins.tolist()

        # visiualize some rays used in the last step
        if self.last_ray_origins is not None:
            # visualize origin points
            points += torch.cat([self.last_ray_origins[:number_ray_vis, :], torch.ones_like(self.last_ray_origins[:number_ray_vis, :1])], 1).tolist()
            # visualize directions
            if self.last_ray_directions is not None:
                # TODO: this seems weird and wrong, maybe there is a more elegant way?
                ray_vector = [{"start": self.last_ray_origins[i, :].tolist(), "end": (self.last_ray_origins[i, :] + ray_length * self.last_ray_directions[i, :]).tolist()} for i in range(number_ray_vis)]
                vectors += ray_vector

        boxes = np.array(boxes)
        points = np.array(points)
        vectors = np.array(vectors)

        wandb.log(
            {
                "NeRF Boxes": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": points,
                        "boxes": boxes,
                        "vectors": vectors,
                    }
                )
            }, step=step
        )

    @property
    def main_to_sub_transformation(self) -> TensorType[4, 4]:
        transformation: TensorType[4, 4] = torch.eye(4, device=self.device)
        transformation[:3, :3] = (self.scale[0] * torch.eye(3, device=self.device)) @ quaternion_to_rotation_matrix(self.rotation)
        transformation[:3, 3] = self.translate
        return transformation

    @property
    def sub_to_main_transformation(self) -> TensorType[4, 4]:
        transformation: TensorType[4, 4] = torch.eye(4, device=self.device)
        transformation[:3, :3] = ((1.0 / self.scale[0]) * torch.eye(3, device=self.device)) @ quaternion_to_rotation_matrix(self.rotation).T
        transformation[:3, 3] = -transformation[:3, :3] @ self.translate
        return transformation

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks: List[TrainingCallback] = []
        if wandb.run is not None:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=50,
                    func=self.log_wand_3d_nerf_box,
                    kwargs={"number_ray_vis": 200, "coordinate_system_length": 0.5, "ray_length": 0.1},
                )
            )
        return callbacks

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

        self.fake_translation = torch.nn.Parameter(torch.tensor(self.config.fake_translation))
        self.fake_scaling = torch.nn.Parameter(torch.tensor(self.config.fake_scaling))
        self.fake_rotation = torch.nn.Parameter(torch.tensor(self.config.fake_rotation))

        # load and fix main model
        self.main_model = load_model_from_config(self.config.main_method_config)
        self.main_model.requires_grad_(False)

        # load and fix sub model
        self.sub_model = load_model_from_config(self.config.sub_method_config)
        self.sub_model.requires_grad_(False)

        # losses
        self.rgb_loss = MSELoss()

        self.last_ray_origins: Optional[TensorType[..., 3]] = None
        self.last_ray_directions: Optional[TensorType[..., 3]] = None
        self.all_sub_origins: Optional[TensorType[..., 4]] = None

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # TODO: still a bit confused by this...
        param_groups["transform"] = [self.translate, self.rotation, self.scale]
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        outputs = {}

        # save the new origins and directions for wandb visualization
        if wandb.run is not None:
            self.last_ray_origins = ray_bundle.origins
            self.last_ray_directions = ray_bundle.directions

        # get main output
        with torch.no_grad():
            main_outputs = self.main_model(ray_bundle)

        # apply the current transformation to ray bundle
        # first apply fake transforms
        # TODO: remove this, it is just to test the method
        fake_transform = torch.eye(4, device=self.device)
        fake_scaling = self.fake_scaling[0] * torch.eye(3, device=self.device)
        fake_rotation = quaternion_to_rotation_matrix(self.fake_rotation)
        fake_transform[:3, :3] = fake_scaling @ fake_rotation @ fake_transform[:3, :3]
        fake_transform[:3, 3] = self.fake_translation + fake_transform[:3, 3]

        # apply the transformation matrix to the rays origins and directions
        ray_bundle.origins = torch.matmul(self.main_to_sub_transformation @ fake_transform, torch.cat([ray_bundle.origins, torch.ones((*ray_bundle.origins.shape[:-1], 1), dtype=ray_bundle.origins.dtype, device=self.device)], 1).view(-1, 4, 1)).view(*ray_bundle.origins.shape[:-1], 4)[..., :3]
        ray_bundle.directions = torch.matmul(self.main_to_sub_transformation @ fake_transform, torch.cat([ray_bundle.directions, torch.zeros((*ray_bundle.directions.shape[:-1], 1), dtype=ray_bundle.directions.dtype, device=self.device)], 1).view(-1, 4, 1)).view(*ray_bundle.origins.shape[:-1], 4)[..., :3]

        # get sub output
        sub_outputs = self.sub_model(ray_bundle)

        outputs = {f"main_{key}": value for key, value in main_outputs.items()}
        outputs.update({f"sub_{key}": value for key, value in sub_outputs.items()})

        return outputs

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = {}

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

        return metrics_dict

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

