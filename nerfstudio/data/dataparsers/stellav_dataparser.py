"""Data parser for stella vslam data"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Literal, Optional, Type

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1600


@dataclass
class StellaVSlamDataParserConfig(DataParserConfig):
    """StellaVSlam dataset config"""

    _target: Type = field(default_factory=lambda: StellaVSlam)
    """target class to instantiate"""
    data: Path = Path("data/stellaV/aist_store_1")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 0.33
    """How much to scale the scene."""
    orientation_method: Literal["pca", "up"] = "up"
    """The method to use for orientation."""


@dataclass
class StellaVSlam(DataParser):
    """Stella VSlam DatasetParser"""

    config: StellaVSlamDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        img_filenames = []
        poses = []

        with open(self.config.data / "images.txt", 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name = line.split()

                trans_cw = np.array([float(tx), float(ty), float(tz)], dtype=np.float32)
                rotationQuat_w2c = Quaternion(float(qw), float(qx), float(qy), float(qz))

                rotationQuat_c2w = rotationQuat_w2c.inverse
                rot_matrix_c2w = rotationQuat_c2w.rotation_matrix

                pos = -rot_matrix_c2w @ trans_cw
                pos = np.expand_dims(pos, axis=1)

                img_filenames.append(self.config.data / 'images' / img_name)
                poses.append(torch.from_numpy(np.append(rot_matrix_c2w, pos, axis=1).astype(np.float32)))

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses = camera_utils.auto_orient_poses(poses, method=self.config.orientation_method)

        # Scale poses
        scale_factor = 1.0 / torch.max(torch.abs(poses[:, :3, 3]))
        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        width = 800
        height = 400

        with open(self.config.data / "cameras.txt"):
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                cam_id, cam_model, width, height, _ = line.split()
                break

        distortion_params = camera_utils.get_distortion_params(
            k1=0.0,
            k2=0.0,
            k3=0.0,
            k4=0.0,
            p1=0.0,
            p2=0.0,
        )

        cameras = Cameras(
            fx=float(width),
            fy=float(width),
            cx=float(width) / 2,
            cy=float(height) / 2,
            height=int(height),
            width=int(width),
            distortion_params=distortion_params,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PANORAMA,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=img_filenames,
            cameras=cameras,
            scene_box=scene_box
        )
        return dataparser_outputs
