"""Data parser for stella vslam data"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Type

import numpy as np
import torch
from pyquaternion import Quaternion
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color

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
                rot_matrix_c2w = rotationQuat_w2c.rotation_matrix.T

                pos = -rot_matrix_c2w @ trans_cw

                pos = np.expand_dims(pos, axis=1)

                # TODO: unsure why this is... but otherwise it is the wrong direction. I think it ist because of the y/z axis exchange...
                pos[1] = -pos[1]

                img_filenames.append(self.config.data / 'images' / img_name)
                pose = np.append(rot_matrix_c2w, pos, axis=1).astype(np.float32)
                pose = np.append(pose, np.array([[0, 0, 0, 1]], dtype=np.float32), axis=0)
                poses.append(pose)

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

        with open(self.config.data / "cameras.txt") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                cam_id, cam_model, width, height = line.split()[:4]
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
            fx=float(width) / 2,
            fy=float(width) / 2,
            cx=float(width) / 2,
            cy=float(height) / 2,
            height=int(height),
            width=int(width),
            distortion_params=distortion_params,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.PANORAMA,
        )

        alpha_color_tensor = get_color("white")

        dataparser_outputs = DataparserOutputs(
            image_filenames=img_filenames,
            cameras=cameras,
            scene_box=scene_box,
            alpha_color=alpha_color_tensor,
        )
        return dataparser_outputs
