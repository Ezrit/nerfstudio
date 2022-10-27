# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for stella vslam data"""

from __future__ import annotations

from cmath import pi
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Literal, Type

import numpy as np
import torch
from PIL import Image
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


class WeightingType(Enum):
    """Supported Weighting types."""

    ALPHA = auto()
    POLAR = auto()
    ALPHA_POLAR = auto()
    UNIFORM = auto()


CAMERA_MODEL_TO_TYPE = {
    "ALPHA": WeightingType.ALPHA,
    "POLAR": WeightingType.POLAR,
    "ALPHA_POLAR": WeightingType.ALPHA_POLAR,
    "UNIFORM": WeightingType.UNIFORM,
}


def get_weights_and_masks(image_idx: int, alpha_threshold: float, weighting_type: WeightingType, filenames: List[Path]):
    """function to process additional weighting and mask information

    Args:
        image_idx: specific image index to work with
        alpha_threshold: masking threshold, every pixel with alpha lower than this will be masked
        weighting_type: weigthing type for images
        filenames: List of all filenames, to read the image again... :/
    """
    # nothing to do if threshold is 0 or less and weighting type is uniform -> dont read the image again! 
    if alpha_threshold <= 0.0 and weighting_type == WeightingType.UNIFORM.value:
        return

    # read the image... once again... not ideal
    # TODO: think whether/how to improve this. This was read in datasets.py already once.
    #       ofc I could add this functionality in datasets.py, but it seems to break the idea behind the structure
    image_filename = filenames[image_idx]
    pil_image = Image.open(image_filename)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    image = torch.from_numpy(image.astype("float32") / 255.0)

    # get alpha values or ones if no alpha channel is present
    if image.shape[-1] == 4:
        alpha = image[:, :, 3:]
    else:
        alpha = torch.ones_like(image[:, :, 0:1])

    additional_data = {}
    # add "mask" to data and mask all pixels with alpha < threshold
    if alpha_threshold > 0.0:
        mask = torch.ones_like(alpha)
        mask[alpha < alpha_threshold] = 0
        additional_data["mask"] = mask
    
    # add "weight" to data for weighted MSE Loss depending on weighting type selected
    if weighting_type == WeightingType.ALPHA.value:
        additional_data["weight"] = alpha
    elif weighting_type == WeightingType.POLAR.value:
        weights = torch.sin(torch.arange(alpha.shape[0], dtype=torch.float32).view(-1, 1, 1) / alpha.shape[0] * pi)
        additional_data["weights"] = weights.repeat(1, alpha.shape[1], 1)
    elif weighting_type == WeightingType.ALPHA_POLAR.value:
        weights = torch.sin(torch.arange(alpha.shape[0], dtype=torch.float32).view(-1, 1, 1) / alpha.shape[0] * pi)
        additional_data["weights"] = weights.repeat(1, alpha.shape[1], 1) * alpha
    return additional_data


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
    weighting: WeightingType = CAMERA_MODEL_TO_TYPE["POLAR"]
    """Weighting for the MSE Loss for the images"""
    alpha_threshold: float = 0.6
    """Threshold for alpha masking. Alpha values below this will be masked completely (No rays cast)."""


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

                rotationQuat_w2c = Quaternion(float(qw), float(qx), float(qy), float(qz))

                change_axis = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
                rot_matrix_c2w = rotationQuat_w2c.rotation_matrix.T

                trans_cw = np.array([float(tx), -float(ty), float(tz)], dtype=np.float32)

                pos = -rot_matrix_c2w @ trans_cw

                pos = np.expand_dims(pos, axis=1)

                img_filenames.append(self.config.data / 'images' / img_name)
                pose = np.append(rot_matrix_c2w, pos, axis=1).astype(np.float32)
                pose = np.append(change_axis @ pose, np.array([[0, 0, 0, 1]], dtype=np.float32), axis=0)
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

        alpha_color_tensor = get_color("green")

        add_inputs = {}
        add_inputs["weights_and_mask"] = {"func": get_weights_and_masks, "kwargs": {"alpha_threshold": self.config.alpha_threshold, "weighting_type": self.config.weighting, "filenames": img_filenames}}

        dataparser_outputs = DataparserOutputs(
            image_filenames=img_filenames,
            cameras=cameras,
            scene_box=scene_box,
            alpha_color=alpha_color_tensor,
            additional_inputs=add_inputs
        )
        return dataparser_outputs
