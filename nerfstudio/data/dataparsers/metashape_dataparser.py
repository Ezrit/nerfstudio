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

import random
import xml.etree.ElementTree as ET
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
        filenames: List of all filenames, to read the mask 
    """
    # nothing to do if threshold is 0 or less and weighting type is uniform -> dont read the image again! 
    if alpha_threshold <= 0.0 and weighting_type == WeightingType.UNIFORM:
        return

    # read the mask
    image_filename = filenames[image_idx]
    pil_alpha = Image.open(image_filename)
    alpha = np.array(pil_alpha, dtype="uint8")  # shape is (h, w)
    if len(alpha.shape) == 3:
        alpha = alpha[:, :, 0]
    alpha = alpha[..., np.newaxis]
    assert len(alpha.shape) == 3
    assert alpha.dtype == np.uint8
    alpha = torch.from_numpy(alpha.astype("float32") / 255.0)

    additional_data = {}
    # add "mask" to data and mask all pixels with alpha < threshold
    if alpha_threshold > 0.0:
        mask = torch.ones_like(alpha)
        mask[alpha < alpha_threshold] = 0
        additional_data["mask"] = mask

    # add "weight" to data for weighted MSE Loss depending on weighting type selected
    if weighting_type == WeightingType.ALPHA:
        additional_data["weight"] = alpha
        # additional_data["probability"] = alpha
    elif weighting_type == WeightingType.POLAR:
        weights = torch.sin((torch.arange(alpha.shape[0], dtype=torch.float32).view(-1, 1, 1) + 0.5) / alpha.shape[0] * pi)
        additional_data["weight"] = weights.repeat(1, alpha.shape[1], 1)
        # additional_data["probability"] = weights.repeat(1, alpha.shape[1], 1)
    elif weighting_type == WeightingType.ALPHA_POLAR:
        weights = torch.sin((torch.arange(alpha.shape[0], dtype=torch.float32).view(-1, 1, 1) + 0.5) / alpha.shape[0] * pi)
        additional_data["weight"] = weights.repeat(1, alpha.shape[1], 1) * alpha
        # additional_data["probability"] = weights.repeat(1, alpha.shape[1], 1) * alpha
    return additional_data


@dataclass
class MetashapeDataParserConfig(DataParserConfig):
    """Metashape dataset config"""

    _target: Type = field(default_factory=lambda: Metashape)
    """target class to instantiate"""
    data: Path = Path("Dataset/Metashape/0_Takadanobaba_ELM")
    """Directory specifying location of data."""
    direction: int = -1
    """specifies the direction to train. (-1 means there is none, 0 means all at once, others are the specified direction only"""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the scene."""
    orientation_method: Literal["pca", "up"] = "up"
    """The method to use for orientation."""
    weighting: WeightingType = CAMERA_MODEL_TO_TYPE["POLAR"]
    """Weighting for the MSE Loss for the images"""
    alpha_threshold: float = 0.6
    """Threshold for alpha masking. Alpha values below this will be masked completely (No rays cast)."""
    max_images: int = 200
    """Number of images to be used for training. Will randomly draw from all images available at the start. (it breaks with too many)"""


@dataclass
class Metashape(DataParser):
    """Metashape DatasetParser"""

    config: MetashapeDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        img_filenames = []
        msk_filenames = []
        poses = []
        sub_folder = ''
        split = 'train'

        change_axis = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        change_axis_rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        change_axis_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # TODO: care for None in case there is no chunk etc.... but actually the data should be just like that...
        tree = ET.parse(self.config.data / 'cameras.xml')
        root = tree.getroot()
        chunk = root.find('chunk')
        cameras = chunk.find('cameras')

        groups_exist = (cameras.find('group') is not None)
        if groups_exist:
            for group in cameras.findall('group'):
                if self.config.direction >= 0 and int(group.get('id')) != self.config.direction:
                    continue
                sub_folder = group.get('label')
                group_folder: Path = self.config.data / sub_folder / 'images' / split
                group_mask_folder: Path = self.config.data / sub_folder / 'masks' / split
                for camera in group.findall('camera'):
                    file_name = camera.get('label') + '.png'
                    img_filename = group_folder / file_name
                    if not img_filename.exists():
                        continue
                    img_filenames.append(group_folder / file_name)
                    msk_filenames.append(group_mask_folder / file_name)
                    pose = np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4))
                    pose[:3, :3] = (change_axis_rot[:3, :3] @ pose[:3, :3].transpose()).transpose()
                    pose[:, 3] = change_axis_pos @ pose[:, 3]
                    poses.append(change_axis @ pose)
        else:
            group_folder: Path = self.config.data / sub_folder / 'images' / split
            group_mask_folder: Path = self.config.data / sub_folder / 'masks' / split
            for camera in cameras.findall('camera'):
                file_name = camera.get('label') + '.png'
                img_filename = group_folder / file_name
                if not img_filename.exists():
                    continue
                img_filenames.append(group_folder / file_name)
                msk_filenames.append(group_mask_folder / file_name)
                pose = np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4))
                pose[:3, :3] = (change_axis_rot[:3, :3] @ pose[:3, :3].transpose()).transpose()
                pose[:, 3] = change_axis_pos @ pose[:, 3]
                poses.append(change_axis @ pose)

        sampled_indices = range(len(img_filenames))
        if self.config.max_images > 0 and len(img_filenames) > self.config.max_images:
            indices = range(len(img_filenames))
            sampled_indices = random.sample(indices, self.config.max_images)
            img_filenames = [img_filenames[i] for i in sampled_indices]
            msk_filenames = [msk_filenames[i] for i in sampled_indices]
            # poses = [poses[i] for i in sampled_indices] this needs to be done after auto_orient_poses, so the final world coordinate system doesnt change every time
            
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        self.transform = camera_utils.auto_orient_poses_transform(poses, method=self.config.orientation_method)
        poses = camera_utils.auto_orient_poses(poses, method=self.config.orientation_method)
        poses = poses[sampled_indices, ...]

        # Scale poses
        scale_factor = 1.0 / torch.max(torch.abs(poses[:, :3, 3]))
        poses[:, :3, 3] *= scale_factor * self.config.scale_factor

        combined_scale_factor = scale_factor * self.config.scale_factor
        self.transform = torch.diag(torch.tensor([combined_scale_factor, combined_scale_factor, combined_scale_factor, 1], device=self.transform.device)) @ self.transform

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        resolution_element = chunk.find('sensors').find('sensor').find('resolution')
        width = int(resolution_element.get('width'))
        height = int(resolution_element.get('height'))

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
        add_inputs["weights_and_mask"] = {"func": get_weights_and_masks, "kwargs": {"alpha_threshold": self.config.alpha_threshold, "weighting_type": self.config.weighting, "filenames": msk_filenames}}

        dataparser_outputs = DataparserOutputs(
            image_filenames=img_filenames,
            cameras=cameras,
            scene_box=scene_box,
            alpha_color=alpha_color_tensor,
            additional_inputs=add_inputs
        )
        return dataparser_outputs
