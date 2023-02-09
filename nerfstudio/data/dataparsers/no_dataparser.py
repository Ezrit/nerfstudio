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


@dataclass
class NoDataParserConfig(DataParserConfig):
    """Metashape dataset config"""

    _target: Type = field(default_factory=lambda: NoDataParser)
    """target class to instantiate"""

@dataclass
class NoDataParser(DataParser):
    """Metashape DatasetParser"""

    config: NoDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        poses = torch.eye(n=3, m=4, dtype=torch.float).view(1, 3, 4)
        cameras = Cameras(
            fx=float(0),
            fy=float(0),
            cx=float(0),
            cy=float(0),
            height=int(0),
            width=int(0),
            distortion_params=camera_utils.get_distortion_params(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            camera_to_worlds=poses[:, :3, :4],
            camera_type=CameraType.EQUIRECTANGULAR,
        )
        dataparser_outputs = DataparserOutputs(
            image_filenames=[],
            cameras=cameras,
        )
        return dataparser_outputs
