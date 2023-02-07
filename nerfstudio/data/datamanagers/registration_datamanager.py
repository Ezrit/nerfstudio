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

"""
Depth datamanager.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Parameter
from torchtyping import TensorType

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes


@dataclass
class RegistrationDataManagerConfig(InstantiateConfig):
    """A registration datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: RegistrationDataManager)
    """Target class to instantiate."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = 200
    """Number of images to sample during training iteration."""
    train_image_size: Tuple[int, int] = (960, 1920)
    """Image size to sample from."""


class RegistrationDataManager(base_datamanager.DataManager):  # pylint: disable=abstract-method
    """Data manager implementation for feeding positions and directions for optimizing for transformation
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RegistrationDataManagerConfig
    pixel_sampler: Optional[EquirectangularPixelSampler] = None
    training_callbacks: List[TrainingCallback] = []
    position_sampler: Optional[MultivariateNormal] = None

    def update_position_sampler(self, nerf_centers: TensorType) -> None:
        pass

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        assert self.position_sampler is not None, "position sample is not initialized!"

        sampled_positions = self.position_sampler.rsample(torch.Size((self.config.train_num_images_to_sample_from, )))

        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        return self.next_train(step)

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        """Returns the next eval image."""
        raise NotImplementedError

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
    pass
