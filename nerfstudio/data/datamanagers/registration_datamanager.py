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
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.datamanagers import base_datamanager
from nerfstudio.data.dataparsers.no_dataparser import NoDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator


def covariance_matrix(eigenvectors: torch.Tensor, eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Computes the covariance matrix from eigenvectors and eigenvalues
    Args:
    - eigenvectors: a tensor of shape (num_features, num_features) representing the eigenvectors
    - eigenvalues: a tensor of shape (num_features,) representing the eigenvalues
    Returns:
    - A tensor of shape (num_features, num_features) representing the covariance matrix
    """
    # check input shapes
    num_features = eigenvectors.shape[0]
    assert eigenvectors.shape == (num_features, num_features), "eigenvectors should be of shape (num_features, num_features)"
    assert eigenvalues.shape == (num_features,), "eigenvalues should be of shape (num_features,)"
    assert torch.abs(eigenvectors[0,:].dot(eigenvectors[1,:])) < 0.01, "not orthogonal basis"

    # normalize the eivenvectors and then scale according to the square roots of the eigenvectors 
    # see https://en.wikipedia.org/wiki/Covariance_matrix
    norm_eigenvectors = eigenvectors / torch.linalg.vector_norm(eigenvectors, ord=2, dim=1)
    scaled_eigenvectors = eigenvalues.sqrt() * norm_eigenvectors

    # construct diagonal matrix from eigenvalues
    eigenvalues_matrix = torch.diag(eigenvalues)

    # compute covariance matrix using eigenvectors and eigenvalues
    covariance = scaled_eigenvectors @ eigenvalues_matrix @ scaled_eigenvectors.t()

    return covariance


def get_multivariate_normal_3d(centers: torch.Tensor) -> torch.distributions.multivariate_normal.MultivariateNormal:
    """ 
    Computes a multivariate normal from 2 NeRF centers.
    The sampling should have the highes chance at the mean of the centers.
    The covariance depends on the distance between the centers.
    Args:
    - centers: a tensor of shape (2, 3) representing the centers of the NeRFs
    Returns:
    - a MultivariateNormal for sampling positions accordingly.
    """
    assert centers.shape == (2, 3), "eigenvectors should be of shape (num_features, num_features)"

    mean = centers.mean(dim=0)

    direction = centers[0, :] - centers[1, :]
    direction_length = torch.linalg.vector_norm(direction, ord=2)
    direction_norm = direction / direction_length

    up_direction = torch.tensor([0, 0, 1], dtype=direction_norm.dtype, device=direction_norm.device)
    ortho_direction = torch.cross(up_direction, direction_norm)
    ortho_up = torch.cross(direction_norm, ortho_direction)

    eigenvectors = torch.stack((ortho_direction, direction_norm, ortho_up), dim=0)
    eigenvalues = torch.tensor([torch.sqrt(direction_length/2)/3/2, torch.sqrt(direction_length/2)/3, 0.01], dtype=torch.float, device=direction_norm.device)

    cov = covariance_matrix(eigenvectors, eigenvalues)

    return torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)


@dataclass
class RegistrationDataManagerConfig(base_datamanager.VanillaDataManagerConfig):
    """A registration datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: RegistrationDataManager)
    """Target class to instantiate."""
    train_num_rays_per_batch: int = 1024
    """Number of rays per batch to use per training iteration."""
    train_num_images_to_sample_from: int = 200
    """Number of images to sample during training iteration."""
    train_num_images_to_sample: int = 200
    """Number of images to sample during training iteration."""
    train_image_size: Tuple[int, int] = (960, 1920)
    """Image size to sample from."""
    dataparser: NoDataParserConfig = NoDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    # eval_num_rays_per_batch: None = None
    # eval_num_images_to_sample_from: None = None
    # eval_num_times_to_repeat_images: None = None
    # eval_image_indices: None = None
    # camera_res_scale_factor: None = None
    # train_num_times_to_repeat_images: None = None


class RegistrationDataManager(base_datamanager.VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for feeding positions and directions for optimizing for transformation
    Args:
        config: the DataManagerConfig used to instantiate class
    """

    pixel_sampler: Optional[EquirectangularPixelSampler] = None
    train_dataset: InputDataset
    training_callbacks: List[TrainingCallback] = []
    position_sampler: Optional[MultivariateNormal] = None

    def __init__(
        self,
        config: RegistrationDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(config)
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.test_mode = test_mode
        dummy_idx = torch.tensor(range(self.config.train_num_images_to_sample_from), dtype=torch.int)
        dummy_images = torch.zeros((self.config.train_num_images_to_sample_from, self.config.train_image_size[0], self.config.train_image_size[1], 3), dtype=torch.float)
        self.image_batch = {}
        self.image_batch['image_idx'] = dummy_idx
        self.image_batch['image'] = dummy_images
        self.setup_train()

    def update_ray_generator(self, nerf_centers: TensorType[2, 3], step: int) -> None:
        assert self.ray_generator is not None
        # create new c2w matrizes without the translation for now
        # rotation should not be needed as we use equirectangular cameras atm
        # new_camera_to_worlds = torch.eye(3, dtype=torch.float, device=self.device).reshape((1, 3, 3))
        new_camera_to_worlds = torch.index_select(torch.eye(3, dtype=torch.float, device=self.device), 1, torch.tensor([0, 2, 1], device=self.device)).reshape((1, 3, 3))
        new_camera_to_worlds = new_camera_to_worlds.repeat(self.config.train_num_images_to_sample_from, 1, 1)

        # generate new translations
        self.position_sampler = get_multivariate_normal_3d(nerf_centers)
        new_translations = self.position_sampler.rsample(torch.Size((self.config.train_num_images_to_sample_from, )))
        new_translations_reshaped = new_translations.reshape((-1, 3, 1)).to(self.device)

        # stack the translations (NOT! negative cause its camera -> world, not the other way)
        new_camera_to_worlds = torch.cat((new_camera_to_worlds, new_translations_reshaped), dim=2)

        # update the cameras
        # need to generate a new generator i think, cause of the image_coords... well or just update them...
        self.ray_generator.cameras.camera_to_worlds = new_camera_to_worlds
        self.ray_generator.image_coords = nn.Parameter(self.ray_generator.cameras.get_image_coords(), requires_grad=False)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training.

        Here you will define any subclass specific object attributes from the attribute"""

        # setup the position sampler, start with no correlation between the axes (diag matrix as covariance matrix)
        starting_mean = torch.tensor([0, 0, 0], dtype=torch.float, device=self.device)
        # calculate starting deviation for x and z -> divide by 3 should enfore 99.9% to be within the given size (half world)
        scatter_x_z = float(self.world_size) / 2.0 / 3.0
        # starting deviation for y just fixed to 0.1, so we dont randomly sample from within the ground
        scatter_y = 0.1
        starting_covariance = torch.tensor([scatter_x_z, scatter_y, scatter_x_z], dtype=torch.float, device=self.device)
        self.position_sampler = MultivariateNormal(starting_mean, covariance_matrix=torch.diag(starting_covariance))

        # setup initial ray_generator
        # create c2w matrizes without the translation for now
        # rotation should not be needed as we use equirectangular cameras atm
        camera_to_worlds = torch.index_select(torch.eye(3, dtype=torch.float, device=self.device), 1, torch.tensor([0, 2, 1], device=self.device)).reshape((1, 3, 3))
        camera_to_worlds = camera_to_worlds.repeat(self.config.train_num_images_to_sample_from, 1, 1)

        # generate new translations
        translations = self.position_sampler.rsample(torch.Size((self.config.train_num_images_to_sample_from, ))).reshape((-1, 3, 1)).to(self.device)

        # stack the translations (NOT! negative cause its camera -> world, not the other way)
        camera_to_worlds = torch.cat((camera_to_worlds, translations), dim=2)

        # create cameras, no distortion since equirectangular
        height, width = self.config.train_image_size
        self.cameras = Cameras(
            fx=float(width) / 2,
            fy=float(width) / 2,
            cx=float(width) / 2,
            cy=float(height) / 2,
            height=int(height),
            width=int(width),
            distortion_params=camera_utils.get_distortion_params(k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0),
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.EQUIRECTANGULAR,
        )

        # setup camera optimizer
        # the optimizer can be turned off in the config!
        self.camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.config.train_num_images_to_sample_from, device=self.device
        )

        # setup the ray generator
        self.ray_generator = RayGenerator(
            self.cameras.to(self.device),
            self.camera_optimizer,
        )

        self.pixel_sampler = EquirectangularPixelSampler(self.config.train_num_rays_per_batch)

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation"""

        return self.setup_train()

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        """Returns the next batch of data from the train data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        assert self.position_sampler is not None, "position sample is not initialized!"

        # image_batch = {}
        # image_batch['image_idx'] = torch.randint(self.config.train_num_images_to_sample_from, (self.config.train_num_images_to_sample,))
        # image_batch['image'] = self.dummy_images
        assert self.pixel_sampler is not None, "pixel sampler is None"
        batch = self.pixel_sampler.sample(self.image_batch)
        assert self.ray_generator is not None, "ray generator is None"
        ray_indices = batch["indices"]
        ray_bundle = self.ray_generator(ray_indices)

        return ray_bundle, batch

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        """Returns the next batch of data from the eval data manager.

        This will be a tuple of all the information that this data manager outputs.
        """
        return self.next_train(step)

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        """Returns the next eval image."""
        # raise NotImplementedError
        rand_idx = torch.randint(self.config.train_num_images_to_sample_from, (1,))
        ray_bundle = self.ray_generator.cameras.generate_rays(camera_indices=int(rand_idx[0]))
        assert ray_bundle.camera_indices is not None
        image_idx = int(ray_bundle.camera_indices[0, 0, 0])
        batch = {'image_idx': image_idx}
        return image_idx, ray_bundle, batch

    def get_training_callbacks(  # pylint:disable=no-self-use
        self, training_callback_attributes: TrainingCallbackAttributes  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        """Returns a list of callbacks to be used during training."""
        return self.training_callbacks

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.

        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        return {}
    pass
