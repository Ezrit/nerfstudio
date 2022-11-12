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
Code for sampling pixels.
"""

import random
from typing import Dict

import torch


def collate_image_dataset_batch(batch: Dict, num_rays_per_batch: int, keep_full_image: bool = False):
    """
    Operates on a batch of images and samples pixels to use for generating rays.
    Returns a collated batch which is input to the Graph.
    It will sample only within the valid 'mask' if it's specified.

    Args:
        batch: batch of images to sample from
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """
    device = batch["image"].device
    num_images, image_height, image_width, _ = batch["image"].shape

    # sample according to the given probabilities if probabilities are in the batch
    if "probability" in batch:
        mask = batch["mask"][..., 0] if "mask" in batch else torch.ones_like(batch["image"][..., 0])
        probability = batch["probability"][..., 0]
        nonzero_indices = torch.nonzero(mask, as_tuple=False)
        masked_probability = probability[nonzero_indices.chunk(chunks=3, dim=1)]

        # TODO: think about a workaround... as the number of categories (masked_probabilitz[:, 0] cannot exceed 2^24)
        chosen_indices = torch.multinomial(masked_probability[:, 0], num_samples=num_rays_per_batch, replacement=False)
        indices = nonzero_indices[chosen_indices]
    # only sample within the mask, if the mask is in the batch
    elif "mask" in batch:
        nonzero_indices = torch.nonzero(batch["mask"][..., 0], as_tuple=False)
        chosen_indices = random.sample(range(len(nonzero_indices)), k=num_rays_per_batch)
        indices = nonzero_indices[chosen_indices]
    else:
        indices = torch.floor(
            torch.rand((num_rays_per_batch, 3), device=device)
            * torch.tensor([num_images, image_height, image_width], device=device)
        ).long()

    c, y, x = (i.flatten() for i in torch.split(indices, 1, dim=-1))
    image = batch["image"][c, y, x]
    weight, mask, semantics_stuff, semantics_thing = None, None, None, None
    if "mask" in batch:
        mask = batch["mask"][c, y, x]
    if "semantics_stuff" in batch:
        semantics_stuff = batch["semantics_stuff"][c, y, x]
    if "semantics_thing" in batch:
        semantics_thing = batch["semantics_thing"][c, y, x]
    if "weight" in batch:
        weight = batch["weight"][c, y, x]
    assert image.shape == (num_rays_per_batch, 3), image.shape

    # Needed to correct the random indices to their actual camera idx locations.
    local_indices = indices.clone()
    indices[:, 0] = batch["image_idx"][c]
    collated_batch = {
        "local_indices": local_indices,  # local to the batch returned
        "indices": indices,  # with the abs camera indices
        "image": image,
    }
    if mask is not None:
        collated_batch["mask"] = mask
    if semantics_stuff is not None:
        collated_batch["semantics_stuff"] = semantics_stuff
    if semantics_thing is not None:
        collated_batch["semantics_thing"] = semantics_thing
    if weight is not None:
        collated_batch["weight"] = weight

    if keep_full_image:
        collated_batch["full_image"] = batch["image"]

    return collated_batch


class PixelSampler:  # pylint: disable=too-few-public-methods
    """Samples 'pixel_batch's from 'image_batch's.

    Args:
        num_rays_per_batch: number of rays to sample per batch
        keep_full_image: whether or not to include a reference to the full image in returned batch
    """

    def __init__(self, num_rays_per_batch: int, keep_full_image: bool = False) -> None:
        self.num_rays_per_batch = num_rays_per_batch
        self.keep_full_image = keep_full_image

    def set_num_rays_per_batch(self, num_rays_per_batch: int):
        """Set the number of rays to sample per batch.

        Args:
            num_rays_per_batch: number of rays to sample per batch
        """
        self.num_rays_per_batch = num_rays_per_batch

    def sample(self, image_batch: Dict):
        """Sample an image batch and return a pixel batch.

        Args:
            image_batch: batch of images to sample from
        """
        pixel_batch = collate_image_dataset_batch(
            image_batch, self.num_rays_per_batch, keep_full_image=self.keep_full_image
        )
        return pixel_batch
