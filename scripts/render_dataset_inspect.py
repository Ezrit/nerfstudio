#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from torchtyping import TensorType

import nerfstudio.utils.poses as pose_utils
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _render_dataset_images(
    pipeline: Pipeline,
    cameras: Cameras,
    rendered_output_names: List[str],
    output_image_dir: Path,
    number_images: int,
) -> None:
    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    cameras = cameras.to(pipeline.device)
    transforms_dict = {
        "fl_x": cameras.fx[0, 0].item(),
        "fl_y": cameras.fy[0, 0].item(),
        "cx": cameras.cx[0, 0].item(),
        "cy": cameras.cy[0, 0].item(),
        "w": cameras.width[0, 0].item(),
        "h": cameras.height[0, 0].item(),
    }
    frames_list: List[dict] = []
    with progress:
        for camera_idx in progress.track(range(number_images), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            render_image = []
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(
                        f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center"
                    )
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                if output_image.shape[-1] == 1:
                    output_image = np.concatenate((output_image,) * 3, axis=-1)
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)
            output_file = output_image_dir / f"{camera_idx:05d}.png"
            media.write_image(output_file, render_image)
            image_transform_matrix = torch.eye(4)
            image_transform_matrix[:3, :4] = cameras.camera_to_worlds[camera_idx, :3, :4]
            frame_dict = {
                "file_path": output_file.name, 
                "transform_matrix": image_transform_matrix.tolist(),
            }
            frames_list.append(frame_dict)

    transforms_dict["frames"] = frames_list
    json_object = json.dumps(transforms_dict, indent=4)
    json_file = output_image_dir / 'transforms.json'
    with open(json_file.as_posix(), 'w') as f:
        f.write(json_object)
    


@dataclass
class RenderDataset:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    # Name of the dataset path.
    dataset_path: Path = Path('./dataset/')
    # Name of the output file path, will be appended to dataset_path
    output_path: Path = Path('dump')
    # resolution of images
    resolution: Tuple[int, int] = (256, 256)
    # scene divisions for rendering
    scene_divisions: Tuple[int, int, int] = (8, 8, 8)
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
        )

        modelBox: SceneBox = pipeline.model.scene_box

        number_unique_positions = self.scene_divisions[0] * self.scene_divisions[1] * self.scene_divisions[2]
        number_unique_views = 6
        number_images = number_unique_positions * number_unique_views

        x_div = torch.linspace(modelBox.aabb[0][0], modelBox.aabb[1][0], steps=self.scene_divisions[0], dtype=torch.float)
        y_div = torch.linspace(modelBox.aabb[0][1], modelBox.aabb[1][1], steps=self.scene_divisions[1], dtype=torch.float)
        z_div = torch.linspace(modelBox.aabb[0][2], modelBox.aabb[1][2], steps=self.scene_divisions[2], dtype=torch.float)
        positions_grid = torch.stack(torch.meshgrid(x_div, y_div, z_div, indexing='ij'), dim=-1)
        unique_positions = torch.flatten(positions_grid, end_dim=2)
        positions: TensorType['number_views', 3, float] = unique_positions.repeat(number_unique_views, 1)

        unique_directions: TensorType[8, 3, float] = torch.tensor([[1, 0, 0], [-1, 0, 0],
                                                                   [0, 1, 0], [0, -1, 0],
                                                                   [0, 0, 1], [0, 0, -1]], dtype=torch.float)
        directions: TensorType['number_views', 3, float] = unique_directions.repeat_interleave(number_unique_positions, dim=0)
        up: TensorType[3] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)

        list_c2whs = []
        for i in range(number_images):
            lookat = directions[i]
            center = positions[i]
            current_up = up
            if torch.linalg.norm(current_up - lookat) < 0.01:
                current_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float)
            elif torch.linalg.norm(current_up + lookat) < 0.01:
                current_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float)
            c2w = camera_utils.viewmatrix(lookat, current_up, center)
            c2wh = pose_utils.to4x4(c2w)
            list_c2whs.append(c2wh)
        c2whs = torch.stack(list_c2whs, dim=0)

        fl_x: float = 110.0
        fl_y: float = 110.0
        cameras: Cameras = Cameras(camera_to_worlds=c2whs[:, :3, :4], fx=fl_x, fy=fl_y, cx=self.resolution[1]/2.0, cy=self.resolution[0]/2.0, width=self.resolution[1], height=self.resolution[0])
        combined_output_path: Path = self.dataset_path / self.output_path
        combined_output_path.mkdir(parents=True)
        _render_dataset_images(pipeline=pipeline, cameras=cameras, rendered_output_names=self.rendered_output_names, output_image_dir=combined_output_path, number_images=number_images)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderDataset).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
