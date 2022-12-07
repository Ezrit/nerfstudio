#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Literal, Optional

import mediapy as media
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
from typing_extensions import assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.data.dataparsers.stellav_dataparser import StellaVSlamDataParserConfig
from nerfstudio.data.utils.datasets import InputDataset
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_name: Name of the renderer output to use.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Number for the output video.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx).to(pipeline.device)
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if rendered_output_name not in outputs:
                CONSOLE.rule("Error", style="red")
                CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                sys.exit(1)
            image = outputs[rendered_output_name].cpu().numpy()
            images.append(image)

    fps = len(images) / seconds
    # make the folder if it doesn't exist
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
        media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


@dataclasses.dataclass
class DualRenderTrajectory:
    """Load 2 checkpoints, render the same trajectory from both, interpolate frames according to interpolation method and save to a video file."""
    
    # Path to config main YAML file - main refers to the NeRF in whichs coordinate system the path is given.
    main_load_config: Path
    # Path to second config YAML file.
    sub_load_config: Path
    # Name of the renderer output to use. rgb, depth, etc.
    rendered_output_name: str = "rgb"
    # Trajectory to render. Currently only filename trajectory allowed!
    traj: Literal["filename"] = "filename"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None
    # Specifies whether the not inteprolated videos should be saved as well
    save_single_videos: bool = False
    # Specifies whether the interpolated frames should be saved
    save_interpolated_frames: bool = False
    # Specifies whether the not interpolated frames should be saved
    save_single_frames: bool = False

    def main(self) -> None:
        """Main function."""
        main_config, main_pipeline, _ = eval_setup(
            self.main_load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
        )

        main_dataparser = main_config.pipeline.datamanager.dataparser.setup()
        _ = main_dataparser.get_dataparser_outputs(split="train")

        install_checks.check_ffmpeg_installed()
        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args

        if self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        # Render path in main NeRF
        main_output_path = Path(self.output_path.parent) / 'main_{name}'.format(name=self.output_path.name)
        _render_trajectory_video(
            main_pipeline,
            camera_path,
            output_filename=main_output_path,
            rendered_output_name=self.rendered_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
        )

        # load the Dataset from main NeRF to get the transformation matrix
        # TODO
        main_transform = main_dataparser.transform

        del main_config, main_pipeline

        # load sub config & pipeline
        sub_config, sub_pipeline, _ = eval_setup(
            self.sub_load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
        )
        sub_dataparser = sub_config.pipeline.datamanager.dataparser.setup()
        _ = sub_dataparser.get_dataparser_outputs(split="train")

        # load the dataset from the seub NeRF to get their transformation matrix
        # TODO
        sub_transform = sub_dataparser.transform

        # transform the camera_path from the main coordinate system to the sub coordinate system
        # first multiply the c2w matrizes with the inverse of the main_transform to get them in stellav coordinates
        # then multiply this with the sub_transform to get them in sub-NeRF coordinates

        # extract the cameras c2w from main expand them to 'cameras'x4x4  
        main_cameras_c2w = torch.cat((camera_path.camera_to_worlds, torch.tensor([0, 0, 0, 1]).view(1, 1, 4).expand(camera_path._num_cameras, -1, -1)), 1)
        # multiply transform matrizes accordingly to get cameras c2w for sub
        sub_cameras_c2w = torch.matmul(sub_transform, torch.matmul(torch.inverse(main_transform), main_cameras_c2w))
        #

        camera_path.camera_to_worlds = sub_cameras_c2w[:, :3, :]

        # Render path in main NeRF
        sub_output_path = Path(self.output_path.parent) / 'sub_{name}'.format(name=self.output_path.name)
        _render_trajectory_video(
            sub_pipeline,
            camera_path,
            output_filename=sub_output_path,
            rendered_output_name=self.rendered_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
        )
        pass


@dataclasses.dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer output to use. rgb, depth, etc.
    rendered_output_name: str = "rgb"
    #  Trajectory to render.
    traj: Literal["spiral", "interp", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "interp":
            # cameras_a = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
            # cameras_b = pipeline.datamanager.eval_dataloader.get_camera(image_idx=10)
            # camera_path = get_interpolated_camera_path(cameras, steps=30)
            raise NotImplementedError("Interpolated camera path not implemented.")
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_name=self.rendered_output_name,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(DualRenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(DualRenderTrajectory)  # noqa
