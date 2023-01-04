import dataclasses
import datetime
import pathlib
import sys
import typing
import xml.etree.ElementTree as ET

import mediapy as media
import numpy as np
import scipy
import scipy.interpolate
import scipy.spatial.transform
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

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def render_trajectory_frames(
    pipeline: Pipeline,
    cameras: Cameras,
    rendered_output_name: str,
    rendered_resolution_scaling_factor: float = 1.0,
) -> typing.List[np.ndarray]:
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
    return images


def save_video(images: typing.List[np.ndarray], output_filename: pathlib.Path, seconds: float = 10.0) -> None:
    fps = len(images) / seconds
    # make the folder if it doesn't exist
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
        media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


def render_cameras() -> None:
    pass


def get_nerf_transform(load_config_path: pathlib.Path) -> typing.Tuple[Pipeline, torch.Tensor]:
    config, pipeline, _ = eval_setup(
        load_config_path,
        None,
    )
    dataparser = config.pipeline.datamanager.dataparser.setup()
    _ = dataparser.get_dataparser_outputs(split="train")

    # load the dataset from the seub NeRF to get their transformation matrix
    transform = dataparser.transform

    return pipeline, transform


def read_transforms(dataset_folder: pathlib.Path, start_direction: int, start_frame: int, target_direction: int, target_frame: int) -> typing.Tuple[np.ndarray, np.ndarray]:
    tree = ET.parse(dataset_folder / 'cameras.xml')
    root = tree.getroot()
    chunk = root.find('chunk')
    cameras = chunk.find('cameras')

    start_transform: np.ndarray = np.array(())
    target_transform: np.ndarray = np.array(())

    change_axis_rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    change_axis_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    groups_exist = (cameras.find('group') is not None)
    if groups_exist:
        for group in cameras.findall('group'):
            if int(group.get('id')) == start_direction:
                for camera in group.findall('camera'):
                    if int(camera.get('label')) == start_frame:
                        start_transform = np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4))
                        start_transform[:3, :3] = (change_axis_rot[:3, :3] @ start_transform[:3, :3].transpose()).transpose()
                        start_transform[:, 3] = change_axis_pos @ start_transform[:, 3]
                        break
            if int(group.get('id')) == target_direction:
                for camera in group.findall('camera'):
                    if int(camera.get('label')) == start_frame:
                        target_transform = np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4))
                        target_transform[:3, :3] = (change_axis_rot[:3, :3] @ target_transform[:3, :3].transpose()).transpose()
                        target_transform[:, 3] = change_axis_pos @ target_transform[:, 3]
                        break
    else:
        return (np.array(()), np.array(()))
    return (start_transform, target_transform)


# return an numpy array with the interpolated transforms. For the interpolation a middle point is generated with start x and end z (or vice versa depending on direction).
# y for the middle point is just the mean of both
def get_interpolation_transforms(start_transform: torch.Tensor, end_transform: torch.Tensor, num_interpolation: int = 500, direction: int = 0) -> torch.Tensor:

    # first interpolate the positions via spline...
    start_point = np.array(start_transform[:3, 3])
    end_point = np.array(end_transform[:3, 3])

    middle_point = np.array((start_point[0], (start_point[1] + end_point[1]) / 2.0, end_point[2]))
    if direction > 0:
        middle_point[0] = end_point[0]
        middle_point[2] = start_point[2]
    key_times = np.array((0, 1, 2))
    key_positions = np.array((start_point, middle_point, end_point))

    spline = scipy.interpolate.CubicSpline(key_times, key_positions)

    # then interpolate the rotation via slerp (linear...)
    key_rotations = scipy.spatial.transform.Rotation.from_matrix([np.array(start_transform[:3, :3]), np.array(end_transform[:3, :3])])
    key_times = [0, 2]

    slerp = scipy.spatial.transform.Slerp(key_times, key_rotations)

    interpolation_times = np.linspace(0, 2, num_interpolation, dtype=np.float32)
    interpolated_positions = spline(interpolation_times)
    interpolated_rotations = slerp(interpolation_times).as_matrix()
    interpolated_transforms = np.append(interpolated_rotations, interpolated_positions.reshape(-1, 3, 1), axis=2)

    append_bottom = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1,1,4).repeat(num_interpolation, axis=0)

    camera_to_worlds = torch.from_numpy(np.append(interpolated_transforms, append_bottom, axis=1).astype(np.float32))

    return camera_to_worlds


def read_transforms_from_xml(xml_file: pathlib.Path, direction_group: int, frame_numbers: typing.Tuple[int, int]) -> typing.Dict[int, np.ndarray]:
    transform_dict = {}

    if frame_numbers[0] > frame_numbers[1]:
        frame_numbers = (frame_numbers[1], frame_numbers[0])

    tree = ET.parse(xml_file)
    root = tree.getroot()
    chunk = root.find('chunk')
    cameras = chunk.find('cameras')

    change_axis_rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    change_axis_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    groups_exist = (cameras.find('group') is not None)
    if groups_exist:
        for group in cameras.findall('group'):
            if int(group.get('id')) != direction_group:
                continue
            for camera in group.findall('camera'):
                frame = int(camera.get('label'))
                if frame >= frame_numbers[0] and frame < frame_numbers[1]:
                    transform = np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4))
                    transform[:3, :3] = (change_axis_rot[:3, :3] @ transform[:3, :3].transpose()).transpose()
                    transform[:, 3] = change_axis_pos @ transform[:, 3]
                    transform_dict[frame] = transform
    return transform_dict


def get_cameras(resolution: typing.Tuple[int, int], camera_to_worlds: torch.Tensor) -> Cameras:
    return Cameras(
        fx=resolution[1] / 2,
        fy=resolution[1] / 2,
        cx=resolution[1] / 2,
        cy=resolution[0] / 2,
        height=resolution[0],
        width=resolution[1],
        camera_to_worlds=camera_to_worlds,
        camera_type=CameraType.PANORAMA,
    )


def dict_to_tensor(array_dict: typing.Dict[int, np.ndarray], dict_key_range: typing.Tuple[int, int]) -> torch.Tensor:
    array_list: typing.List[np.ndarray] = []

    for i in range(dict_key_range[0], dict_key_range[1]):
        array_list.append(array_dict[i])

    return torch.from_numpy(np.array(array_list).astype(np.float32))


@dataclasses.dataclass
class IntersectionTransitionArguments:
    """Intersection Transition Arguments"""

    dataset_folder: pathlib.Path
    """Directory specifying location of data."""
    start_load_config: pathlib.Path
    """Path to config main YAML file - main refers to the NeRF in whichs coordinate system the path is given."""
    target_load_config: pathlib.Path
    """Path to second config YAML file."""
    start_direction: int = 0
    """specifies the starting direction for the transition video"""
    start_transistion_frame_numbers: typing.Tuple[int, int] = (0, 100)
    """specifies the frames of starting direction for the transition video. Hereby the weighting between original frames will decrease (to 0 by the second framenumber)"""
    target_direction: int = 1
    """specifies the target direction for the transition video"""
    target_transistion_frame_numbers: typing.Tuple[int, int] = (100, 200)
    """specifies the frames of target direction for the transition video. Hereby the weighting between original frames will increase (to 1 by the second framenumber)"""
    camera_type: typing.Literal[CameraType.PERSPECTIVE, CameraType.PANORAMA] = CameraType.PANORAMA
    """Specifies camera type for output frames"""
    output_folder: pathlib.Path = pathlib.Path('Transition')
    """Output directory (relative to dataset_folder)"""
    image_resolution: typing.Tuple[int, int] = (600, 1200)
    """Resolution of output frames (height, width)"""
    interpolate_direction: int = 0
    """Determines the corner point for the inteprolation"""


# TODO: define an area where we interpolate between original frames and network generated frames
if __name__ == '__main__':
    args = tyro.cli(IntersectionTransitionArguments)

    now = datetime.datetime.now()

    # first get camera transforms in genreal 'world' coordinates (metashape coordinates)
    start_transforms_dict: typing.Dict[int, np.ndarray] = read_transforms_from_xml(args.dataset_folder / 'cameras.xml', args.start_direction, args.start_transistion_frame_numbers)
    target_transforms_dict: typing.Dict[int, np.ndarray] = read_transforms_from_xml(args.dataset_folder / 'cameras.xml', args.target_direction, args.target_transistion_frame_numbers)

    start_transforms: torch.Tensor = dict_to_tensor(start_transforms_dict, args.start_transistion_frame_numbers)
    target_transforms: torch.Tensor = dict_to_tensor(target_transforms_dict, args.target_transistion_frame_numbers)
    between_transforms: torch.Tensor = get_interpolation_transforms(start_transforms[-1, ...], target_transforms[0, ...], 500, args.interpolate_direction)
    print(start_transforms.shape, between_transforms.shape)

    all_transforms_world = torch.cat((start_transforms, between_transforms, target_transforms), dim=0)
    print(all_transforms_world.shape)

    # get camera_to_worlds for the respective nerfs
    start_nerf_pipeline, start_nerf_transform = get_nerf_transform(args.start_load_config)
    target_nerf_pipeline, target_nerf_transform = get_nerf_transform(args.target_load_config)

    print(start_nerf_transform.dtype, all_transforms_world.dtype)
    all_transforms_start_nerf: torch.Tensor = torch.matmul(start_nerf_transform, all_transforms_world)
    all_transforms_target_nerf: torch.Tensor = torch.matmul(target_nerf_transform, all_transforms_world)

    cameras_start_nerf: Cameras = get_cameras(args.image_resolution, all_transforms_start_nerf)
    cameras_target_nerf: Cameras = get_cameras(args.image_resolution, all_transforms_target_nerf)

    # render all images and save them in a list
    rendered_frames_start = render_trajectory_frames(start_nerf_pipeline, cameras_start_nerf, "rgb")
    rendered_frames_target = render_trajectory_frames(target_nerf_pipeline, cameras_target_nerf, "rgb")

    # render video choosing which frames to use on some arbitratry condition
    save_video(rendered_frames_start, pathlib.Path("rendered_frames_start.mp4"))
    save_video(rendered_frames_target, pathlib.Path("rendered_frames_target.mp4"))
