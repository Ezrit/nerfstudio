import dataclasses
import datetime
import os
import pathlib
import re
import sys
import typing
import xml.etree.ElementTree as ET
from typing import Callable

import mediapy as media
import numpy as np
import scipy
import scipy.interpolate
import scipy.spatial.transform
import torch
import tyro
from PIL import Image
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


def frame_list_transition(
    frame_list_start: typing.List[np.ndarray],
    frame_list_end: typing.List[np.ndarray],
    multiplier_function: typing.Optional[Callable[[float], float]] = None
) -> typing.List[np.ndarray]:
    transitioned_frame_list: typing.List[np.ndarray] = []
    length = float(min(len(frame_list_start), len(frame_list_end)))
    for i, (start_frame, end_frame) in enumerate(zip(frame_list_start, frame_list_end)):
        linear_multiplier: float = i / length
        actual_multiplier: float = linear_multiplier if multiplier_function is None else multiplier_function(linear_multiplier)

        transitioned_frame: np.ndarray = (1.0 - actual_multiplier) * start_frame + actual_multiplier * end_frame
        transitioned_frame_list.append(transitioned_frame)
    return transitioned_frame_list


def get_transition_frames(
    leading_frames: typing.List[np.ndarray],
    leading_transistion_frames: typing.List[np.ndarray],
    starting_transistion_frames: typing.List[np.ndarray],
    starting_path_frames: typing.List[np.ndarray],
    targeting_path_frames: typing.List[np.ndarray],
    targeting_transistion_frames: typing.List[np.ndarray],
    ending_transistion_frames: typing.List[np.ndarray],
    ending_frames: typing.List[np.ndarray]
) -> typing.List[np.ndarray]:
    # start with leading frames
    transitioned_frames: typing.List[np.ndarray] = leading_frames

    # Transition from leading -> starting nerf
    transitioned_frames.extend(frame_list_transition(leading_transistion_frames, starting_transistion_frames))

    # Transition from starting nerf -> targeting nerf
    transitioned_frames.extend(frame_list_transition(starting_path_frames, targeting_path_frames))

    # Transition from targeting nerf -> ending
    transitioned_frames.extend(frame_list_transition(targeting_transistion_frames, ending_transistion_frames))

    # end with ending frames
    transitioned_frames.extend(ending_frames)
    return transitioned_frames


def extract_number(f: str) -> typing.Tuple[int, str]:
    s = re.findall(r'(\d+).png', f)
    return (int(s[0]) if s else -1, f)


def get_data_frames(
    image_folder: pathlib.Path,
    frame_range: typing.Tuple[int, int],
    resolution: typing.Tuple[int, int]
) -> typing.List[np.ndarray]:
    frames: typing.List[np.ndarray] = []

    if frame_range[1] < 0:
        list_of_files: typing.List[str] = os.listdir(image_folder)
        max_number_file: str = max(list_of_files, key=extract_number)
        frame_range = (frame_range[0], extract_number(max_number_file)[0])

    for frame_number in range(frame_range[0], frame_range[1]):
        frame = Image.open(image_folder / '{number:06d}.png'.format(number=frame_number))
        np_frame = np.asarray(frame.resize((resolution[1], resolution[0]), Image.Resampling.LANCZOS)).astype(np.float32)
        np_frame = np_frame / 255.0
        frames.append(np_frame)

    return frames


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


def save_frames(iamges: typing.List[np.ndarray], output_folder) -> None:
    pass


def save_video(images: typing.List[np.ndarray], output_filename: pathlib.Path, seconds: float = 15.0) -> None:
    fps = len(images) / seconds
    # make the folder if it doesn't exist
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
        media.write_video(output_filename, images, fps=fps)
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


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


def get_intersection_point(start_transforms: torch.Tensor, target_transforms: torch.Tensor, degree: int = 1) -> np.ndarray:
    # get polynomial of start_transforms
    start_x: np.ndarray = start_transforms[:, 0].numpy()
    start_y: np.ndarray = start_transforms[:, 2].numpy()
    start_polynomial: np.polynomial.Polynomial = np.polynomial.Polynomial.fit(start_x, start_y, degree)

    # get polynomial of target_transforms
    target_x: np.ndarray = target_transforms[:, 0].numpy()
    target_y: np.ndarray = target_transforms[:, 2].numpy()
    target_polynomial: np.polynomial.Polynomial = np.polynomial.Polynomial.fit(target_x, target_y, degree)

    # get intersection (should only be 1 with degree 1 at least..., so just use the first for now)
    intersection_x: float = (start_polynomial - target_polynomial).roots()[0]
    intersection_y: float = start_polynomial(intersection_x)

    intersection_z: float = float((np.mean(start_transforms[:, 1].numpy()) + np.mean(target_transforms[:, 1].numpy())) / 2.0)
    intersection_point: np.ndarray = np.array([intersection_x, intersection_z, intersection_y])

    return intersection_point


# return an numpy array with the interpolated transforms. For the interpolation a middle point is generated with start x and end z (or vice versa depending on direction).
# y for the middle point is just the mean of both
def get_interpolation_transforms(start_transform: torch.Tensor, end_transform: torch.Tensor, intersection_point: np.ndarray, num_interpolation: int = 500) -> torch.Tensor:

    # first interpolate the positions via spline...
    start_point = np.array(start_transform[:3, 3])
    end_point = np.array(end_transform[:3, 3])

    middle_point = intersection_point[:3]

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

    append_bottom = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, 4).repeat(num_interpolation, axis=0)

    camera_to_worlds = torch.from_numpy(np.append(interpolated_transforms, append_bottom, axis=1).astype(np.float32))

    return camera_to_worlds


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


def dict_to_tensor(array_dict: typing.Union[typing.Dict[typing.Any, np.ndarray], typing.Dict[typing.Any, torch.Tensor]]) -> torch.Tensor:
    array_list: typing.List[np.ndarray] = []

    for key in sorted(array_dict):
        array_list.append(np.array(array_dict[key]))

    return torch.from_numpy(np.array(array_list).astype(np.float32))


class XMLCameraHandler:

    change_axis_rot: np.ndarray = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    change_axis_pos: np.ndarray = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    def __init__(self, xml_path: pathlib.Path):
        self.xml_file: pathlib.Path = xml_path
        self.tree: ET.ElementTree = ET.parse(self.xml_file)
        self.current_element: ET.Element = self.tree.getroot()

    def resetElement(self) -> None:
        self.current_element = self.tree.getroot()

    def getElement(self, *sub_elements: typing.Union[str, typing.Tuple[str, Callable[[ET.Element], bool]]]) -> typing.Optional[ET.Element]:
        search_element: ET.Element = self.current_element

        for sub_element in sub_elements:
            found_result = None

            if isinstance(sub_element, str):
                found_result = search_element.find(sub_element)
            else:
                for result in search_element.findall(sub_element[0]):
                    if sub_element[1](result):
                        found_result = result
                        break

            if found_result is None:
                return None
            search_element = found_result

        return search_element

    def getGroupImagesPath(self, group_id: int = -1) -> pathlib.Path:
        current_element = self.current_element
        self.resetElement()

        # sub_elements = ['chunk', 'cameras', ]
        group_element = self.getElement('chunk', 'cameras', ('group', lambda el: int(el.get('id')) == group_id))
        if group_element is None:
            return pathlib.Path()
        self.current_element = current_element

        group_name = group_element.get('label')
        if group_name is None:
            group_name = ''
        return self.xml_file.parent / group_name / 'images'

    def setElement(self, *sub_elements: typing.Union[str, typing.Tuple[str, Callable[[ET.Element], bool]]]) -> bool:
        found_element = self.getElement(*sub_elements)
        if found_element is None:
            return False
        self.current_element = found_element
        return True

    @staticmethod
    def adaptTransform(transform: np.ndarray) -> np.ndarray:
        transform[:3, :3] = (XMLCameraHandler.change_axis_rot[:3, :3] @ transform[:3, :3].transpose()).transpose()
        transform[:, 3] = XMLCameraHandler.change_axis_pos @ transform[:, 3]
        return transform

    def getTransforms(self,
                      masking_function: Callable[[ET.Element], bool],
                      *sub_elements: typing.Union[str, typing.Tuple[str, Callable[[ET.Element], bool]]]
                      ) -> typing.Optional[torch.Tensor]:
        search_element = self.getElement(*sub_elements)
        if search_element is None:
            return None

        picked_transforms = []
        for camera in search_element.findall('camera'):
            if masking_function(camera):
                transform = XMLCameraHandler.adaptTransform(np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4)))
                picked_transforms.append(transform)

        return torch.from_numpy(np.array(picked_transforms, dtype=np.float32))

    def getTransformsDict(self,
                          masking_function: Callable[[ET.Element], bool],
                          dict_key_function: Callable[[ET.Element], typing.Any],
                          *sub_elements: typing.Union[str, typing.Tuple[str, Callable[[ET.Element], bool]]]
                          ) -> typing.Optional[typing.Dict[typing.Any, torch.Tensor]]:
        search_element = self.getElement(*sub_elements)
        if search_element is None:
            return None

        transforms_dict = {}
        for camera in search_element.findall('camera'):
            if masking_function(camera):
                transform = XMLCameraHandler.adaptTransform(np.array([float(i) for i in camera.find('transform').text.split(' ')], dtype=np.float32).reshape((4,4)))
                transforms_dict[dict_key_function(camera)] = torch.from_numpy(transform)

        return transforms_dict


def read_transforms_from_xml(xml_handler: XMLCameraHandler, direction_group: int, frame_numbers: typing.Tuple[int, int]) -> typing.Dict[int, torch.Tensor]:
    stash_element = xml_handler.current_element
    xml_handler.resetElement()

    if frame_numbers[0] > frame_numbers[1]:
        frame_numbers = (frame_numbers[1], frame_numbers[0])

    xml_handler.setElement('chunk', 'cameras', ('group', lambda el: int(el.get('id')) == direction_group))

    transform_dict = xml_handler.getTransformsDict(lambda el: int(el.get('label')) >= frame_numbers[0] and int(el.get('label')) < frame_numbers[1], lambda el: el.get('label'))
    xml_handler.current_element = stash_element
    if transform_dict is None:
        return {}
    return transform_dict


# TODO: use the currently unused config variables
@dataclasses.dataclass
class PathInterpolationConfig:
    """Config for the interpolation between the paths"""

    position_interpolation_method: typing.Literal["spline"] = "spline"
    """The method for pathing positions - TODO currently unused"""
    orientation_interpolation_method: typing.Literal["slerp"] = "slerp"
    """The method for pathing orientation - TODO currently unused"""
    number_points: int = 500
    """Number of points for the path interpolation"""


@dataclasses.dataclass
class DatasetTransitionGroupData:
    """ """

    nerf_config: pathlib.Path
    """ """
    group_id: int = 0
    """ """
    transition: typing.Tuple[int, int] = (0, 100)
    """specifies the frames of this group for the transition video. Hereby the weighting between original frames will decrease (to 0 by the second framenumber)"""


@dataclasses.dataclass
class OutputConfig:
    """ """

    output_path: pathlib.Path = pathlib.Path('')
    """Output directory"""
    camera_type: typing.Literal[CameraType.PERSPECTIVE, CameraType.PANORAMA] = CameraType.PANORAMA
    """Specifies camera type for output frames"""
    image_resolution: typing.Tuple[int, int] = (600, 1200)
    """Resolution of output frames (height, width)"""
    output_single_frames: bool = False
    """Save the frames for the whole path from both dataset groups"""
    output_single_video: bool = False
    """Save the videos for the whole path from both dataset groups"""
    output_combined_frames: bool = False
    """Save the combined frames for the whole path"""
    output_combined_video: bool = True
    """Save the combined video for the whole path"""


@dataclasses.dataclass
class IntersectionTransition:
    """Intersection Transition Arguments"""

    data_folder: pathlib.Path
    """Directory specifying location of data."""
    starting_data: DatasetTransitionGroupData
    targeting_data: DatasetTransitionGroupData
    path_config: PathInterpolationConfig
    output_config: OutputConfig
    interpolate_direction: int = 0
    """Determines the corner point for the inteprolation"""

    def main(self) -> None:
        now = datetime.datetime.now()

        xml_handler = XMLCameraHandler(self.data_folder / 'cameras.xml')
        # first get camera transforms in genreal 'world' coordinates (metashape coordinates)
        start_transforms_dict: typing.Dict[int, torch.Tensor] = read_transforms_from_xml(xml_handler, self.starting_data.group_id, self.starting_data.transition)
        target_transforms_dict: typing.Dict[int, torch.Tensor] = read_transforms_from_xml(xml_handler, self.targeting_data.group_id, self.targeting_data.transition)

        start_transforms: torch.Tensor = dict_to_tensor(start_transforms_dict)
        target_transforms: torch.Tensor = dict_to_tensor(target_transforms_dict)
        intersection_point: np.ndarray = get_intersection_point(start_transforms, target_transforms, 1)
        between_transforms: torch.Tensor = get_interpolation_transforms(start_transforms[-1, ...], target_transforms[0, ...], intersection_point, self.path_config.number_points)

        # get camera_to_worlds for the respective nerfs
        start_nerf_pipeline, start_nerf_transform = get_nerf_transform(self.starting_data.nerf_config)
        target_nerf_pipeline, target_nerf_transform = get_nerf_transform(self.targeting_data.nerf_config)

        """
        all_transforms_world = torch.cat((start_transforms, between_transforms, target_transforms), dim=0)

        all_transforms_start_nerf: torch.Tensor = torch.matmul(start_nerf_transform, all_transforms_world)
        all_transforms_target_nerf: torch.Tensor = torch.matmul(target_nerf_transform, all_transforms_world)

        cameras_start_nerf: Cameras = get_cameras(self.output_config.image_resolution, all_transforms_start_nerf)
        cameras_target_nerf: Cameras = get_cameras(self.output_config.image_resolution, all_transforms_target_nerf)

        # render all images and save them in a list
        rendered_frames_start = render_trajectory_frames(start_nerf_pipeline, cameras_start_nerf, "rgb")
        rendered_frames_target = render_trajectory_frames(target_nerf_pipeline, cameras_target_nerf, "rgb")

        # render video choosing which frames to use on some arbitratry condition
        save_video(rendered_frames_start, pathlib.Path("rendered_frames_start.mp4"))
        save_video(rendered_frames_target, pathlib.Path("rendered_frames_target.mp4"))
        """

        # get leading and ending frames if any
        leading_frames: typing.List[np.ndarray] = get_data_frames(xml_handler.getGroupImagesPath(self.starting_data.group_id), (0, self.starting_data.transition[0]), self.output_config.image_resolution)
        leading_transition_frames: typing.List[np.ndarray] = get_data_frames(xml_handler.getGroupImagesPath(self.starting_data.group_id), (self.starting_data.transition[0], self.starting_data.transition[1]), self.output_config.image_resolution)

        cameras_starting_transition: Cameras = get_cameras(self.output_config.image_resolution, torch.matmul(start_nerf_transform, start_transforms))
        starting_transition_frames: typing.List[np.ndarray] = render_trajectory_frames(start_nerf_pipeline, cameras_starting_transition, "rgb")

        cameras_starting_path: Cameras = get_cameras(self.output_config.image_resolution, torch.matmul(start_nerf_transform, between_transforms))
        starting_path_frames: typing.List[np.ndarray] = render_trajectory_frames(start_nerf_pipeline, cameras_starting_path, "rgb")

        cameras_targeting_path: Cameras = get_cameras(self.output_config.image_resolution, torch.matmul(target_nerf_transform, between_transforms))
        targeting_path_frames: typing.List[np.ndarray] = render_trajectory_frames(target_nerf_pipeline, cameras_targeting_path, "rgb")

        cameras_targeting_transition: Cameras = get_cameras(self.output_config.image_resolution, torch.matmul(target_nerf_transform, target_transforms))
        targeting_transition_frames: typing.List[np.ndarray] = render_trajectory_frames(target_nerf_pipeline, cameras_targeting_transition, "rgb")

        ending_transition_frames: typing.List[np.ndarray] = get_data_frames(xml_handler.getGroupImagesPath(self.targeting_data.group_id), (self.targeting_data.transition[0], self.targeting_data.transition[1]), self.output_config.image_resolution)
        ending_frames: typing.List[np.ndarray] = get_data_frames(xml_handler.getGroupImagesPath(self.targeting_data.group_id), (self.targeting_data.transition[1], -1), self.output_config.image_resolution)

        transitioned_frames = get_transition_frames(leading_frames, leading_transition_frames,
                                                    starting_transition_frames, starting_path_frames,
                                                    targeting_path_frames, targeting_transition_frames,
                                                    ending_transition_frames, ending_frames)

        if self.output_config.output_combined_frames:
            save_frames(transitioned_frames, self.output_config.output_path)
        if self.output_config.output_combined_video:
            save_video(transitioned_frames, self.output_config.output_path / "transistioned_video.mp4")

        # write info about how the script was executed
        infoFile = self.output_config.output_path / 'info.txt'
        with open(str(infoFile.resolve()), 'w') as f:
            f.write('executed "{filename}" on {date}'.format(filename=sys.argv[0], date=now.strftime('%d/%m/%Y %H:%M:%S')))
            f.write('\n')
            f.write('exact call was:')
            f.write('\n\n')
            f.write('python {argv}'.format(argv=" ".join(sys.argv)))


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(IntersectionTransition).main()


if __name__ == "__main__":
    entrypoint()
