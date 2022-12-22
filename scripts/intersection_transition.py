import dataclasses
import datetime
import pathlib
import typing
import xml.etree.ElementTree as ET

import numpy as np
import scipy
import scipy.interpolate
import scipy.spatial.transform
import torch
import tyro

from nerfstudio.cameras.cameras import Cameras, CameraType


def render_cameras() -> None:
    pass


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
def get_interpolation_transforms(start_transform: np.ndarray, end_transform: np.ndarray, num_interpolation: int = 100, resolution: typing.Tuple[int, int] = (600, 1200), direction: int = 0) -> torch.Tensor:

    # first interpolate the positions via spline...
    start_point = start_transform[:3, 3]
    end_point = end_transform[:3, 3]

    middle_point = np.array((start_point[0], (start_point[1] + end_point[1]) / 2.0, end_point[2]))
    if direction > 0:
        middle_point[0] = end_point[0]
        middle_point[2] = start_point[2]
    key_times = np.array((0, 1, 2))
    key_positions = np.array((start_point, middle_point, end_point))

    spline = scipy.interpolate.CubicSpline(key_times, key_positions)

    # then interpolate the rotation via slerp (linear...)
    key_rotations = scipy.spatial.transform.Rotation.from_matrix([start_transform[:3, :3], end_transform[:3, :3]])
    key_times = [0, 2]

    slerp = scipy.spatial.transform.Slerp(key_times, key_rotations)

    interpolation_times = np.linspace(0, 2, num_interpolation)
    interpolated_positions = spline(interpolation_times)
    interpolated_rotations = slerp(interpolation_times).as_matrix()
    interpolated_transforms = np.hstack((interpolated_rotations, interpolated_positions))

    camera_to_worlds = torch.from_numpy(interpolated_transforms)

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


def getCameras(resolution: typing.Tuple[int, int], camera_to_worlds: torch.Tensor) -> Cameras:
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


@dataclasses.dataclass
class IntersectionTransitionArguments:
    """Intersection Transition Arguments"""

    dataset_folder: pathlib.Path = pathlib.Path("/home/moritz/Dataset/Metashape/0_Takadanobaba_ELM")
    """Directory specifying location of data."""
    start_direction: int = 0
    """specifies the starting direction for the transition video"""
    start_transistion_frame_numbers: typing.Tuple[int, int] = (0, 100)
    """specifies the frames of starting direction for the transition video. Hereby the weighting between original frames will decrease (to 0 by the second framenumber)"""
    target_direction: int = 1
    """specifies the target direction for the transition video"""
    start_transistion_frame_numbers: typing.Tuple[int, int] = (100, 200)
    """specifies the frames of target direction for the transition video. Hereby the weighting between original frames will increase (to 1 by the second framenumber)"""
    camera_type: typing.Literal[CameraType.PERSPECTIVE, CameraType.PANORAMA] = CameraType.PANORAMA
    """Specifies camera type for output frames"""
    output_folder: pathlib.Path = pathlib.Path('Transition')
    """Output directory (relative to dataset_folder)"""
    image_resolution: typing.Tuple[int, int] = (600, 1200)
    """Resolution of output frames (height, width)"""


# TODO: define an area where we interpolate between original frames and network generated frames
if __name__ == '__main__':
    args = tyro.cli(IntersectionTransitionArguments)

    now = datetime.datetime.now()
    pass
