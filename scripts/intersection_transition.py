import argparse
import datetime
import pathlib
import typing

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=pathlib.Path, help='dataset folder with cameras.xml')
    parser.add_argument('start_direction', type=int, help='Starting direction for the transition')
    parser.add_argument('start_transistion_frame', type=int, help='frame of starting direction at which the transition should start')
    parser.add_argument('target_direction', type=int, help='Target direction for the transition')
    parser.add_argument('target_transistion_frame', type=int, help='frame of target direction at which the transition should end')

    args = parser.parse_args()
    now = datetime.datetime.now()
    pass
