'''
A test script to load any model form either the google-deepmind/mujoco_menagerie 
via the robot_descriptions package or the custom robot model
'''

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
from common.utils import(
    load_custom_model,
    load_menagerie_model,
    launch_viewer,
    make_camera,
    record_video,
    generic_wave_controller,
)

def main():
    ap = 0

if __name__ == "__main__":
    main()