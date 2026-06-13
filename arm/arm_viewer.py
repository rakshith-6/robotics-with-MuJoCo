'''
A test script to load and view any model form either,
(1) google-deepmind/mujoco_menagerie via the robot_descriptions package 
 or 
(2) load custom robot model xml file
'''

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common.utils import(
    load_custom_model,
    load_menagerie_model,
    launch_viewer,
    make_camera,
    record_video,
    generic_wave_controller,
)

def main():
    custom_model = "arm6dof.xml" # name of custom model xml file
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true",
                    help ="record an MP4 instead of opening a window")
    ap.add_argument("--menagerie", metavar="DESCRIPTION", default=None,
                    help="load a model from mujoco_menagerie using robot discription")
    ap.add_argument("--variant", default=None,
                    help="model variant, only used with --menagerie")
    ap.add_argument("--scene", default="scene.xml",
                    help="xml file of robot with robot + floor + light + skybox"
                         "Only used with --menagerie, pass 'none' to load bare file")
    args = ap.parse_args()

    if args.menagerie:
        scene = None if args.scene.lower() == "none" else args.scene
        
        model, data = load_menagerie_model(args.menagerie, args.variant, scene)
        
        variant_str = f"(variant={args.variant})" if args.variant else ""

        print(f"Loaded mujoco_menagerie/{args.menagerie}{variant_str}:"
              f"{model.nq} generalized coordinates dim, {model.nu} actuators")
        
    else:   
        model, data = load_custom_model(custom_model)
        print(f"Loaded {custom_model}: {model.nq} generalized coordinates dim, {model.nu} actuators")

    if args.video:
        if args.menagerie:
            wave = generic_wave_controller(model)
            cam = make_camera(lookat=(0.0, 0.0, 0.3), distance=2.0, azimuth=135, elevation=-20)
            robot_model = args.menagerie
        else: # for custom model arm6dof.xml
            def wave(m, d, t):
                d.ctrl[0] = 0.6 * np.sin(0.8 * t) # base pan
                d.ctrl[1] = -0.6 + 0.3 * np.sin(0.6 * t) # shoulder
                d.ctrl[2] = 1.2 + 0.3 * np.cos(0.6 * t) # elbow
                d.ctrl[4] = 0.9
            cam = make_camera(lookat=(0.1, 0.1, 0.4), distance=1.7, azimuth=135, elevation=-20)
            robot_model = os.path.splitext(custom_model)[0]

        record_video(model, data, wave, duration=6.0, fps=30, camera=cam, 
                     filename=f"{robot_model}.mp4", reset_key=0)
    else:
        launch_viewer(model, data, reset_key=0)

if __name__ == "__main__":
    main()