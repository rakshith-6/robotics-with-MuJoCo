"""Helper functions used by all the robots"""

from __future__ import annotations
import numpy as np
import os, mujoco
from robot_descriptions.loaders.mujoco import load_robot_description
import imageio.v2 as imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")

'''
Helper functions to load custom robot xml stored in models/ folder and 
return (model, data)
'''
def model_path_create(filename:str) -> str:
    return os.path.join(MODELS_DIR, filename)

def load_custom_model(filename:str):
    m = mujoco.MjModel.from_xml_path(model_path_create(filename))
    d = mujoco.MjData(m)
    return m, d

'''
Helper functions to load robot models from google-deepmind/mujoco_menagerie
and return (model, data). 

description : robot_description name("panda_mj_description")
variant : model variants names(for panda without gripper variant is 
"panda_nohand")
'''
def load_menagerie_model(description: str, variant: str | None = None):
    if variant is not None:
        m = load_robot_description(description, variant=variant)
    else:
        m = load_robot_description(description)
    d = mujoco.MjData(m)
    return m, d

def load_model(spec: str, variant: str | None = None):
    if spec.startswith("menagerie:"):
        description = spec.split(":", 1)[1]
        return load_menagerie_model(description, variant)
    return load_custom_model(spec)

'''
Helper functions for calling mojoco interactive viewer and rendering 
of video into MP4(can run headless in a server)
'''

def make_camera(lookat=(0, 0, 0.4), distance=1.6, azimuth=135, elevation=-20):
    '''
    Creates a camera pointed at lookat with some distance, azimuth and 
    elevation angle
    ''' 
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam

def reset_to_keyframe(model, data, reset_key=None):
    '''
    To return the robot to a known safe pose or will 
    '''
    if reset_key is not None and model.nkey > reset_key:
        mujoco.mj_resetDataKeyframe(model, data, reset_key)
    else:
        mujoco.mj_resetData(model, data)

def launch_viewer(model, data, reset_key=None):
    '''
    Launches interactive mujoco viewer(with display)
    Headless machine, we raise the exception
    '''
    try:
        import mujoco.viewer
    except Exception as e:
        print("Could not import the viewer:", e)
        return
    
    reset_to_keyframe(model, data, reset_key)
    if reset_key is not None:
        mujoco.mj_forward(model, data)

    print("Opening interactive viewer. Controls:")
    print("  - Left-drag  : rotate camera")
    print("  - Right-drag : pan")
    print("  - Scroll     : zoom")
    print("  - Double-click a body, then Ctrl+drag : apply forces")
    print("  - Press Esc or close the window to quit.")
    
    mujoco.viewer.launch(model, data)

def record_video(model, data, controller, duration=5.0, fps=30,
                 camera=None, width=960, height=720, filename="out.mp4",
                 reset_key=None):
    
    """
    Run the simulation while calling controller(model, data, t) every physics
    step, render at fps, and save an MP4 to media/<filename>
    """

    reset_to_keyframe(model, data, reset_key) # reset_key : keyframe id to reset
    mujoco.mj_forward(model, data)

    if camera is None:
        camera = make_camera()

    os.makedirs(MEDIA_DIR, exist_ok=True)

    # To avoid exceeding of dimensions of MuJoCo's offscreen framebuffer
    if model.vis.global_.offwidth < width:
        model.vis.global_.offwidth = width
    if model.vis.global_.offwidth < height:
        model.vis.global_.offwidth = height

    renderer = mujoco.Renderer(model, height, width)

    n_steps = int(duration / model.opt.timestep)
    frame_every = max(1, int(1.0 / (fps * model.opt.timestep)))
    
    frames = []
    for i in range(n_steps):
        t = data.time
        controller(model, data, t) # sets data.ctrl
        mujoco.mj_step(model, data)
        if i % frame_every == 0:
            renderer.update_scene(data, camera)
            frames.append(renderer.render())

    out = os.path.join(MEDIA_DIR, filename)
    imageio.mimwrite(out, frames, fps=fps)
    renderer.close()
    print(f"  saved {len(frames)} frames -> {out}")
    return out
