"""Helper functions used by all the robots"""

from __future__ import annotations
import numpy as np
import os, mujoco
from robot_descriptions.loaders.mujoco import load_robot_description
import importlib
import imageio.v2 as imageio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MEDIA_DIR = os.path.join(PROJECT_ROOT, "media")

"""
Helper functions to load custom robot xml stored in models/ folder and 
return (model, data)
"""
def model_path_create(filename:str) -> str:
    return os.path.join(MODELS_DIR, filename)

def load_custom_model(filename:str):
    m = mujoco.MjModel.from_xml_path(model_path_create(filename))
    d = mujoco.MjData(m)
    return m, d

"""
Helper functions to load robot models from google-deepmind/mujoco_menagerie
and return (model, data). 
"""
def load_menagerie_model(description: str, variant: str | None = None,
                         scene: str | None = "scene.xml"):
    '''
    description : robot_description name("panda_mj_description")
    
    variant : model variants names(for panda without gripper variant is 
    "panda_nohand")
    
    scene : instead of bare robot file, we load the scene.xml(default name) 
    from model's package folder(pass scene=None if building custom scene)
    '''
    if variant is not None:
        m = load_robot_description(description, variant=variant)
        d = mujoco.MjData(m)
        return m, d
    
    if scene:
        try:
            module = importlib.import_module(f"robot_descriptions.{description}")
            scene_path = os.path.join(module.PACKAGE_PATH, scene)
        except(ModuleNotFoundError, AttributeError):
            scene_path = None

        if scene_path and os.path.exists(scene_path):
            m = mujoco.MjModel.from_xml_path(scene_path)
            d = mujoco.MjData(m)
            return m, d
        else:
            print(f"'{scene}' not found for '{description}'. Loading "
                  f"the bare robot model instead (no floor/lights/skybox). "
                  f"Pass scene=... to use a different scene file if this "
                  f"robot has one with another name in its package folder.")
    
    m = load_robot_description(description)
    d = mujoco.MjData(m)
    return m, d

def load_model(spec: str, variant: str | None = None, 
                      scene: str | None = "scene.xml"):
    if spec.startswith("menagerie:"):
        description = spec.split(":", 1)[1]
        return load_menagerie_model(description, variant, scene)
    return load_custom_model(spec)

"""
Helper functions for calling mojoco interactive viewer and rendering 
of video into MP4(can run headless in a server)
"""
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
    To return the robot to a saved pose(<keyframe> : full set of joint values) 
    or default pose 
    '''
    if reset_key is not None and model.nkey > reset_key:
        mujoco.mj_resetDataKeyframe(model, data, reset_key)
    else:
        mujoco.mj_resetData(model, data)

def launch_viewer(model, data, reset_key=None):
    '''
    Launches interactive mujoco viewer(with display)
    If in a Headless machine, we raise the exception
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
    print("  - Double-click a body, then ctrl-right-drag : apply forces")
    print("  - Press Esc or close the window to quit.")
    
    mujoco.viewer.launch(model, data)

def record_video(model, data, controller, duration=5.0, fps=30,
                 camera=None, width=960, height=720, filename="out.mp4",
                 reset_key=None):
    
    """
    Runs the simulation while calling controller(model, data, t) every physics
    step, rendering at fps, and saves MP4 and GIF to media/<filename>
    """
    reset_to_keyframe(model, data, reset_key) # reset_key : keyframe id to reset
    mujoco.mj_forward(model, data)

    if camera is None:
        camera = make_camera()

    os.makedirs(MEDIA_DIR, exist_ok=True)

    # To avoid exceeding of dimensions of MuJoCo's offscreen framebuffer
    if model.vis.global_.offwidth < width:
        model.vis.global_.offwidth = width
    if model.vis.global_.offheight < height:
        model.vis.global_.offheight = height

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

    gif_filename = os.path.splitext(filename)[0] + ".gif"
    gif_out = os.path.join(MEDIA_DIR, gif_filename)
    imageio.mimwrite(gif_out, frames, fps=fps)

    renderer.close()
    print(f"  saved {len(frames)} frames -> {out}")
    print(f"  saved {len(frames)} frames -> {gif_out}")
    return out

"""
A simple controller to demo joint movements of robot loaded
"""
def generic_wave_controller(model, amplitude_frac=0.3, freq=0.3):
    ctrlrange = model.actuator_ctrlrange.copy()
    has_range = model.actuator_ctrllimited.astype(bool)
    lo = np.where(has_range, ctrlrange[:, 0], -0.5)
    hi = np.where(has_range, ctrlrange[:, 1], 0.5)

    mid = (lo + hi) / 2.0
    amp = amplitude_frac * (hi - lo) / 2.0
    phase = np.arange(model.nu)

    def controller(m, d, t):
        if m.nu == 0:
            return
        d.ctrl[:] = mid + amp * np.sin(freq * t + phase)
    return controller

"""
Kinematics helper functions
"""
def joint_qpos_addr(model, joint_names):
    '''
    Returns the data.qpos indices for list of joints
    '''
    return np.array([model.joint(nm).qposadr[0] for nm in joint_names])

def joint_dof_addr(model, joint_names):
    '''
    Returns the data.qvel/DOF indices for list of joints
    '''
    return np.array([model.joint(nm).dofadr[0] for nm in joint_names])

def site_pose(model, data, site_name):
    '''
    Returns (position[3], rotation_matrix[3x3]) of a site in world frame
    
    A site is a marker attached to the body(usually IK targets the site at the 
    end-effector with site name "ee")
    '''
    sid = model.site(site_name).id
    pos = data.site_xpos[sid].copy()
    rot = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, rot

def damped_least_squares_ik(model, data, site_name, joint_names, target_pos, 
                            target_mat=None, target_z_axis=None, max_iters=200,
                            tol=1e-4, damping=1e-2, step_scale=0.5, 
                            rotation_weight=1.0):
    '''
    Returns reaching success in boolean and final_error value in float

    Implements 3 IK modes :
    * position only : target_mat=None, target_z_axis=None
    * top-down, yaw-free : target_z_axis=[0,0,-1] 
    (align the tool's z-axis with this world direction, for top grasps)
    * full 6-D pose : we use target_mat = desired 3x3 rotation

    rotation_weight : parameter to control the relative importance of orientation

    We are solving IK with damped least squares method
    (e is task error, J is end effector jacobian)
    dx = J dx -> dx = e -> but J-1 maynot exist so we use the below method
    We solve, dq = J^T (J J^T + lambda^2 I)^-1  e ,and then scale dq before update 
    '''
    sid = model.site(site_name).id
    dof = joint_dof_addr(model, joint_names)
    qadr = joint_qpos_addr(model, joint_names)

    joint_range = np.array([model.joint(nm).range for nm in joint_names])
    jacp = np.zeros((3, model.nv)) # position jacobian
    jacr = np.zeros((3, model.nv)) # rotation jacobian

    for _ in range(max_iters):
        mujoco.mj_forward(model, data) # FK
        
        cur_pos = data.site_xpos[sid] # ee current position
        err_pos = target_pos - cur_pos
        
        mujoco.mj_jacSite(model, data, jacp, jacr, sid) # compute jacobians of site

        if target_z_axis is not None:
            cur_mat = data.site_xmat[sid].reshape(3, 3) # ee current rotation matrix
            z_cur = cur_mat[:, 2]
            z_des = np.asarray(target_z_axis, dtype=float)
            z_des = z_des / np.linalg.norm(z_des)
            err_rot = np.cross(z_cur, z_des)
            err = np.concatenate([err_pos, rotation_weight * err_rot])
            J = np.vstack([jacp[:, dof], rotation_weight * jacr[:,dof]])
        
        elif target_mat is not None:
            cur_mat = data.site_xmat[sid].reshape(3, 3)
            r_err = target_mat @ cur_mat.T # rotation error
            
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r_err.flatten()) # conversion to quaternion
            
            ang = np.zeros(3)
            mujoco.mju_quat2Vel(ang, quat, 1.0) # conversion to rotation vector
            err = np.concatenate([err_pos, rotation_weight * ang])
            J = np.vstack([jacp[:, dof], rotation_weight * jacr[:, dof]])

        else:
            err = err_pos
            J = jacp[:, dof]
            
        if np.linalg.norm(err) < tol:
            return True, float(np.linalg.norm(err))
        
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(JJt.shape[0]), err)
        dq *= step_scale
        q = data.qpos[qadr] + dq
        q = np.clip(q, joint_range[:, 0], joint_range[:, 1])
        data.qpos[qadr] = q

    mujoco.mj_forward(model, data)

    return False, float(np.linalg.norm(err))
