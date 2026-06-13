# robotics-with-MuJoCo

Pick-and-place manipulation, and quadruped locomotion using the MoJoCo physics engine

# Overview

1. [About the project](#about-the-project-)
2. [Instructions for setup](#instructions-for-setup-)
3. [Load models and visualize](#load-models-and-visualize-)
4. 

---
### About the project :

We use the MoJoCo physics engine on two robots,

1. A **6-DOF robot manipulator** that does forward kinematics, inverse kinematics, PD control, and a full **pick-and-place** task.
2. A **quadruped robot** that stands up, balances against a push, and moves forward with trot gait across the floor.

### Instructions for setup :
Create virtual environment, activate the environment and install all the dependencies using following commands

```bash
python3 -m venv mojoco_venv
source mojoco_venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```
### Load models and visualize :

Both robot folders has a visulaization scripts : `arm/arm_viewer.py` and `quadruped/quadruped_viewer.py`, 
using which we can load and visulaize the model. 

```bash
# Interactive viewer (default custom model : arm6dof.xml)
python arm/arm_viewer.py

# Record a demo video (runs headless)
python arm/arm_viewer.py --video

# Load a MuJoCo menagerie model
python arm/arm_viewer.py --menagerie <model_name>

# Load a specific scene (default: scene.xml)
python arm/arm_viewer.py --menagerie <model_name> --scene <scene_file>

# Load a model variant
python arm/arm_viewer.py --menagerie <model_name> --variant <variant_name>

# Record video of a menagerie model gently waving its actuators
python arm/arm_viewer.py --menagerie <model_name> --video
```

Examples:

* `--menagerie panda_mj_description`
* `--scene mjx_scene.xml` or `none` for bear model
* `--variant panda_nohand` for no gripper variant

Videos are saved to `media/<robot_model>.mp4`.

Checkout all the available robot model names and there variants at: [robot-descriptions]( https://github.com/robot-descriptions/robot_descriptions.py) and [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie.git) repositories.