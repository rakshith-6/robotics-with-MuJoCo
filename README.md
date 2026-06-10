# robotics-with-MuJoCo

Pick-and-place manipulation, and quadruped locomotion using the MoJoCo physics engine

# Overview

1. [About the project](#about-the-project)
2. [Instructions to run](#instructions-to-run)
3. 

---
### About the project

We use the MoJoCo physics engine on two robots,

1. A **6-DOF robot manipulator** that does forward kinematics, inverse kinematics, PD control, and a full **pick-and-place** task.
2. A **quadruped robot** that stands up, balances against a push, and moves forward with trot gait across the floor.

### Instructions to run
Create virtual environment, activate the environment and install all the dependencies using following commands

```
python3 -m venv mojoco_venv
source mojoco_venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```
