# RSG: Fast Learning Adaptive Skills for Quadruped Robots by Skill Graph

The code repository contains relevant configuration requirements, fundamental skills training, RSG construction, inference and composition code. This repository is based off of Nikita Rudin's [legged_gym](https://github.com/leggedrobotics/legged_gym) and [AMP](https://bit.ly/3hpvbD6) repo, and enables us to train policies using [Isaac Gym](https://developer.nvidia.com/isaac-gym).

##### 1.1 CODE STRUCTURE


1. Each env is defined by an env file (`legged_gym/envs/base/legged_robot.py`) and a config file (such as `legged_gym/envs/a1/a1_amp_forward_walking_config.py`). The config file contains two classes: one conatianing all the environment parameters (`LeggedRobotCfg`) and one for the training parameters (`LeggedRobotCfgPPo`).
2. Both env and config classes use inheritance.
3. Each non-zero reward scale specified in `cfg` will add a function with a corresponding name to the list of elements which will be summed to get the total reward. The AMP reward parameters are defined in `LeggedRobotCfgPPO`, as well as the path to the reference data.
4. Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`. This is done in `legged_gym/envs/__init__.py`.
5. Skill construction code can be found in the `rsg_construction` folder.
   

## Usage

### 1. Train a single fundamental skill
##### 1.1 Installation

1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended). i.e. with conda:
   - `conda create -n sg python==3.8`
   - `conda activate sg`
2. Install pytorch 1.10 with cuda-11.3:
   - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs (`isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   - Clone this repository
   - `cd AMP_for_hardware/rsl_rl && pip install -e .`
5. Install legged_gym
   - `cd ../ && pip install -e .`

##### 1.2 Not using AMP:

```
CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --task=a1_amp_forward_walking --actor_critic_class=ActorCritic --terrain_id=16 --num_envs=3000 --max_iterations=5000 --isObservationEstimation --isEnvBaseline --headless
```

CUDA_VISIBLE_DEVICES: Specify the GPU device on which the program is running.

--task: different task.

--actor_critic_class: utilizing Actor-Critic framework (PPO).

--terrain_id: different terrain.

--num_envs: the number of environments in parallel.

--max_iterations: the number of PPO algorithm iterations.

--isObservationEstimation: context-aided estimator network (CENet) architecture.

--isEnvBaseline: goal-conditional policy.

--headless: Does not display the graphical interface.

##### 1.3 Using AMP:

The realistic demonstrations data for AMP are available [google drive](https://drive.google.com/drive/folders/13-zO4nEXpO8GyfT-_Lj77RWGAWs7o92c?usp=drive_link).

```
CUDA_VISIBLE_DEVICES=0 python legged_gym/scripts/train.py --task=a1_amp_forward_walking --actor_critic_class=ActorCritic --skills_descriptor_id=5 --terrain_id=0 --headless
```

--skills_descriptor_id: Differential weighting of intrinsic and extrinsic rewards.

### 2. RSG Construction

##### 2.1 Installation

Setup python `virtualenv` and install  packages as following:

```bash
cd rsg_construction/
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

##### 2.2 Training and evaluation

The trained skills and task descriptions are available [google drive](https://drive.google.com/file/d/1Rb6FBOC19RJNcucNqnV60VEWv0IVX-vX/view?usp=drive_link).

Train and evaluate the built RSG:

```bash
cd rsg_construction/
python train.py
```

and

```bash
cd rsg_construction/
python test.py
```

### 3. Skill inference and composition

##### 3.1 Skill inference and execution

```
python legged_gym/scripts/train.py --task=a1_amp_ct_b_sequential_2 --terrain_id=18 --num_envs=1 --max_iterations=200 --isObservationEstimation --case_id=2
```

##### 3.2 Skill composition

```
python legged_gym/scripts/train.py --task=a1_amp_ct_b_sequential_3 --terrain_id=18 --num_envs=1 --max_iterations=200 --isObservationEstimation --case_id=0
```

The skill inference is implemented in class `BOOnPolicyRunnerSequentialCase1` in file `rsl_rl/rsl_rl/runners/sg_on_policy_runner.py`.

The skill composition is implemented in class `NewCompositeActor` in file `rsl_rl/rsl_rl/modules/composite_actor_bo.py`.

The BO method is implemented in class `CompositeBO` in file `rsl_rl/rsl_rl/algorithms/ppo.py`.

