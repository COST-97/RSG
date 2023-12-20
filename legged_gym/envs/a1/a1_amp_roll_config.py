# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob('datasets/mocap_motions/*')
# MOTION_FILES = glob.glob('expert_data/Indoor Marble Flat Floor/roll_amp.pkl')

MOTION_FILES = glob.glob('expert_demo/indoor/roll.pkl')

class A1AMPCfg_ro( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 1024
        include_history_steps = None  # Number of steps of history to include.
        # num_observations = 42
        # num_privileged_obs = 48
        num_observations = 44
        num_privileged_obs = 44

        episode_length_s = 5 # episode length in seconds
        # episode_length_s = 10 # episode length in seconds too long

        reference_state_initialization = True
        reference_state_initialization_prob = 0.5
        amp_motion_files = MOTION_FILES
        
        observation_amp_dim = 43
        num_skills = 1 # number of expert trajectories

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.3] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.9,     # [rad]
            'RL_thigh_joint': 0.9,   # [rad]
            'FR_thigh_joint': 0.9,     # [rad]
            'RR_thigh_joint': 0.9,   # [rad]

            'FL_calf_joint': -1.8,   # [rad]
            'RL_calf_joint': -1.8,    # [rad]
            'FR_calf_joint': -1.8,  # [rad]
            'RR_calf_joint': -1.8,    # [rad]
        }
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.0,   # [rad]
        #     'RL_hip_joint': 0.0,   # [rad]
        #     'FR_hip_joint': 0.0 ,  # [rad]
        #     'RR_hip_joint': 0.0,   # [rad]

        #     'FL_thigh_joint': 1.56,     # [rad]
        #     'RL_thigh_joint': 1.56,   # [rad]
        #     'FR_thigh_joint': 1.56,     # [rad]
        #     'RR_thigh_joint': 1.56,   # [rad]

        #     'FL_calf_joint': -2.7,   # [rad]
        #     'RL_calf_joint': -2.7,    # [rad]
        #     'FR_calf_joint': -2.7,  # [rad]
        #     'RR_calf_joint': -2.7,    # [rad]
        # }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 80.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # none, plane, heightfield or trimesh
        measure_heights = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.25]
        
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        
        push_robots = True
        push_interval_s = 1
        max_push_vel_xy = 0.3
        
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.3
            gravity = 0.05

            
            body_height = 0.03
            foot_contact = 0.2
            foot_height = 0.02
            foot_pos = 0.02

            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.15
        base_height_target_min = 0.04
        only_positive_rewards = True
        kappa_gait_probs = 0.07
        class scales( LeggedRobotCfg.rewards.scales ):

            # forward walking 
            # termination = 0.0
            # tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.0 
            # feet_air_time =  0.0
            # collision = 0.0
            # feet_stumble = 0.0 
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

            # roll
            termination = 0.0
            tracking_lin_vel = 0. #1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0. # 0.5 * 1. / (.005 * 6)
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            dof_vel = 0 # -0.
            dof_acc = 0 # -2.5e-7

            # base_height = 0.0 
            feet_air_time = 0.
            collision = 0.0
            feet_stumble = 0.0 
            stand_still = 0.0
            dof_pos_limits = 0.0
            feet_contact_forces = 0.
            dof_vel_zeros=0.
            base_height = 0.

            # lie_orientation=10.
            # 5 6
            roll_vel_lie = 0.
            lie_orientation=0.

            # lie=2. # 1
            
            # 8/8
            lie=3. # 1
            
            BaseUprightness=1.
            foot_full_contact = 1.
            dof_acc = -0.000001
            joint_power = -0.00001
            action_rate = -0.05 #-0.1 

            # 8/7
            # yaw_vel = 1.
            
            # 8/8
            yaw_vel = 2.

            # 57
            stand = 0.
            torques = -0.000 # -0.001 

            # good ! can roll 427 
            # roll_vel_lie = 10.
            # lie_orientation=10.
            # foot_full_contact = 10.
            # torques = -0.001 # -0.001 
            # action_rate = -0.1 #-0.1 

            # roll_vel_lie = 10.
            # lie_orientation=10.
            # foot_full_contact = 10.
            # torques = -0.01 # -0.001 
            # action_rate = -1. #-0.1 

            # bad ! can not roll
            # torques = -0.1 # -0.001 
            # action_rate = -10. #-0.1 

            # may good
            # torques = -0.05 # -0.01 
            # action_rate = -5. #-1 

            # bad
            # torques = -0.1 # -0.01 
            # action_rate = -10. #-1 
            
            # bad 4.11
            # torques = -0.01 # -0.01 
            # action_rate = -3. #-1 
            
            # bad
            # torques = -0.01 # -0.01 
            # action_rate = -5. #-1 

            # no roll,just stand
            # torques = -0.01 # -0.01 
            # action_rate = -1. #-1 

            # no roll,just stand
            # torques = -0.001 # -0.01 
            # action_rate = -0.1 #-1 

            # torques = -0.1 # -0.01 
            # action_rate = -1. #-1 
            # imitation = 1.


    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 8 #4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:

            # forward walking 
            # lin_vel_x = [0.6,1.] # [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-0.05,0.05] #[-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-0.05,0.05] #[-1, 1]    # min max [rad/s]
            # heading = [-0.05,0.05] # [-3.14, 3.14]
            lin_vel_z = [-0.0004,0.0004] 
            # roll 
            lin_vel_x = [-0.0001,0.0001] # [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.0001,0.0001] #[-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-0.0001,0.0001] #[-1, 1]    # min max [rad/s]
            heading = [-0.0001,0.0001] # [-3.14, 3.14]

            # ang_vel_roll = [4.,4.1] #[-1, 1]    # min max [rad/s]
            # ang_vel_pitch = [-0.0001,0.0001] #[-1, 1]    # min max [rad/s]
            
            limit_gait_frequency = [4.0, 4.1]
            limit_gait_phase = [0.99, 0.999]
            limit_gait_offset = [0.99, 0.999]
            limit_gait_bound = [0.99, 0.999]

            # lin_vel_x = [-1.0, 2.0] # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            # ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            # heading = [-3.14, 3.14]

class A1AMPCfgPPO_ro( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01 #0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''

        experiment_name = 'roll'

        isMSELoss = True

        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 5000 # number of policy updates

        save_interval = 500 # check for potential saves every this many iterations

        # observation_amp_dim = 43

        # num_skills = 1 # number of expert trajectories

        amp_reward_coef = 2.0
        # amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        
        amp_task_reward_lerp = 0.3 # 0.7, 0.3
        
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

  
