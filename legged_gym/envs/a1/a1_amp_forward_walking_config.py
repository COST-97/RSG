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

# todo:
# MOTION_FILES = glob.glob('datasets/mocap_motions/*')
MOTION_FILES = glob.glob('expert_demo/indoor/forward.pkl')
# MOTION_FILES = glob.glob('expert_demo/indoor_slow/forward_walking_slow.pkl')

# todo:
class A1AMPCfg_fw( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 1024 # 10
        
        include_history_steps = None  # Number of steps of history to include.
        
        
        # num_observations = 42
        # num_privileged_obs = 48
        num_observations = 44 #44
        num_privileged_obs = 44 #44


        episode_length_s = 10 # episode length in seconds

        reference_state_initialization = True
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        
        observation_amp_dim = 43
        num_skills = 1 # number of expert trajectories

    class init_state( LeggedRobotCfg.init_state ):
        # pos = [0.0, 0.0, 0.42] # x,y,z [m]
        # pos = [0., 0.0, 0.42] # x,y,z [m]
        pos = [0., 0.0, 0.29] # x,y,z [m]
        """
        0.26 0.7 -0.4 0.8
          0.27 -0.2 0.6 
          0.28 -0.3 0.4
          0.29 -0.3 0.3
          0.3  -0.3 0.2
          0.34 -0.5 0.2
        """
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

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        
        # stiffness = {'joint': 25.}  # [N*m/rad]
        # damping = {'joint': 0.5}     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 6

        """
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

        25
        0.5
        """

    class terrain( LeggedRobotCfg.terrain ):
        # todo:
        mesh_type = 'plane' # none, plane, heightfield or trimesh
        
        measure_heights = False

        curriculum = True
        selected = False

        # terrain_length = 8.
        # terrain_width = 8.


        # terrain_name = 'Marble Slope Uphill'
        terrain_name_list = ['Marble Slope Uphill',
                            'Marble Slope Downhill',
                            'Grassland',
                            'Grassland Slope Uphill',
                            'Grassland Slope Downhill',
                            'Grass and Pebble',
                             'Indoor Floor with Board',

                             'Grass and Sand',
                              'Grass and Mud',
                              
                              'Brushwood',
                            ]
        terrain_name = terrain_name_list[0]


    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robot_descriptions/anymal_b_simple_description/urdf/anymal.urdf'

        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = [
            "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
  
    class domain_rand:
        randomize_friction = True
        # friction_range = [0.25, 1.75]
        
        # todo:
        friction_range = [0.4, 0.7]
        # friction_range = [0.2, 0.4]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        
        push_robots = False
        push_interval_s = 7
        # max_push_vel_xy = 0.2
        max_push_vel_xy = 1.
        
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

            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        base_height_target_min = 0.24
        class scales( LeggedRobotCfg.rewards.scales ):
            
            # todo:
            # forward walking 
            termination = 0.0
            # tracking_lin_vel = 1.5 * 1. / (.005 * 6) # 50 5
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6) # 16.67 1.5
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0

            # torques = 0.0
            dof_vel = 0 # -0.
            dof_acc = 0 # -2.5e-7

            feet_air_time = 0.
            collision = 0.0
            feet_stumble = 0.0 
            # action_rate = 0.0
            stand_still = 0.0
            dof_pos_limits = 0.0
            feet_contact_forces = 0.

            tracking_lin_vel = 5.
            tracking_ang_vel = 1.5
            base_height = -10. 
            
            torques = -0.01 # -0.001 
            action_rate = -1. #-0.1 

            # imitation = 1.

            # roll
            # termination = 0.0
            # tracking_lin_vel = 0. #1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0. # 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 1.
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.
            # feet_air_time =  0.0
            # collision = 0.0
            # feet_stumble = 0.0
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

            # roll_pitch_vel=1.

            # torques = -0.00001
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0. 
            # feet_air_time =  1.0
            # collision = -1.
            # feet_stumble = -0.0 
            # action_rate = -0.01


    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 6 #4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            
            # todo:
            # forward walking 
            # lin_vel_x = [0.75,0.85] # [-1.0, 1.0] # min max [m/s]
            # lin_vel_x = [0.4,0.41] # [-1.0, 1.0] # min max [m/s]
            # lin_vel_x = [0.25,0.26] # [-1.0, 1.0] # min max [m/s]
            # lin_vel_x = [0.35,0.36] # [-1.0, 1.0] # min max [m/s]
            # lin_vel_x = [0.3,0.31] # no walking in the reality
            
            # lin_vel_x = [0.45,0.46] # [-1.0, 1.0] # min max [m/s]
            
            lin_vel_x = [0.5,0.51] # [-1.0, 1.0] # min max [m/s] #512
            lin_vel_y = [-0.0001,0.0001] #[-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-0.0001,0.0001] #[-1, 1]    # min max [rad/s]


            # lin_vel_x = [0.8,0.81] # [-1.0, 1.0] # min max [m/s] #512

            lin_vel_z = [-0.0004,0.0004] 
            lin_vel_y = [-0.0001,0.0001] #[-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-0.0001,0.0001] #[-1, 1]    # min max [rad/s]
            
            heading = [-0.0001,0.0001] # [-3.14, 3.14]
            ang_vel_roll = [-0.0001,0.0001] #[-1, 1]    # min max [rad/s]
            ang_vel_pitch = [-0.0001,0.0001] #[-1, 1]    # min max [rad/s]

            # roll 
            # lin_vel_x = [-0.01,0.01] # [-1.0, 1.0] # min max [m/s]
            # lin_vel_y = [-0.05,0.05] #[-1.0, 1.0]   # min max [m/s]
            # ang_vel_yaw = [-0.05,0.05] #[-1, 1]    # min max [rad/s]
            # heading = [-0.05,0.05] # [-3.14, 3.14]


            # ang_vel_roll = [3.5,4.5] #[-1, 1]    # min max [rad/s]
            # ang_vel_pitch = [-0.05,0.05] #[-1, 1]    # min max [rad/s]


            # lin_vel_x = [-1.0, 2.0] # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            # ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            # heading = [-3.14, 3.14]

# todo:
class A1AMPCfgPPO_fw( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''

        # todo:
        experiment_name = 'forward_walking'
        
        isMSELoss = True

        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 5000 # number of policy updates

        save_interval = 500 # check for potential saves every this many iterations

        # observation_amp_dim = 40

        # num_skills = 2 # number of expert trajectories

        amp_reward_coef = 2.0
        # amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        
        amp_reward_lerp = 0.7
        
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

        # load and resume
        # resume = False
        # load_run = "Mar07_14-37-05_" # -1 = last run
        # checkpoint = -1 # -1 = last saved model
        # resume_path = None # updated from load_run and chkpt
  