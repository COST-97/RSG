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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import csv
from itertools import combinations

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict
import random
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
# from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.storage.replay_buffer import ReplayBuffer
import pickle

torch.pi = torch.acos(torch.zeros(1)).item() * 2

COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET

task_name = {
"Task1":"forward_walking",
"Task2":"forward_right",
"Task3":"forward_left",
"Task4":"backward_walking",
"Task5":"backward_right_walking",
"Task6":"backward_left_walking",
"Task7":"sidestep_right",
"Task8":"sidestep_left",
"Task9":"spin_clockwise",
"Task10":"spin_counterclockwise",
"Task11":"gallop",

"Task12":"forward_walking_fast",
"Task13":"forward_mass",
"Task14":"forward_noise",

"Task15":"up",
"Task16":"up1",
"Task17":"up_backward",
"Task18":"up_forward",
"Task19":"up_left",
"Task20":"up_right",

"Task21":"roll",
"Task22":"standup",

"Task23":"gallop_fast",
}

task_command = {
"Task1":[0.8,0.,0.,0.,0.,0.],
"Task2":[0.4,0.,0.,0.,0.,-0.4], #"Task2":"forward_right",
"Task3":[0.4,0.,0.,0.,0.,0.4],

"Task4":[-0.6,0.,0.,0.,0.,0.], #"Task4":"backward_walking",
"Task5":[-0.5,0.,0.,0.,0.,0.4],

"Task6":[-0.4,0.,0.,0.,0.,-0.4], #"Task6":"backward_left_walking",

"Task7":[0.,-0.3,0.,0.,0.,0.],
"Task8":[0.,0.3,0.,0.,0.,0.], #"Task8":"sidestep_left",

"Task9":[0.,0.,0.,0.,0.,-2.], #"Task9":"spin_clockwise",
"Task10":[0.,0.,0.,0.,0.,2.], #"Task10":"spin_counterclockwise",
"Task11":[1.5,0.,0.,0.,0.,0.], # "Task11":"gallop",

"Task12":[1.5,0.,0.,0.,0.,0.], # "Task12":"forward_walking_fast",
"Task13":[0.6,0.,0.,0.,0.,0.],
"Task14":[0.6,0.,0.,0.,0.,0.],

"Task15":[0.,0.,2.,0.,0.,0.],
"Task16":[0.,0.,2.,0.,0.,0.],

"Task17":[-1.,0.,2.,0.,0.,0.],
"Task18":[1.,0.,2.,0.,0.,0.],
"Task19":[0.,0.75,2.,0.,0.,0.],
"Task20":[0.,-0.75,2.,0.,0.,0.],

"Task21":[0.,0.,0.,0.,0.,0.],
"Task22":[0.,0.,0.,0.,0.,0.],

"Task23":[2.,0.,0.,0.,0.,0.], # "Task11":"gallop_fast"
}

skills_name = ["up_oe",
    "up1_oe",
    "up_backward_oe",
    "up_forward_oe",
    "up_left_oe",
    "up_right_oe",
    "roll_oe",
    "standup_oe",
    "backward_left_walking_oe",
    "backward_right_walking_oe",
    "backward_walking_oe",
    "forward_left_oe",
    "forward_mass_oe",
    "forward_noise_oe",
    "forward_right_oe",
    "forward_walking_oe",
    "forward_walking_fast_oe",
    "gallop_oe",
    "gallop_fast_oe",
    "sidestep_left_oe",
    "sidestep_right_oe",
    "spin_clockwise_oe",
    "spin_counterclockwise_oe"
    ]

skills_name_normal = [
    "backward_left_walking_oe",
    "backward_right_walking_oe",
    "backward_walking_oe",

    "forward_mass_oe",
    "forward_noise_oe",
    "forward_walking_oe",

    "gallop_oe",
    "gallop_fast_oe",
    ]

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless,
     amp_replay_buffer_size=100000,
     skills_descriptor_id=None,
     terrain_id=None,
     leg_lift_id=None,
     isActionCorrection=False,
     case_id=None,
     curriculum_start_num=0,
     isObservationEstimation=False,
     isEnvBaseline=False,
     isBOEnvBaseline=False
  
    ):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        
        self.debug_viz = False

        self.init_done = False

        self.external_force_list = []
        
        self.device = sim_device

        self.case_id=case_id
        
        self.files = self.cfg.env.amp_motion_files[0]
        
        self.isEnvBaseline = isEnvBaseline

        self.isBOEnvBaseline=isBOEnvBaseline
        
        self.cfg.env.reference_state_initialization = False
 
        self.skills_descriptor_id=skills_descriptor_id


        self.cfg.env.IsObservationEstimation = isObservationEstimation

        self.cfg.control.stiffness['joint']=40. # [N*m/rad]
        self.cfg.control.damping['joint']=1.

        self.down_num = 0

        self.cfg.domain_rand.push_robots = False # real experiment show that: False

        if isObservationEstimation: 
            self.cfg.env.num_observations = 45 
            self.cfg.env.num_privileged_obs = 244 

            if self.isEnvBaseline:
                self.cfg.env.useRealData = False

                self.cfg.env.usePD = False
                
                self.cfg.env.useRewardDone = False
                
                self.cfg.env.useTrackingError = False
                self.cfg.env.useTrackingError_two = False

                if  self.cfg.env.useTrackingError:
                    self.cfg.env.include_history_steps = 7                    
                else:
                    self.cfg.env.include_history_steps = 6

               
                self.cfg.env.useWBC = False

                self.cfg.env.usePrivilegeLabel = True

                self.cfg.env.ObsEstModelName = "VAE"

                self.cfg.env.reward_dim = None

                self.cfg.env.current_addition_dim = 3+12 # command and last action 

                self.cfg.env.state_est_dim = 3 # vel 3 foot pos 20 # foot height 12

                self.cfg.env.num_observations = 45 + self.cfg.env.state_est_dim
                self.cfg.env.num_privileged_obs = 232 + self.cfg.env.state_est_dim + (12+4+1+1+12+12)
                
              
                if self.cfg.env.usePD:
                    self.cfg.env.num_observations+=(12+12)
                    self.cfg.env.num_privileged_obs+=(12+12)

                    self.cfg.env.current_addition_dim+=24

              

                self.cfg.control.stiffness['joint']=28. # [N*m/rad]
                self.cfg.control.damping['joint']=0.7

                self.cfg.terrain.curriculum = True
                self.cfg.commands.curriculum = True   

                self.cfg.terrain.measure_heights = True

                self.cfg.control.decimation = 4

            elif (("roll" in self.files) or ("stand_up" in self.files)):
                self.cfg.domain_rand.push_robots = True

                self.cfg.env.useRealData = False

                self.cfg.env.usePD = False
                
                self.cfg.env.useRewardDone = False
                
                self.cfg.env.useTrackingError = False
                self.cfg.env.useTrackingError_two = False

                if  self.cfg.env.useTrackingError:
                    self.cfg.env.include_history_steps = 7                    
                else:
                    self.cfg.env.include_history_steps = 6

                self.cfg.env.ObsEstModelName = "VAE"

                self.cfg.env.reward_dim = 6
                if "stand_up" in self.files:
                    self.cfg.env.reward_dim = 7

                self.cfg.env.current_addition_dim = 3+12 # command and last action 

                self.cfg.env.state_est_dim = 3 # vel 3 foot pos 20 # foot height 12

                self.cfg.env.num_observations = 45 + self.cfg.env.state_est_dim
                self.cfg.env.num_privileged_obs = 232 + self.cfg.env.state_est_dim + (12+4+1+1+12+12)
              
                if self.cfg.env.usePD:
                    self.cfg.env.num_observations+=(12+12)
                    self.cfg.env.num_privileged_obs+=(12+12)

                    self.cfg.env.current_addition_dim+=24

                self.cfg.control.stiffness['joint']=28. # [N*m/rad]
                self.cfg.control.damping['joint']=0.7

                self.cfg.terrain.curriculum = True
                self.cfg.commands.curriculum = False    

                self.cfg.terrain.measure_heights = True

            elif "down_up" in self.files:
                self.cfg.env.useRealData = False

                self.cfg.env.usePD = False
                
                self.cfg.env.useRewardDone = False
                
                self.cfg.env.useTrackingError = False
                self.cfg.env.useTrackingError_two = False

                if  self.cfg.env.useTrackingError:
                    self.cfg.env.include_history_steps = 7                    
                else:
                    self.cfg.env.include_history_steps = 6

                self.cfg.env.ObsEstModelName = "VAE"

                self.cfg.env.reward_dim = 8

                self.cfg.env.current_addition_dim = 12 # command and last action 

                self.cfg.env.state_est_dim = 3 # vel 3 foot pos 20 # foot height 12

                self.cfg.env.num_observations = 45 + self.cfg.env.state_est_dim
                self.cfg.env.num_privileged_obs = 232 + self.cfg.env.state_est_dim + (12+4+1+1+12+12)
                # self.cfg.env.num_privileged_obs = 232 + 3 + (12+4+1)
                if self.cfg.env.usePD:
                    self.cfg.env.num_observations+=(12+12)
                    self.cfg.env.num_privileged_obs+=(12+12)

                    self.cfg.env.current_addition_dim+=24

                self.cfg.control.stiffness['joint']=28. # [N*m/rad]
                self.cfg.control.damping['joint']=0.7

                self.cfg.terrain.curriculum = False
                self.cfg.commands.curriculum = False    

                self.cfg.terrain.measure_heights = True

        self.alpha = torch.ones(self.cfg.env.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        self.isActionCorrection = isActionCorrection
        if self.isActionCorrection:
            self.reset_action_correction_param()
        
        self._enable_action_filter = False
        action_filter_highcut = 5.
        self._action_filter_highcut = action_filter_highcut

        if self._enable_action_filter:
            self._action_filter_list = []
            for _ in range(self.cfg.env.num_envs):
                self._action_filter = self._BuildActionFilter(
                    [self._action_filter_highcut])
                self._action_filter_list.append(self._action_filter)
        
        self.step_counter = 0

        self.curriculum_start_num = curriculum_start_num

        self.set_leg_id(leg_lift_id)

        self.set_terrain_params(terrain_id)

        self.sg = None

        if self.terrain_name in ['ComplexTerrain_NewEnv','ComplexTerrain_Sequential','ComplexTerrain_Sequential_Case_1'] and self.isEnvBaseline is False:
            from skill_graph import SkillGraph

            TRAINED_FOLDER = "models"

            self.sg = SkillGraph({"hidden_dim": 512, "skill_dim": 96, "read_skill": True})
            self.sg.load(TRAINED_FOLDER)

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        self.observation_amp_dim = self.cfg.env.observation_amp_dim

        self.amp_data = ReplayBuffer(
        self.cfg.env.observation_amp_dim, amp_replay_buffer_size, sim_device)

        self.isStateInterpolationNumber = 1

        Num_skills =self.cfg.env.num_skills

        for k in range(len(self.cfg.env.amp_motion_files)):
            print(self.cfg.env.amp_motion_files[k])
            
            f_read = open(self.cfg.env.amp_motion_files[k], 'rb')
            dict2 = pickle.load(f_read)
            states_array = dict2["states"][:Num_skills]
            next_states_array = dict2["next_states"][:Num_skills]
            f_read.close()

            for i in range(Num_skills):
                if self.isStateInterpolationNumber>1:
                    states_inter = self.state_interpolation(states=states_array[i], inter_num=self.isStateInterpolationNumber)
                    # next_states = self.state_interpolation(states=next_states_array[i], inter_num=self.isStateInterpolationNumber)
                    states = states_inter[:-1]
                    next_states = states_inter[1:]
                else:
                    states=states_array[i]
                    next_states=next_states_array[i]

                states = torch.tensor(states,dtype=torch.float, device=self.device)
                next_states =  torch.tensor(next_states,dtype=torch.float, device=self.device)
                self.amp_data.insert_expert_data(states, next_states)

        print('='*50)
        # print(self.cfg.env.amp_motion_files[0])
        print("expert num_samples:",self.amp_data.num_samples)
        print('='*50) 

        self.skills_weight = torch.tensor([[0.2500, 0.2500, 0.5000], 
                                          
                                [0.2500, 0.5000, 0.2500],
                                [0.5000, 0.2500, 0.2500], 
                                
            
                                [0.6000, 0.200, 0.200], 
                              

                                [0.8000, 0.100, 0.100], 

                                [0.200, 0.200, 0.6000], 

                                [0.100, 0.100, 0.8000], 

                                 [1.00, 1.00, 0.1000], 
                                ], device=self.device, requires_grad=False, dtype=torch.float)
        
        self.set_skills_descriptor()

    def state_interpolation(self,states, inter_num=1):
        x = np.linspace(0,len(states),len(states))
        xvals = np.linspace(0,len(states),int(inter_num*len(states)))

        states_inter = []
        for i in range(len(states[0,:])):
            y = states[:,i]
            yinterp = np.interp(xvals, x, y)
            states_inter.append(yinterp)
        states_inter = np.array(states_inter).T
        return states_inter

    def set_seed(self, seed):
        if seed == -1:
            seed = np.random.randint(0, 10000)
        print("Setting seed: {}".format(seed))
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def reset_action_correction_param(self):
        num_envs = self.cfg.env.num_envs

        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        self._model=pinocchio.buildModelFromUrdf(asset_path)
        self._data=self._model.createData()

        self._F_pos = torch.ones(num_envs, 12, device=self.device, requires_grad=False) * 1000.
        self._motor_qr = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._P_error = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._F_y_dot = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._F_y = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._motor_qd_d = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._motor_qd = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._motor_qr_d = torch.ones(num_envs, 12, device=self.device, requires_grad=False) * 1000.
        self._motor_qr_a = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._motor_s = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._motor_qc_d = torch.zeros(num_envs, 12, device=self.device, requires_grad=False)
        self._motor_qc = torch.ones(num_envs, 12, device=self.device, requires_grad=False) * 1000.
        self._q_b_dot_o = torch.zeros(num_envs, 12, device=self.device, requires_grad=False) 
        self._ruo = torch.zeros(num_envs, 12, device=self.device, requires_grad=False) 
        # self.K1 = 0.9  # 10Hz
        # self.beta = 5.
        self.Kp_factors = torch.ones(num_envs, 12, device=self.device, requires_grad=False, dtype=torch.float)
        self.Kd_factors = torch.ones(num_envs, 12, device=self.device, requires_grad=False, dtype=torch.float)
        self.torques = torch.zeros(num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.alpha = torch.ones(num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False).view(num_envs,-1)
        
        self.desired_dof_pos = torch.ones(num_envs, 12, device=self.device, requires_grad=False)*torch.tensor([0.,0.9,-1.8]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        # print(self.desired_dof_pos)

        self.current_desired_dof_pos = torch.ones(num_envs, 12, device=self.device, requires_grad=False)*torch.tensor([0.,0.9,-1.8]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        
        self.K1 = 0.016*np.identity(12)  # 10Hz
        self.beta = 10.
        
        # self.Kp = np.identity(12) * 1.0
        # self.Ki = np.identity(12) * 70.
        self.Kp = np.identity(12) * 0.7
        self.Ki = np.identity(12) * 28.

        self._eta = 0.05
        self._e_gain = 5.5
        self._d_hat = 0.
        self._k = 0.000
        self._rho = 0.01

        self._flag_s = 0


    def _BuildActionFilter(self, highcut=None):
        sampling_rate = 1. / 0.03 #(self.time_step * self._action_repeat)
        num_joints = 12 #self.GetActionDimension()
        a_filter = ActionFilterButter(
            sampling_rate=sampling_rate,
            num_joints=num_joints,
            highcut=highcut)
        return a_filter

    def _ResetActionFilter(self,env_ids):
        default_action = self.dof_pos.cpu().numpy() 
      

        for i in env_ids.cpu().numpy():
            self._action_filter_list[i].reset()
     
            self._action_filter_list[i].init_history(default_action[i])

    def _FilterAction(self, action):
        action = action.cpu().numpy()        
        filtered_action_list = []
        for i in range(self.num_envs):
            filtered_action = self._action_filter_list[i].filter(action[i])
            filtered_action_list.append(filtered_action)

        filtered_action = torch.tensor(np.array(filtered_action_list),dtype=torch.float, device=self.device)
        return filtered_action
    
    def set_leg_id(self,leg_lift_id):
        self.leg_id = leg_lift_id
        if self.leg_id is not None:
            self.cfg.rewards.scales.lie_orientation=0.1
            self.cfg.rewards.scales.joint_pos = 1.5
            self.cfg.rewards.scales.foot_contact = -2.

    def set_terrain_params(self,terrain_id):
        self.terrain_name_list = [
                                'Indoor Floor', #0
                         
                            'Ice Surface', #1
                             

                            'UpStairs', # 2
                            'DownStairs', # 3

                            'Marble Slope Uphill', #4
                            'Marble Slope Downhill', #5
                            'Grassland', #6
                            'Grassland Slope Uphill', #7
                            'Grassland Slope Downhill', #8
                            'Grass and Pebble', #9
                            'Steps', #10
                            'Grass and Sand', #11
                          
                            'Hills', # 12

                         
                            'ComplexTerrain_NewEnv', # 13
                            'ComplexTerrain_NewTask', # 14
                            'ComplexTerrain_Sequential', # 15

                            'ComplexTerrain_Baseline', # 16

                            "ComplexTerrain_NewEnv_opt", # 17

                            'ComplexTerrain_Sequential_Case_1', # 18
                            ]
        
     
        print("terrain_id",terrain_id)
        self.terrain_name = self.terrain_name_list[terrain_id]   
        print("="*50)
        print( self.terrain_name)
        print("="*50)

        self.cfg.terrain.terrain_name = self.terrain_name

        if self.terrain_name in [
                            'Indoor Floor',
                            'Ice Surface',
                            ]:
            self.cfg.terrain.mesh_type = 'plane' # none, plane, heightfield or trimesh
        else:
            self.cfg.terrain.mesh_type = 'trimesh' # none, plane, heightfield or trimesh

        self.cfg.init_state.pos = [0.0, 0.0, 0.29] # x,y,z [m]

        

        if "gallop" in self.files:                       
            self.cfg.init_state.pos = [-8.0, 0.0, 0.29] # x,y,z [m]   
            
        if self.terrain_name in ['Marble Slope Uphill',
                            'Marble Slope Downhill',
                            'Grassland Slope Uphill',
                            'Grassland Slope Downhill',
                            
                        
                            ]:
            self.cfg.terrain.terrain_length = 6.
            self.cfg.terrain.terrain_width = 20.  
            self.cfg.terrain.curriculum = True   

        if self.terrain_name in [
                            
                            'UpStairs', #14
                            'DownStairs', #15
               
                            ]:
            self.cfg.terrain.terrain_length = 8.
            self.cfg.terrain.terrain_width = 8.  
            self.cfg.terrain.curriculum = True #training  

        if self.terrain_name in [
                            
                      
                            'Hills'
                            ]:
            self.cfg.terrain.terrain_length = 8.
            self.cfg.terrain.terrain_width = 8.  
            self.cfg.terrain.curriculum = True  

        if self.terrain_name in ['Indoor Floor']:
            self.cfg.domain_rand.friction_range = [0.6, 0.9]
        elif self.terrain_name in ['Steps']:
            self.cfg.domain_rand.friction_range = [0.6, 1.2]
        elif self.terrain_name in [
                                 'Marble Slope Uphill',
                                'Marble Slope Downhill',
                                ]:
            self.cfg.domain_rand.friction_range = [0.7, 1.1]

        elif self.terrain_name in ['Ice Surface']:
            self.cfg.domain_rand.friction_range = [0.01, 0.1]

        elif self.terrain_name in ['UpStairs','DownStairs']:
            self.cfg.domain_rand.friction_range = [1.2, 1.5]

        elif self.terrain_name in ['Grassland']:
            self.cfg.domain_rand.friction_range = [0.5, 0.7]
        elif self.terrain_name in ['Grassland Slope Uphill',
                                   'Grassland Slope Downhill']:
            self.cfg.domain_rand.friction_range = [0.5, 0.7]

        elif self.terrain_name in ["Hills"]:
            self.cfg.domain_rand.friction_range = [0.2, 0.3]

        elif self.terrain_name in ["Grassland and Pebble"]:
            self.cfg.domain_rand.friction_range = [0.05, 0.1]
        elif self.terrain_name in ["Grassland and Sand"]:
            self.cfg.domain_rand.friction_range = [0.3, 0.4]

        if self.isEnvBaseline:
            self.cfg.env.episode_length_s = 20 
         
            self.cfg.domain_rand.friction_range = [0.1, 1.25]

            self.cfg.terrain.mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
            self.cfg.terrain.horizontal_scale = 0.1 # [m]
            self.cfg.terrain.vertical_scale = 0.005 # [m]
            self.cfg.terrain.border_size = 25 # [m]

            self.cfg.terrain.static_friction = 1.0
            self.cfg.terrain.dynamic_friction = 1.0
            self.cfg.terrain.restitution = 0.
            # rough terrain only:
            self.cfg.terrain.measure_heights = True
            self.cfg.terrain.measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
            self.cfg.terrain.measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
            self.cfg.terrain.selected = False # select a unique terrain type and pass all arguments
            self.cfg.terrain.terrain_kwargs = None # Dict of arguments for selected terrain
            self.cfg.terrain.max_init_terrain_level = 5 # starting curriculum state
            self.cfg.terrain.terrain_length = 8.
            self.cfg.terrain.terrain_width = 8.
            self.cfg.terrain.num_rows= 10 # number of terrain rows (levels)
            self.cfg.terrain.num_cols = 20 # number of terrain cols (types)

            # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
            self.cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
            
   
            if self.case_id is not None:
                self.cfg.terrain.terrain_proportions = [0.4, 0.4, 0., 0., 0.2]
           
       
            self.cfg.terrain.slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

        elif (("roll" in self.files) or ("stand_up" in self.files)):
            self.cfg.env.episode_length_s = 5
            self.cfg.domain_rand.friction_range = [0.1, 1.25]

            self.cfg.terrain.mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
            self.cfg.terrain.horizontal_scale = 0.1 # [m]
            self.cfg.terrain.vertical_scale = 0.005 # [m]
            self.cfg.terrain.border_size = 25 # [m]

            self.cfg.terrain.static_friction = 1.0
            self.cfg.terrain.dynamic_friction = 1.0
            self.cfg.terrain.restitution = 0.
            # rough terrain only:
            self.cfg.terrain.measure_heights = True
            self.cfg.terrain.measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
            self.cfg.terrain.measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
            self.cfg.terrain.selected = False # select a unique terrain type and pass all arguments
            self.cfg.terrain.terrain_kwargs = None # Dict of arguments for selected terrain
            self.cfg.terrain.max_init_terrain_level = 5 # starting curriculum state
            self.cfg.terrain.terrain_length = 8.
            self.cfg.terrain.terrain_width = 8.
            self.cfg.terrain.num_rows= 10 # number of terrain rows (levels)
            self.cfg.terrain.num_cols = 20 # number of terrain cols (types)
            # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
            self.cfg.terrain.terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
            # self.cfg.terrain.terrain_proportions = [1., 0., 0., 0., 0.]
            # trimesh only:
            self.cfg.terrain.slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

            self.cfg.init_state.pos = [0., 0.0, 0.25]
            if "stand_up" in self.files:
                self.cfg.init_state.pos = [0., 0.0, 0.25]

        if self.terrain_name in ['ComplexTerrain_NewEnv']:
            self.design_tasks()
            

            self.init_terrain_id = [0,0] #  num_cols, num_rows
            self.cfg.init_state.pos = [0., 0.0, 0.29] # x,y,z [m]

            self.cfg.terrain.curriculum = False  

            self.cfg.terrain.selected = False 

            self.cfg.terrain.num_rows = 1 # number of terrain rows (levels)
            self.cfg.terrain.num_cols = 1 # number of terrain cols (types)  
            
            self.cfg.commands.curriculum = False

            self.sequential_num = 1


            env_params = np.random.uniform(
                                low=[0.01,-0.35,0.01],
                               high=[1.5,0.3,0.1], # max 
                               size=(100,3)
                               )


            self.cfg.domain_rand.friction_range = [env_params[self.case_id][0],env_params[self.case_id][0]+0.000001]
            self.cfg.terrain.slope = [env_params[self.case_id][1],env_params[self.case_id][1]+0.000001] 
            self.cfg.terrain.terrain_height = [env_params[self.case_id][2],env_params[self.case_id][2]+0.000001]


            self.cfg.terrain.terrain_length = 6.
            self.cfg.terrain.terrain_width =50. 


            self.cfg.rewards.scales.termination = 0. #-0.0
            self.cfg.rewards.scales.tracking_lin_vel = 0.
            self.cfg.rewards.scales.tracking_ang_vel = 0.

            self.cfg.rewards.scales.lin_vel_z = 0.
            self.cfg.rewards.scales.ang_vel_xy = 0.

            self.cfg.rewards.scales.orientation = 0.
            
            self.cfg.rewards.scales.torques = 0. #-0.00001
            self.cfg.rewards.scales.dof_vel = 0. #-0.
            self.cfg.rewards.scales.dof_acc = 0.
            self.cfg.rewards.scales.joint_power = 0.

            self.cfg.rewards.scales.base_height = 0.

            self.cfg.rewards.scales.feet_air_time =  0.

            self.cfg.rewards.scales.collision = 0. # -1.
            self.cfg.rewards.scales.feet_stumble = 0. #-0.0 

            self.cfg.rewards.scales.feet_clearance =  -0.01 # 1.0
            
            self.cfg.rewards.scales.action_rate = 0.
            self.cfg.rewards.scales.smoothness = 0.
            self.cfg.rewards.scales.power_distribution = 0. # -1e-5 # too small
            
            self.cfg.rewards.scales.stand_still = 0. #-0.

            self.cfg.rewards.scales.tracking_vel_all = 10.
            self.cfg.rewards.scales.torques = -0.001 #-0.0005 
            self.cfg.rewards.scales.action_rate = -0.1 #-0.05

            self.cfg.rewards.only_positive_rewards = False

        elif self.terrain_name in ['ComplexTerrain_NewTask']:
            self.init_terrain_id = [0,0] #  num_cols, num_rows
            self.cfg.init_state.pos = [0., 0., 0.29] # x,y,z [m]
            self.cfg.terrain.terrain_length = 10.
            self.cfg.terrain.terrain_width = 10. 
            self.cfg.terrain.curriculum = False  
            self.cfg.terrain.selected = False 
            self.cfg.terrain.num_rows = 1 # number of terrain rows (levels)
            self.cfg.terrain.num_cols = 1 # number of terrain cols (types)  

            #env param
            self.hardness_list =  [0.,0.0001]
            self.resistance_list =  [0.,0.0001]
            self.cfg.domain_rand.friction_range = [0.5,0.5001]
            self.cfg.terrain.slope =  [0.,0.0001]
            self.cfg.terrain.terrain_height = [0.,0.0001]

        elif self.terrain_name in ['ComplexTerrain_Sequential']:
            self.design_tasks()
            self.sequential_num = 5

            self.task_params = np.random.randint(0,len(skills_name_normal),size=(100,self.sequential_num))
    
            self.cfg.env.episode_length_s = 30

            self.cfg.terrain.terrain_length = 10.
            self.cfg.terrain.terrain_width = 10.  
            self.cfg.terrain.curriculum = False  
            self.cfg.terrain.selected = False 
            self.cfg.terrain.terrain_proportions = [1.]
            # self.cfg.terrain.terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.15, 0.1,0.15]       
            self.cfg.terrain.num_rows = 5 #5 # number of terrain rows (levels)
            self.cfg.terrain.num_cols = 5 #5 # number of terrain cols (types)    
            
            self.init_terrain_id = [2,2] #  num_cols, num_rows

            self.cfg.init_state.pos = [0.0, 0.0, 0.29] # x,y,z [m]

            self.cfg.domain_rand.friction_range = [0.8,0.8001]
            self.cfg.terrain.slope =  [0.,0.0001]
            self.cfg.terrain.terrain_height = [0.,0.0001]

        elif self.terrain_name in ['ComplexTerrain_NewEnv_opt']:
            self.init_terrain_id = [0,0] #  num_cols, num_rows
            self.cfg.init_state.pos = [0., 0.0, 0.29] # x,y,z [m]

            self.cfg.terrain.curriculum = True  

            self.cfg.terrain.selected = False 

            self.cfg.commands.curriculum = True

            self.cfg.domain_rand.friction_range = [0.01,1.5]

            self.cfg.terrain.terrain_length = 6.
            self.cfg.terrain.terrain_width = 50. 


        elif self.terrain_name in ['ComplexTerrain_Sequential_Case_1']:
            self.design_tasks()
            self.sequential_num = 5

            self.task_params = np.random.randint(0,len(skills_name_normal),size=(100,self.sequential_num))
    
            self.cfg.env.episode_length_s = 120

            self.cfg.terrain.terrain_length = 6.
            self.cfg.terrain.terrain_width = 6.  

            self.cfg.terrain.curriculum = False  
            self.cfg.terrain.selected = False 
            self.cfg.terrain.terrain_proportions = [1.]

            self.cfg.domain_rand.friction_range = [0.8,0.8001]
            self.cfg.terrain.slope =  [0.,0.0001]
            self.cfg.terrain.terrain_height = [0.,0.0001]

            self.cfg.terrain.case = 2

            if self.cfg.terrain.case == 0:
                self.cfg.terrain.num_rows = 1 #5 # number of terrain rows (levels)
                self.cfg.terrain.num_cols = 3 #5 # number of terrain cols (types)  
                self.init_terrain_id = [1,0] #  num_cols, num_rows
                self.cfg.init_state.pos = [-1.2, -0.2, 0.29] # x,y,z [m]

            elif self.cfg.terrain.case == 1:
                
                self.cfg.env.episode_length_s = 500
                
                self.cfg.domain_rand.friction_range = [0.9,0.9001]

                self.cfg.viewer.pos = [6, -3, 7]  # [m]
                self.cfg.viewer.lookat = [6., 5, 0.]  # [m]

                self.cfg.terrain.num_rows = 1 #5 # number of terrain rows (levels)
                self.cfg.terrain.num_cols = 2 #5 # number of terrain cols (types)  
                
      
                self.cfg.terrain.terrain_length = 8.
                self.cfg.terrain.terrain_width = 6. 

                self.init_terrain_id = [0,0] #  num_cols, num_rows
                self.cfg.init_state.pos = [0., 0., 0.29] # x,y,z [m]


            elif self.cfg.terrain.case == 2:
                self.cfg.terrain.num_rows = 1 #5 # number of terrain rows (levels)
                self.cfg.terrain.num_cols = 1 #5 # number of terrain cols (types)  
                self.cfg.terrain.terrain_length = 20.
                self.cfg.terrain.terrain_width = 20.  
                self.init_terrain_id = [0,0] #  num_cols, num_rows
                self.cfg.init_state.pos = [0., 0., 0.29] # x,y,z [m]

    def design_tasks(self):
        self.body_move_dict = {}
        for name in skills_name:
            Returns = -np.Inf
            file_max_return=None
            
            for filepath,dirnames, filenames in os.walk("entiti_identity/"):
                for filename in filenames:
                    file = os.path.join(filepath,filename)
                    # print(filename)
                    if "/"+name in file:
                        # print(filepath)
                        R = eval(filepath.split("_R_")[1])
                        if R>Returns:
                            Returns = R
                            file_max_return = file

            if file_max_return is not None:
                # print(file_max_return)
                data = np.load(file_max_return)
                
                body_move = data[0,:80:2,:6]

                for key,values in task_name.items():
                    # print("/"+values+"_oe")
                    if "/"+values+"_oe" in file_max_return:
                        task_commands = torch.tensor(task_command[key], dtype=torch.float)
                        break  
                # print(task_commands) 
                self.body_move_dict[name] = [body_move,task_commands] 
        print("Tasks:",self.body_move_dict.keys())

    def get_env_param(self):
        # hardness = np.round(self.hardness,4)
        friction = self.friction_coeffs.squeeze().numpy()

        if self.terrain_name in ['ComplexTerrain_NewEnv']:
            x,y = self.init_terrain_id[0],self.init_terrain_id[1]
        elif self.terrain_name in ['ComplexTerrain_Sequential','ComplexTerrain_Sequential_Case_1']:
            x = (torch.div(self.root_states[:, 0],self.cfg.terrain.terrain_width)).squeeze().cpu().numpy()
            y = (torch.div(self.root_states[:, 1],self.cfg.terrain.terrain_length)).squeeze().cpu().numpy()
            x,y = int(x), int(y)

        flatness = self.terrain.env_param[str(x)+str(y)][0]
        slope = self.terrain.env_param[str(x)+str(y)][1]
        # return [hardness,friction,flatness,slope,resistance]
        return np.round(np.array([friction,flatness,slope]),2)

    def set_skills_descriptor(self):
        # task response, energy consumption, imitation demonstration
        if self.skills_descriptor_id is not None:

            self.skills_descriptor_weight = self.skills_weight[self.skills_descriptor_id]        
            
            print("response task, energy consumption, skill prior")            
            print("Skills descriptor weight",self.skills_descriptor_weight)
        
        
        if self.isEnvBaseline and (self.terrain_name not in ["ComplexTerrain_NewTask","ComplexTerrain_NewEnv","ComplexTerrain_Sequential"]):
            self.cfg.rewards.scales.termination = 0. #-0.0
            self.cfg.rewards.scales.tracking_lin_vel = 1.0 # LVT
            self.cfg.rewards.scales.tracking_ang_vel = 0.5 # AVT

            self.cfg.rewards.scales.torques = -0.000
            self.cfg.rewards.scales.dof_vel = 0. #-0.
            self.cfg.rewards.scales.dof_acc = -2.5e-7 # JA
            self.cfg.rewards.scales.joint_power = -2e-5 # JP

            self.cfg.rewards.scales.base_height = -1. # BH

            self.cfg.rewards.scales.collision = 0.
            self.cfg.rewards.scales.feet_stumble = 0. #-0.0 

            self.cfg.rewards.scales.feet_clearance =  -1.5 # FC
            self.cfg.rewards.scales.feet_air_time =  1.  # FAT

            self.cfg.rewards.scales.power_distribution = -1e-7 

            self.cfg.rewards.scales.lin_vel_z = -2.0 # LVP
            self.cfg.rewards.scales.orientation = -0.2 # BOP
            self.cfg.rewards.scales.action_rate = -0.01 # AR
            self.cfg.rewards.scales.smoothness = -0.01 # AS
            self.cfg.rewards.scales.ang_vel_xy = -0.05 # AVP

            
            if self.case_id is not None:
                self.cfg.rewards.scales.raibert_heuristic = -10.
                
                self.cfg.rewards.scales.tracking_contacts_shaped_force = 10.
                self.cfg.rewards.scales.tracking_contacts_shaped_vel = 10.

                self.cfg.rewards.scales.feet_clearance_cmd_linear = -1.
                self.cfg.rewards.scales.feet_clearance = 0.
                self.cfg.rewards.scales.feet_air_time = 0.1

                self.cfg.rewards.scales.lin_vel_z = -0.5
                self.cfg.rewards.scales.ang_vel_xy = -0.01
                self.cfg.rewards.scales.action_rate = -0.01 # -0.01
                self.cfg.rewards.scales.smoothness = -0.005 # -0.01


            self.cfg.rewards.scales.stand_still = 0. #-0.


            self.cfg.rewards.only_positive_rewards = True


            self.reward_scales = class_to_dict(self.cfg.rewards.scales)
            self._prepare_reward_function()


    def get_imitation_reward_weight(self):
        if self.skills_descriptor_id is not None:
            return self.skills_descriptor_weight[2]
        else:
            return 0.

    def reset(self):
        """ Reset all robots"""

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.robot_pos_zero = torch.clone(self.root_states[:, :3])

        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
            
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def change_joint_order_inverse(self,joint_pos,joint_vel):
        """
        FL,FR,RL,RR --> FR,FL,RR,RL

        """
        FL = joint_pos[:,0:3]
        FR = joint_pos[:,3:6]
        RL = joint_pos[:,6:9]
        RR = joint_pos[:,9:12] 

        joint_pos = torch.cat((FR,FL,RR,RL),dim=-1)   

        FL_ = joint_vel[:,0:3]
        FR_ = joint_vel[:,3:6]
        RL_ = joint_vel[:,6:9]
        RR_ = joint_vel[:,9:12] 

        joint_vel = torch.cat((FR_,FL_,RR_,RL_),dim=-1)  

        return joint_pos, joint_vel
    
    def change_action_order(self,joint_pos):
        """
        FR,FL,RR,RL-->FL,FR,RL,RR

        """

        FR = joint_pos[:,0:3]
        FL = joint_pos[:,3:6]
        RR = joint_pos[:,6:9]
        RL = joint_pos[:,9:12] 

        joint_pos = torch.cat((FL,FR,RL,RR),dim=-1)   

        return joint_pos

    def change_action(self,joint_pos):
        """
        FL,FR,RL,RR --> FR,FL,RR,RL

        """

        FL = joint_pos[:,0:3]
        FR = joint_pos[:,3:6]
        RL = joint_pos[:,6:9]
        RR = joint_pos[:,9:12] 

        joint_pos = torch.cat((FR,FL,RR,RL),dim=-1)   

        return joint_pos
    
    def CompensateUncertainty(self, motor_commands):
        motor_commands = motor_commands.squeeze().cpu().numpy()

        self._q_b = self.dof_pos.squeeze().cpu().numpy()
        self._q_b_dot = self.dof_vel.squeeze().cpu().numpy()
        self._P_error=self._q_b-motor_commands
        self._delta_e = self._e_gain * np.clip(np.linalg.norm(self._P_error) ** 1. - self._eta, 0., np.inf) ** 2 * 2 * self._P_error
        
        if (self._motor_qc[0] == 1000.).all():
            print("RESET Q_C")
            #self._motor_qc=self._motor_qr
            self._motor_qd=motor_commands
            #self._motor_qc = motor_commands
            self._motor_qc=self._q_b
            self._F_y=self._P_error

        self._F_y_dot=-1*self.K1 @ self._F_y+self.K1 @ self._P_error
        self._F_y+=self._F_y_dot*0.005

        self._motor_qd_d=(motor_commands-self._motor_qd)/0.005
        self._motor_qd=motor_commands
        # motor_qr_d=self._motor_qd_d - 1. * self._P_error
        motor_qr_d = self._motor_qd_d - 1. * self._delta_e
        # motor_qr_d = self._motor_qd_d - 10 * self._F_y
        if (self._motor_qr_d[0] == 1000.).all():

            self._motor_qr_d=motor_qr_d
            self._motor_qr=self._motor_qd

        self._motor_qr_a=(motor_qr_d-self._motor_qr_d)/0.005
        self._motor_qr_d=motor_qr_d
        self._motor_qr+=self._motor_qr_d*0.005
        self._motor_s=self._q_b_dot-self._motor_qr_d



        M_matrix = pinocchio.crba(self._model,self._data,self._q_b)
        C_matrix = pinocchio.computeCoriolisMatrix(self._model,self._data,self._q_b,self._q_b_dot)
        G_matrix=pinocchio.computeGeneralizedGravity(self._model,self._data,self._q_b)
        dyn_compen=M_matrix@self._motor_qr_a+C_matrix@self._motor_qr_d+1*G_matrix
        # self._motor_qc_d=self._motor_qr_d+np.linalg.inv(self.Kp) @ self.Ki @ (self._motor_qr-self._motor_qc)\
        #                  +1 * np.linalg.inv(self.Kp) @ (dyn_compen-self.beta*self._P_error)
        self._d_hat_dot = (self._k * np.linalg.norm(self._motor_s) ** 2) / (self._k * np.linalg.norm(self._motor_s) + self._rho)
        self._d_hat += self._d_hat_dot * 0.005
        self._lambda = -(self._k * self._motor_s) / (self._k * np.linalg.norm(self._motor_s) + self._rho) * self._d_hat
        self._motor_qc_d = self._motor_qr_d + np.linalg.inv(self.Kp) @ self.Ki @ (self._motor_qr - self._motor_qc) \
                           + 1 * np.linalg.inv(self.Kp) @ (dyn_compen - self.beta * self._delta_e )
        self._motor_qc+=self._motor_qc_d*0.005
        self._F_y=self._motor_qc
        # print("self._motor_qc",self._motor_qc)
        return torch.from_numpy(self._motor_qc).unsqueeze(0).to(self.device)

  

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions = actions


        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        if self._enable_action_filter:
            self.actions = self._FilterAction(self.actions)
        
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # print("torques,",self.torques[0])
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
            

        else:
            policy_obs = self.obs_buf
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def get_observations(self):
        # print("self.cfg.env.include_history_steps",self.cfg.env.include_history_steps)
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))

        else:
            policy_obs = self.obs_buf
        return policy_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

     
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.contact_force_value = torch.norm(self.sensor_forces,p=2,dim=-1)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        r,p,y = get_euler_xyz(self.base_quat)
        self.base_rpy[:] = torch.cat((r.unsqueeze(1),p.unsqueeze(1),y.unsqueeze(1)),1)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.times_now = self.times_now + self.dt
        

        self.last_foot_pos[:] = self.foot_pos[:]
        self.foot_pos[:] = self.foot_positions_in_base_frame(self.dof_pos[:])

        self.last_actions_1[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        if self.terrain_name in ["ComplexTerrain_Sequential","ComplexTerrain_Sequential_Case_1"]: 
            self.reset_buf = self.root_states[:, 0]<0. or self.root_states[:, 0]>self.cfg.terrain.terrain_width*self.cfg.terrain.num_cols
        
            self.reset_buf |= self.root_states[:, 1]<0. or self.root_states[:, 1]>self.cfg.terrain.terrain_length*self.cfg.terrain.num_rows

            self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
            self.reset_buf |= self.time_out_buf
         
            self.reset_buf |= torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
            
        else:
            if "roll" in self.files:
                self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
                self.reset_buf = self.time_out_buf 
            else:
                self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

                self.reset_bad = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
                
                self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
                self.reset_buf |= self.time_out_buf

    def get_roll_frames(self,frames,env_num):
        
        if self.step_counter<=100:
            roll_threshold = torch.pi/3
        elif self.step_counter<=300:
            roll_threshold = torch.pi/2
        elif self.step_counter<=600:
            roll_threshold = 2.5*torch.pi/4
        elif self.step_counter<=800:
            roll_threshold = 3*torch.pi/4
        else:
            roll_threshold = 3*torch.pi    
        
        roll = frames[:,43]
        
        frames_list = torch.clone(frames)
        k=0
        for i in range(env_num):
            if torch.abs(roll[i])<roll_threshold:
                frames_list[k,:] = frames[i,:]
                k+=1
        k_num = k
        for j in range(k_num,env_num):      
            frames_list[j,:] = frames_list[ j%k_num ,:]        

        frames = torch.clone(frames_list)
        return frames
    
    def get_down_frames(self,frames,env_num):
        
        if self.step_counter<=100:
            roll_threshold = 2.5*torch.pi/4
        elif self.step_counter<=400:
            roll_threshold = torch.pi/2
        elif self.step_counter<=800:
            roll_threshold = 1.5*torch.pi/4
        elif self.step_counter<=1200:
            roll_threshold = torch.pi/4
        else:
            roll_threshold = 0.   
        
        roll = frames[:,43]
        
        frames_list = torch.clone(frames)
        k=0
        for i in range(env_num):
            if torch.abs(roll[i])>roll_threshold:
                frames_list[k,:] = frames[i,:]
                k+=1
        k_num = k
        for j in range(k_num,env_num):      
            frames_list[j,:] = frames_list[ j%k_num ,:]        

        frames = torch.clone(frames_list)
        return frames   

    def get_height_frames(self,frames,env_num):
        
        if self.step_counter<=100:
            height_threshold = 0.18
        elif self.step_counter<=400:
            height_threshold = 0.14
        elif self.step_counter<=800:
            height_threshold = 0.1
        else:
            height_threshold = 0.   
        
        height = frames[:,42]
        # print(torch.max(height),torch.min(height))
        
        frames_list = torch.clone(frames)
        k=0
        for i in range(env_num):
            if torch.abs(height[i])>height_threshold:
                frames_list[k,:] = frames[i,:]
                k+=1
        k_num = k
        for j in range(k_num,env_num):      
            frames_list[j,:] = frames_list[ j%k_num ,:]        

        frames = torch.clone(frames_list)
        return frames       

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # print("env_ids",env_ids)
        if self.isActionCorrection:
            self._F_pos[env_ids, :] *= 0.
            self._F_pos[env_ids, :] += 1000.
            
            self._F_y_dot *= 0.
            self._F_y *= 0.
            self._motor_qd_d *= 0.
            self._motor_qd *= 0.
            self._motor_qr_d *= 0.
            self._motor_qr_d += 1000.
            self._motor_qr_a *= 0.
            self._motor_s *= 0.
            self._motor_qc_d *= 0.
            self._motor_qc *= 0.
            self._motor_qc += 1000.
            self._q_b_dot_o*=0.
            self._ruo*=0.

            self._motor_qr *= 0.
            self._P_error *= 0.

            self.beta = 10.
            self._eta = 0.01
            self._e_gain = 0.1
            self._d_hat = 0.
            self._k = 0.000
            self._rho = 0.01

            self._flag_s = 0
            
            
            # print("env_ids",env_ids)
            # print("self.alpha",self.alpha)
            self.alpha[env_ids,:] = 1.

            self.desired_dof_pos[env_ids,:] = torch.ones(1, 12, device=self.device, requires_grad=False)*torch.tensor([0.,0.9,-1.8]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
            self.current_desired_dof_pos[env_ids,:] = torch.ones(1, 12, device=self.device, requires_grad=False)*torch.tensor([0.,0.9,-1.8]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
            

        self.torques[env_ids,:]*=0.

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        if self.terrain_name in ['ComplexTerrain_NewEnv']:

            x,y = self.init_terrain_id[0],self.init_terrain_id[1]
            self.env_origins[env_ids] = self.terrain_origins[x, y]

        elif self.terrain_name in ['ComplexTerrain_Sequential','ComplexTerrain_Sequential_Case_1']:
            # self.init_terrain_id = list(np.random.randint(low=[0,0],high=[self.cfg.terrain.num_cols,self.cfg.terrain.num_rows]))
            x,y = self.init_terrain_id[0],self.init_terrain_id[1]
            self.env_origins[env_ids] = self.terrain_origins[x, y]

        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # print("self.cfg.commands.curriculum",self.cfg.commands.curriculum)
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        choice = np.random.uniform(0, 1)


        if self.cfg.env.reference_state_initialization and choice < self.cfg.env.reference_state_initialization_prob:
            
            frames = self.amp_data.get_full_frame_batch(len(env_ids))
            

            self._reset_root_states_amp(env_ids, frames)
            self._reset_dofs_amp(env_ids, frames)
            # print("_reset_dofs_amp",env_ids)
        else:
            # print("_reset_dofs",env_ids)
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        if self.step_counter>self.curriculum_start_num: 
            if "forward_mass" in self.files:
                self.cfg.domain_rand.added_mass_range = [-2., 2.]
            elif "forward_noise" in self.files:
                self.cfg.noise.noise_level = 2.
                self.cfg.noise.noise_scales.dof_pos = 0.06
                self.cfg.noise.noise_scales.dof_vel = 3.
                self.cfg.noise.noise_scales.lin_vel = 0.2
                self.cfg.noise.noise_scales.ang_vel = 0.96
                self.cfg.noise.noise_scales.gravity = 0.1 

       

        if self._enable_action_filter:
            self._ResetActionFilter(env_ids)
            
        self.times_now[env_ids] = 0.

        # reset buffers
        self.last_actions_1[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
        self.gait_indices[env_ids] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.reward_list = []
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]

            rew = self.reward_functions[i]() * self.reward_scales[name]


            self.rew_buf += rew
            self.episode_sums[name] += rew
            
            self.reward_list.append(rew)
        self.reward_list = torch.stack((self.reward_list),dim=1).to(self.device)
        # print("self.reward_list",self.reward_list.size()) # torch.Size([4000, 15])


        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """

        if self.cfg.env.IsObservationEstimation:
            if self.isEnvBaseline or (("roll" in self.files) or ("stand_up" in self.files) or ("down_up" in self.files)):

                self.privileged_obs_buf = torch.cat((  
                                        self.projected_gravity, # 3
                                        self.base_ang_vel * self.obs_scales.ang_vel, # 3
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #6:18
                                        self.dof_vel * self.obs_scales.dof_vel, #18-30

                                        self.commands[:, :3] * self.commands_scale, # 30~33
                                        self.actions, # 33~45
                                        # self.base_lin_vel * self.obs_scales.lin_vel,
                                        # self.get_state_estimation()
                                        ),dim=-1)

                if self.cfg.env.usePD:
                    self.privileged_obs_buf = torch.cat((  
                        self.privileged_obs_buf,
                        self.randomized_p_gains*0.1,
                        self.randomized_d_gains
                        ),dim=-1)
                
                if self.cfg.env.state_est_dim>0:
                    self.privileged_obs_buf = torch.cat((  
                        self.privileged_obs_buf,
                        self.get_state_estimation()
                        ),dim=-1)
                  

            else:    
                self.privileged_obs_buf = torch.cat((  
                                        self.projected_gravity, # 3
                                        # self.commands[:, :3] * self.commands_scale,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #3-15
                                        self.dof_vel * self.obs_scales.dof_vel, #15-27
                                        self.actions, # 27-39
                                        self.base_ang_vel * self.obs_scales.ang_vel,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        ),dim=-1)


            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
            
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                # print("heights",heights)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            
         
            if self.isEnvBaseline or (("roll" in self.files) or ("stand_up" in self.files) or ("down_up" in self.files)):    
                foot_contact_forces = self.sensor_forces.reshape(self.num_envs,-1) # 12
                foot_contact_states = (self.contact_force_value>1.) # 4
                # friction_coefficients = self.friction_coeffs.view(self.num_envs,-1).to(self.device) # 1
                external_force = self.external_force # 3
                

            
                friction_coeffs_scale, friction_coeffs_shift = self.get_scale_shift(self.cfg.domain_rand.friction_range)
                # restitutions_scale, restitutions_shift = self.get_scale_shift(self.cfg.domain_rand.restitution_range)
                payloads_scale, payloads_shift = self.get_scale_shift(self.cfg.domain_rand.added_mass_range)
                # com_displacements_scale, com_displacements_shift = self.get_scale_shift(
                                                                                        # self.cfg.domain_rand.com_displacement_range)
                stiffness_multiplier_scale, stiffness_multiplier_shift = self.get_scale_shift(self.cfg.domain_rand.stiffness_multiplier_range)
                damping_multiplier_scale, damping_multiplier_shift = self.get_scale_shift(self.cfg.domain_rand.damping_multiplier_range)
                
                friction_coefficients = self.friction_coeffs.view(self.num_envs,-1).to(self.device) # 1

                p_mult = (self.randomized_p_gains/self.p_gains).clone()
                d_mult = (self.randomized_d_gains/self.d_gains).clone() 

                self.privileged_obs_buf = torch.cat(
                    (self.privileged_obs_buf,
                    0.1*foot_contact_forces, # 12
                    foot_contact_states,# 4
                    (friction_coefficients - friction_coeffs_shift) * friction_coeffs_scale,  # friction coeff 1
        
                    (self.payloads - payloads_shift) * payloads_scale,  # payload # 1 dim
                    (p_mult - stiffness_multiplier_shift) * stiffness_multiplier_scale,  # motor strength 12 dim
                    (d_mult - damping_multiplier_shift) * damping_multiplier_scale,  # motor strength 12 dim
                    # external_force
                    ), dim=-1)
            else:
                foot_contact_forces = self.contact_force_value # 4
                foot_contact_states = (self.contact_force_value>1.) # 4
                friction_coefficients = self.friction_coeffs.view(self.num_envs,-1).to(self.device) # 1
                external_force = self.external_force # 3
                
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, 
                                                 0.1*foot_contact_forces,
                                                 foot_contact_states,
                                                 friction_coefficients,
                                                 external_force
                                                 ), dim=-1)

        else:    
            base_lin_vel = self.base_lin_vel[:, :2]
            # print(base_lin_vel)
            base_ang_vel = self.base_ang_vel

            self.privileged_obs_buf = torch.cat((  
                                        self.projected_gravity, # 3
                                        # self.commands[:, :3] * self.commands_scale,
                                        (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #3-15
                                        self.dof_vel * self.obs_scales.dof_vel, #15-27
                                        self.actions, # 27-39
                                        base_lin_vel * self.obs_scales.lin_vel,
                                        base_ang_vel * self.obs_scales.ang_vel,
                                        ),dim=-1)


            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
           
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)


        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.clone(self.privileged_obs_buf[:,:self.num_obs])

    def get_state_estimation(self):
        foot_pos = self.foot_pos
        foot_height = torch.stack([self.foot_pos[:,3*i+2] for i in range(4)],dim=1)
        foot_contact = (self.contact_force_value>1.) # 4


        body_height = (self.root_states[:, 2] - self.measured_heights[:,self.measured_heights.size()[1]//2]).unsqueeze(1)
        
        body_lin_vel = self.base_lin_vel * self.obs_scales.lin_vel
        
        if self.cfg.env.state_est_dim == 3:
            return torch.cat((  
                            body_lin_vel, #0~3
                            # body_height,
                            # foot_contact,
                            # foot_height
                            # foot_pos
                            ),dim=-1)
        elif self.cfg.env.state_est_dim == 7:
            return torch.cat((  
                            body_lin_vel, #0~3
                            # body_height,
                            # foot_contact,
                            foot_height
                            # foot_pos
                            ),dim=-1)
        elif self.cfg.env.state_est_dim == 15:
            return torch.cat((  
                            body_lin_vel, #0~3
                            # body_height,
                            # foot_contact,
                            # foot_height
                            foot_pos
                            ),dim=-1)
        
        elif self.cfg.env.state_est_dim == 12:
            return torch.cat((  
                            body_lin_vel, #0~3
                            body_height,
                            foot_contact,
                            foot_height
                            # foot_pos
                            ),dim=-1)
        
        elif self.cfg.env.state_est_dim == 20:
            return torch.cat((  
                            body_lin_vel, #0~3
                            body_height,
                            foot_contact,
                            # foot_height
                            foot_pos
                            ),dim=-1)  

    def get_scale_shift(self,range):
        scale = 2. / (range[1] - range[0])
        shift = (range[1] + range[0]) / 2.
        return scale, shift

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # z_pos = self.root_states[:, 2:3]
        z_pos = self.root_states[:, 2:3] - self.env_origins[:,2:3]

        # self.root_states[env_ids, :3] += self.env_origins[env_ids]
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)
        # return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel), dim=-1)
     
    def create_sim(self):
        """ Creates simulation, terrain and environments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        mesh_type = self.cfg.terrain.mesh_type
        
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):

        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            payloads = np.random.uniform(rng[0], rng[1])
            props[0].mass += payloads

            self.payloads = payloads*torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # print(self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt))
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        
        self._resample_commands(env_ids)
        self._step_contact_targets()
        
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # print(self.common_step_counter) 
        self.external_force = torch_rand_float(-0.0001, 0.0001, (self.num_envs, 3), device=self.device) # lin vel x/y 
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
   
            self._push_robots()

        if self.terrain_name=="Pad":
            self._push_robot_feet(feet_height=-0.28,min_rand=-3.,max_rand=-2.5)
        elif self.terrain_name=="Sand":
            self._push_robot_feet(feet_height=-0.26,min_rand=-2.,max_rand=-1.5)   
        elif self.terrain_name=="Grass and Mud":                     
            self._push_robot_feet(feet_height=-0.24,min_rand=-2.5,max_rand=-2.)  
        elif self.terrain_name=="WindySand":
            self._push_robot_feet(feet_height=-0.28,min_rand=-1.,max_rand=-1.5)
            self._push_robot_body(resistance_low=-0.12,resistance_high=-0.03)  
        elif self.terrain_name=="Brushwood": 
            self._push_robot_body(resistance_low=-0.1,resistance_high=-0.01)  
        elif self.terrain_name=="WindyLand": 
            self._push_robot_body(resistance_low=-0.15,resistance_high=-0.05)  


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if self.terrain_name in ["ComplexTerrain_Sequential","ComplexTerrain_Sequential_Case_1"]: 
            self.commands[env_ids, 0] = self.command_ranges["lin_vel_x"][0]*torch.ones(size=(len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = self.command_ranges["lin_vel_y"][0]*torch.ones(size=(len(env_ids), 1), device=self.device).squeeze(1)
            
            self.commands[env_ids, 3] = self.command_ranges["lin_vel_z"][0]*torch.ones(size=(len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = self.command_ranges["ang_vel_yaw"][0]*torch.ones(size=(len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 4] = self.command_ranges["ang_vel_roll"][0]*torch.ones(size=(len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 5] = self.command_ranges["ang_vel_pitch"][0]*torch.ones(size=(len(env_ids), 1), device=self.device).squeeze(1)
        
        else:    
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            

            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["lin_vel_z"][0], self.command_ranges["lin_vel_z"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
   
            self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["limit_gait_frequency"][0], 
                                                         self.command_ranges["limit_gait_frequency"][1], 
                                                         (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 5] = torch_rand_float(self.command_ranges["limit_gait_phase"][0], 
                                                self.command_ranges["limit_gait_phase"][1], 
                                                (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 6] = torch_rand_float(self.command_ranges["limit_gait_offset"][0], 
                                                self.command_ranges["limit_gait_offset"][1], 
                                                (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 7] = torch_rand_float(self.command_ranges["limit_gait_bound"][0], 
                                                self.command_ranges["limit_gait_bound"][1], 
                                                         (len(env_ids), 1), device=self.device).squeeze(1)
            
            category = self.case_id
            if category == 0: #"pronk":  # pronking
                self.commands[env_ids, 5] = (self.commands[env_ids, 5] / 2 - 0.25) % 1
                self.commands[env_ids, 6] = (self.commands[env_ids, 6] / 2 - 0.25) % 1
                self.commands[env_ids, 7] = (self.commands[env_ids, 7] / 2 - 0.25) % 1
            elif category == 1:# "trot":  # trotting
                self.commands[env_ids, 5] = self.commands[env_ids, 5] / 2 + 0.25
                self.commands[env_ids, 6] = 0
                self.commands[env_ids, 7] = 0
            elif category == 2: #"pace":  # pacing
                self.commands[env_ids, 5] = 0
                self.commands[env_ids, 6] = self.commands[env_ids, 6] / 2 + 0.25
                self.commands[env_ids, 7] = 0
            elif category == 3: #"bound":  # bounding
                self.commands[env_ids, 5] = 0
                self.commands[env_ids, 6] = 0
                self.commands[env_ids, 7] = self.commands[env_ids, 7] / 2 + 0.25
            

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":


            self.desired_pos = actions_scaled + self.default_dof_pos # 12 dim: FR 0~3 FL 3~6 RR 6~9 RL 9~12

            if "roll" in self.files:
                self.desired_pos = actions_scaled + self.default_lie_dof_pos
            
 

            if self.isActionCorrection and (abs(self.command_ranges["lin_vel_x"][0])>0.1 or abs(self.command_ranges["lin_vel_y"][0])>0.1 or abs(self.command_ranges["ang_vel_yaw"][0])>0.1):
               
                self.desired_pos = self.CompensateUncertainty(self.desired_pos)

            torques = p_gains*(self.desired_pos - self.dof_pos) - d_gains*self.dof_vel




        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled


        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def get_desired_pos(self):
        return self.desired_pos      

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.

        if "stand_up" in self.files:
            # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.1, 2., (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = self.default_standup_dof_pos * torch_rand_float(0.75, 1.25, (len(env_ids), self.num_dof), device=self.device)

            self.dof_vel[env_ids] = torch_rand_float(-3., 3., (len(env_ids), self.num_dof), device=self.device)

        self.dof_state[:,0] = self.dof_pos.view(-1)
        self.dof_state[:,1] = self.dof_vel.view(-1)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """

        self.dof_pos[env_ids] = self.amp_data.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = self.amp_data.get_joint_vel_batch(frames)
        # print("env_ids_1",env_ids)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # print("env_ids_2",env_ids)

        self.dof_state[:,0] = self.dof_pos.view(-1)
        self.dof_state[:,1] = self.dof_vel.view(-1)

        # print("_reset_dofs_amp",self.dof_state.size())

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))



    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
                     
            if ("roll" not in self.files) and ("stand_up" not in self.files):
                # self.root_states[env_ids, :2] += torch_rand_float(-0.3, 0.3, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
                self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

    
        if "roll" in self.files:
            
            # roll task: training
            roll = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
            pitch = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
            yaw = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
            
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            self.root_states[env_ids, 3:7] = quat.squeeze()

        elif "get_down" in self.files:
            envs_num = len(env_ids)
            
            if self.step_counter<self.curriculum_start_num: 
                roll = torch_rand_float(-torch.pi, -torch.pi/2, (len(env_ids)//2, 1), device=self.device)
                roll_ = torch_rand_float(torch.pi/2, torch.pi, (envs_num - len(env_ids)//2, 1), device=self.device)
                roll =torch.cat((roll, roll_), dim=0)
            else:
                roll = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)

            pitch = torch_rand_float(-torch.pi/2, torch.pi/2, (len(env_ids), 1), device=self.device)
            yaw = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
            
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            self.root_states[env_ids, 3:7] = quat.squeeze()    
        
        elif "stand_up" in self.files:
            envs_num = len(env_ids)
            roll = torch_rand_float(-0.15*torch.pi, 0.15*torch.pi, (len(env_ids), 1), device=self.device)

            pitch = torch_rand_float(-0.15*torch.pi, 0.15*torch.pi, (len(env_ids), 1), device=self.device)
            yaw = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
            
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            self.root_states[env_ids, 3:7] = quat.squeeze()


        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environment ids
        """

        self.root_states[env_ids] = self.base_init_state

     

        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # print(self.root_states[env_ids, :3])
        self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        

        root_orn = self.amp_data.get_rpy_batch(frames)
        root_orn[:,2] = 0.
        root_orn = quat_from_euler_xyz(root_orn[:,0], root_orn[:,1], root_orn[:,2])
        self.root_states[env_ids, 3:7] = root_orn

        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, self.amp_data.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, self.amp_data.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    #  gymtorch.unwrap_tensor(torch.cat((self.root_states,self.root_states_ball),dim=0)),
                                                    gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots_hard(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
     
        self.external_force[:,:3] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device) # lin vel x/y
        
        self.root_states[:, 7:10] = self.external_force

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        self.dof_vel[:] = torch_rand_float(-5.,5., (self.num_envs, self.num_dof), device=self.device) 
      
        self.dof_state[:,1] = self.dof_vel.view(-1)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        
        self.external_force[:,:2] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y


        self.external_force_true = self.external_force


        self.root_states[:, 7:10] = self.external_force
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    

    def _push_robot_body(self,resistance_low=-0.1,resistance_high=-0.01):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        self.root_states[:, 7:9] += self.root_states[:, 7:9]*torch_rand_float(resistance_low, resistance_high, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # zhy
    def _push_robot_feet(self,feet_height,min_rand,max_rand):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        # print("foot_pos", foot_pos)
        flag = torch.zeros(size=foot_pos.size()).to(self.device)
        for env_num in range(self.num_envs):
            for i in range(4):
                if foot_pos[env_num,3*i+2]<=feet_height:
                    flag[env_num,3*i:3*i+2] = 1.

        self.dof_vel[:] += self.dof_vel[:]*torch_rand_float(min_rand,max_rand, (self.num_envs, self.num_dof), device=self.device) * flag
        # print("dof_state", self.dof_state.size())
        self.dof_state[:,1] = self.dof_vel.view(-1)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return

     
        if self.cfg.env.IsObservationEstimation:
            if self.isEnvBaseline or self.terrain_name in ['ComplexTerrain_NewEnv_opt']:
         
                if not self.init_done:
                # don't change on initial reset
                    return
                
                distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
                # robots that walked far enough progress to harder terains
 
                move_up = distance > self.terrain.env_length / 5.

                # robots that walked less than half of their required distance go to simpler terrains
                move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
               
                self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
    
                self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                        torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                        torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
              
                self.env_origins[env_ids] = self.terrain_origins[self.terrain_types[env_ids],self.terrain_levels[env_ids]]
                
         

            else:    
                most_hard_terrain = self.cfg.terrain.num_rows
                if self.step_counter<0.5*self.curriculum_start_num:
                    current_terrain_ind = 1 
                elif self.step_counter>=0.5*self.curriculum_start_num and  self.step_counter<1.5*self.curriculum_start_num:
                    current_terrain_ind = int(1+(self.step_counter-0.5*self.curriculum_start_num)*(most_hard_terrain-1)/(self.curriculum_start_num))  
                else:        
                    current_terrain_ind = most_hard_terrain
                self.terrain_levels = torch.randint(0,current_terrain_ind, (self.num_envs,), device=self.device)
                self.env_origins[env_ids] = self.terrain_origins[self.terrain_types[env_ids], self.terrain_levels[env_ids]]            
        else:
            if self.step_counter>=self.curriculum_start_num: 
                self.terrain_levels = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,), device=self.device)
                self.env_origins[env_ids] = self.terrain_origins[self.terrain_types[env_ids], self.terrain_levels[env_ids]]
    

    def set_step_counter(self,step_counter):
        self.step_counter = step_counter

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        if self.terrain_name in ["ComplexTerrain_NewEnv_opt"]:
            if torch.mean(self.episode_sums["tracking_vel_all"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_vel_all"]:
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.25, -self.cfg.commands.max_curriculum_x, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.25, 0., self.cfg.commands.max_curriculum_x)
                
        else:    
            if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.25, -self.cfg.commands.max_curriculum_x, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.25, 0., self.cfg.commands.max_curriculum_x)
 
                    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

     
        if  self.cfg.env.IsObservationEstimation:
            if self.isEnvBaseline or (("roll" in self.files) or ("stand_up" in self.files) or ("down_up" in self.files)):
            # if self.isEnvBaseline or ("roll" or "stand_up" in self.files):
                noise_vec[:3] = 0. # commands
                noise_vec[3:6] = noise_scales.gravity * noise_level
                noise_vec[6:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                noise_vec[18:30] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                noise_vec[30:42] = 0. # previous actions

                noise_vec[42:45] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                noise_vec[45:48] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                # noise_vec[45:48+9] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
                
                
                if self.cfg.env.state_est_dim == 12:
                    noise_vec[48:49] = noise_scales.body_height * noise_level
                    noise_vec[49:53] = noise_scales.foot_contact * noise_level
                    noise_vec[53:57] = noise_scales.foot_height * noise_level
                elif self.cfg.env.state_est_dim == 20:
                    noise_vec[48:49] = noise_scales.body_height * noise_level
                    noise_vec[49:53] = noise_scales.foot_contact * noise_level
                    noise_vec[53:65] = noise_scales.foot_pos * noise_level

                if self.cfg.terrain.measure_heights:
                    # noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
                    noise_vec[45+self.cfg.env.state_est_dim:235+self.cfg.env.state_est_dim] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
            else:            
                noise_vec[:3] = noise_scales.gravity * noise_level
                noise_vec[3:15] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
                noise_vec[15:27] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
                noise_vec[27:39] = 0. # previous actions

                noise_vec[39:42] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
                noise_vec[42:45] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel

                if self.cfg.terrain.measure_heights:
                    noise_vec[45:232] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        else:
            noise_vec[:3] = noise_scales.gravity * noise_level
            noise_vec[3:15] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
            noise_vec[15:27] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
            noise_vec[27:39] = 0. # previous actions

            noise_vec[39:41] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
            noise_vec[41:44] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel

            if self.cfg.terrain.measure_heights:
                noise_vec[44:231] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements

        self.noise_vec=noise_vec
        return noise_vec

    #zhy----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        
      

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
 
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]
        
    

        self.contact_force_value = torch.norm(self.sensor_forces,p=2,dim=-1)

        self.times_now = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions_1 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_foot_pos = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_pos = self.foot_positions_in_base_frame(self.dof_pos)

        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        r,p,y = get_euler_xyz(self.base_quat)
        self.base_rpy = torch.cat((r.unsqueeze(1),p.unsqueeze(1),y.unsqueeze(1)),1 )

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # print("="*50)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            # print(name)
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_lie_dof_pos = torch.tensor([0.,1.56,-2.7]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        
        self.default_standup_dof_pos = torch.tensor([-0.8026,  3.0904, -2.6928,  
                                                     0.1681,  2.1708, -2.7001,  
                                                     0.0261,  2.2151,-2.6953,  
                                                     0.2497,  2.2081, -2.6958], dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        
        self.default_lie_orientation = torch.tensor([0.,0.,-1.], dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        self.default_stand_dof_pos = torch.tensor([0.,0.9,-1.8]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
     
        self.default_lie_orientation_down = torch.tensor([0.,0.,1.], dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        
 
        self.default_joint_pos = torch.tensor([0.,0.9,-1.8]*4, dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(0)
        
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        
        
        if self.leg_id is not None:
            if "stand_up" in self.files:
                self.default_joint_pos[:,3*self.leg_id] = 0.
                self.default_joint_pos[:,3*self.leg_id+1] = 0.6
                self.default_joint_pos[:,3*self.leg_id+2] = -2.4
                self.default_joint_pos = self.default_joint_pos * torch_rand_float(0.9, 1.1, (1,self.num_dof), device=self.device)
            else:
                self.default_joint_pos[:,3*self.leg_id] = 0.
                self.default_joint_pos[:,3*self.leg_id+1] = 1.56
                self.default_joint_pos[:,3*self.leg_id+2] = -2.7
                self.default_joint_pos = self.default_joint_pos * torch_rand_float(0.9, 1.1, (1,self.num_dof), device=self.device)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)

    def compute_randomized_gains(self, num_envs):
        p_mult = ((
            self.cfg.domain_rand.stiffness_multiplier_range[0] -
            self.cfg.domain_rand.stiffness_multiplier_range[1]) *
            torch.rand(num_envs, self.num_actions, device=self.device) +
            self.cfg.domain_rand.stiffness_multiplier_range[1]).float()
        d_mult = ((
            self.cfg.domain_rand.damping_multiplier_range[0] -
            self.cfg.domain_rand.damping_multiplier_range[1]) *
            torch.rand(num_envs, self.num_actions, device=self.device) +
            self.cfg.domain_rand.damping_multiplier_range[1]).float()
        
        return p_mult * self.p_gains, d_mult * self.d_gains


    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip

        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        # print(foot_positions[0])
        # foot_positions = foot_positions.view(self.num_envs,-1)
        # print(foot_positions[0])
        # input()        
        return foot_positions
    

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                # if self.skills_descriptor_id is not None:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)

        # self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_cols, self.terrain.tot_rows).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
       
        # self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_cols, self.terrain.tot_rows).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        sensor_pose = gymapi.Transform()
        for name in feet_names:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False # for example gravity
            sensor_options.enable_constraint_solver_forces = True # for example contacts
            sensor_options.use_world_frame = True # report forces in world frame (easier to get vertical components)
            index = self.gym.find_asset_rigid_body_index(robot_asset, name)
            self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
    
        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

       

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            
            if self.cfg.terrain.curriculum:
                self.terrain_levels = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
                if self.isEnvBaseline:
                    max_init_level = self.cfg.terrain.max_init_terrain_level
                    if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
                    self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)

            else:
                self.terrain_levels = torch.randint(0, self.cfg.terrain.num_rows, (self.num_envs,), device=self.device)

            
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            
            self.max_terrain_level = self.cfg.terrain.num_rows
            
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            # self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.env_origins[:] = self.terrain_origins[self.terrain_types, self.terrain_levels]

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt

        # print("self.dt",self.dt) # 0.03 = 0.005*6

        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        # print(self.dt)
        # print(self.cfg.domain_rand.push_interval)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    

    #######################################
    ## roll
    def _reward_BaseUprightness(self):
        # print(self.projected_gravity[:, 2:3])
        return torch.sum((1. - self.projected_gravity[:, 2:3]), dim=1)

    #######################################
    
    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) # *(self.root_states[:, 0]<self.commands[:, 6])
    

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):

        base_height = self.root_states[:, 2] - self.measured_heights[:,self.measured_heights.size()[1]//2]
        
        return torch.abs(base_height - self.cfg.rewards.base_height_target)

    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_joint_power(self):
        # Penalize dof velocities
        return torch.sum(torch.abs(self.torques*self.dof_vel), dim=1)
    
    def _reward_power_distribution(self):
        # Penalize dof velocities
        # return torch.var(torch.abs(self.torques*self.dof_vel),dim=1)
        return torch.var(torch.square(self.torques*self.dof_vel),dim=1) # too large
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_actions_1), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)


    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))
        return reward / 4.

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))
        return reward / 4.
    
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.root_states[:,:3].unsqueeze(1) #self.env.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])
        


        desired_ys_nom = torch.tensor([0.1630,  -0.1586,
                                     0.1621, -0.1577], device=self.device).unsqueeze(0)

        desired_stance_length = 0.42
        desired_xs_nom = torch.tensor([0.1811,  0.1811,
                                        -0.2427, -0.2427], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        
        frequencies = self.commands[:, 4]

        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]

        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        self.footsteps_in_body_frame=footsteps_in_body_frame
        self.desired_footsteps_body_frame=desired_footsteps_body_frame
        
        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        
        return reward
    
    def _step_contact_targets(self):
    
        frequencies = self.commands[:, 4]

        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR        

    
    def _reward_feet_clearance_cmd_linear(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)# - reference_heights
        # target_height = self.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        target_height = 0.24*torch.ones(size=(self.num_envs,4),device=self.device) * phases + 0.02 # offset for foot radius 2cm
        
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip
    

    def _reward_footswing_height_tracking(self):
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)

        footsteps_in_body_frame = self.foot_pos.reshape((self.num_envs,4,3))
        desired_foot_height = 0.2*torch.ones(size=(self.num_envs,4),device=self.device) #57 test2
        
        foot_height = footsteps_in_body_frame[:, :, 2] # - reference_heights

        base_height = (self.root_states[:, 2] - self.measured_heights[:,self.measured_heights.size()[1]//2]).unsqueeze(1)
        
        desired_foot_height = desired_foot_height*phases
        rew_foot_clearance = torch.abs((base_height - torch.abs(foot_height)) - desired_foot_height) * (1 - self.desired_contact_states)
        
        return torch.sum(rew_foot_clearance, dim=1)
    

    def _reward_gait(self):
        a_pre = self.actions*self.cfg.control.action_scale+self.default_dof_pos
        a_label = (self.actions*self.cfg.control.action_scale+self.default_dof_pos).detach()
         
        if self.case_id ==0:
            a_label = torch.concat((a_label[:,3:6],a_label[:,0:3],a_label[:,9:12],a_label[:,6:9]),dim=-1)
        elif self.case_id ==1:
            a_label = torch.concat((a_label[:,6:9],a_label[:,9:12],a_label[:,0:3],a_label[:,3:6]),dim=-1)
        elif self.case_id ==2:
            a_label = torch.concat((a_label[:,9:12],a_label[:,6:9],a_label[:,3:6],a_label[:,0:3]),dim=-1)
        
        loss_gait = torch.mean(torch.abs(torch.abs(a_pre) - torch.abs(a_label)),dim=-1)
             
        return loss_gait


    
    def _reward_jump(self):
        body_height = self.root_states[:, :3] - self.env_origins[:,:3]
        jump_height_target = self.commands[:, 6:9]


        pos_error = torch.sum(torch.abs(body_height- jump_height_target),dim=-1)
     
        return torch.exp(-pos_error/self.cfg.rewards.tracking_sigma)*(self.base_lin_vel[:, 2])*(self.base_lin_vel[:, 2]>0.)

    def _reward_jump1(self):
        body_height = self.root_states[:, :3] - self.env_origins[:,:3]

        jump_height_target = self.commands[:, 6:9]
        pos_error = torch.sum(torch.abs(body_height[:,2].unsqueeze(-1) - jump_height_target[:,2].unsqueeze(-1)),dim=-1)
     
        return torch.exp(-pos_error/self.cfg.rewards.tracking_sigma)*(self.base_lin_vel[:, 2])*(self.base_lin_vel[:, 2]>0.)


    def _reward_tracking_ang_vel(self):

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_lin_vel(self):
 
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_vel_all(self):
        # Tracking of linear velocity commands (xy axes)
        x = self.commands[:, 0].unsqueeze(-1)
        y = self.commands[:, 1].unsqueeze(-1)
        z = self.commands[:, 3].unsqueeze(-1)

        roll = self.commands[:, 4].unsqueeze(-1)
        pitch = self.commands[:, 5].unsqueeze(-1) # float(torch.cat((self.commands[:, 4],self.commands[:, 5],self.commands[:, 2]),dim=-1)*self.dt*N)
        yaw = self.commands[:, 2].unsqueeze(-1) # float(torch.cat((self.commands[:, 4],self.commands[:, 5],self.commands[:, 2]),dim=-1)*self.dt*N)

        commands = torch.cat((x,y,z,roll,pitch,yaw),dim=-1)
        current_vel = torch.cat((self.base_lin_vel,self.base_ang_vel),dim=-1)

        lin_vel_error = torch.sum(torch.abs(commands - current_vel), dim=1)
        
        return torch.exp(-0.3*lin_vel_error)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt

        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_tracking_ang_vel_all(self):
        # Tracking of roll angular velocity commands
        ang_vel_error = torch.sum(torch.abs(self.commands[:, 2].unsqueeze(-1) - self.base_ang_vel[:, 2].unsqueeze(-1)))+ \
                        torch.sum(torch.abs(self.commands[:, 4:6] - self.base_ang_vel[:, :2]), dim=1)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) 

    def _reward_roll_vel_lie(self):
        # Tracking of roll angular velocity commands
        # print(self.base_ang_vel[:, :1])
        ang_vel_error = torch.sum(torch.abs(self.commands[:, 4:5] - torch.abs(self.base_ang_vel[:, :1])), dim=1)
        # print(ang_vel_error.size())
        r = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) * (torch.abs(self.base_rpy[:, :1])>0.5*torch.pi).squeeze()
        return r
    
    def _reward_roll_vel(self):
        # Tracking of roll angular velocity commands
        # print(self.base_ang_vel[:, :1])
        ang_vel_error = torch.sum(torch.abs(self.commands[:, 4:5] - self.base_ang_vel[:, :1]), dim=1)
        r = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) 
        return r
    
    def _reward_yaw_vel(self):
        # Tracking of roll angular velocity commands
        # print(self.base_ang_vel[:, :1])
        ang_vel_error = torch.sum(torch.abs(0. - self.base_ang_vel[:, 2:3]), dim=1)
        r = torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) 
        return r
    
    def _reward_pitch_vel(self):
        # print(self.base_ang_vel[:, 1:2])
        # Tracking of pitch angular velocity commands
        ang_vel_error = torch.sum(torch.abs(self.commands[:, 5:6] - self.base_ang_vel[:, 1:2]), dim=1)
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma) 
    

    def _reward_dof_vel_zeros(self):
        # Penalize dof velocities too close to the zeros
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        dof_vel_error = torch.sum((torch.abs(self.dof_vel) - 0.).clip(min=0., max=1.), dim=1)  
        dof_vel_error = dof_vel_error * (torch.abs(self.base_rpy[:,0])>0.75)  
        return dof_vel_error
    
    def joint_vel_to_foot_vel(self,dof_pos, dof_vel):
        foot_vel = torch.zeros_like(dof_pos)
        for i in range(4):
            l_hip_sign = (-1)**(i)
            theta = dof_pos[:, i * 3:i * 3 + 3].clone()
            
            theta[:,0]= -1.*theta[:,0]
            theta[:,1] = -1.*theta[:,1]

            angle_dot = dof_vel[:, i * 3:i * 3 + 3]
            a0 = 0.08505 * l_hip_sign
            a1 = 0.2  
            a2 = 0.2  
            
            # theta= L0 L1 L2
            s0 = torch.sin(theta[:,0])# / 180.0 * math.pi)
            c0 = torch.cos(theta[:,0])# / 180.0 * math.pi)
            s1 = torch.sin(theta[:,1])# / 180.0 * math.pi)
            c1 = torch.cos(theta[:,1])# / 180.0 * math.pi)

            s12 = torch.sin(theta[:,1]  + theta[:,2] )
            c12 = torch.cos(theta[:,1] + theta[:,2] )

            transJ = torch.zeros((self.num_envs,3, 3)).to(self.device)
            transJ[:,0, 0] = 0
            transJ[:, 0, 1] = -a0 * c1 - a2 * c12
            transJ[:, 0, 2] = -a2 * c12

            transJ[:,1, 0] = c0 * (a0 + a1 * c1 + a2 * c12)
            transJ[:,1, 1] = -s0 * (a1 * s1 + a2 * s12)
            transJ[:,1, 2] = - a2 * s0 * s12

            transJ[:,2, 0] = s0 * (a0 + a1 * c1 + a2 * c12)
            transJ[:,2, 1] = c0 * (a1 * s1 + a2 * s12)
            transJ[:,2, 2] = a2 * c0 * s12
    
            foot_dot = torch.bmm(transJ.transpose(2,1), angle_dot.unsqueeze(-1))
            # print("foot_dot",foot_dot.size())
            foot_vel[:, i * 3:i * 3 + 3].copy_(foot_dot.squeeze())
        return foot_vel
    
    def _reward_feet_clearance(self):
        desired_foot_height = 0.26
        base_height = (self.root_states[:, 2] - self.measured_heights[:,self.measured_heights.size()[1]//2]).unsqueeze(1)
        foot_z_pos = torch.abs((self.foot_pos.view(self.num_envs,4,3)[:,:,2]).view(self.num_envs,-1))
        foot_vel = (torch.sqrt(torch.sum(torch.square(self.foot_velocities[:, :, 0:2]),dim=-1)) ).view(self.num_envs, -1)
    
        rew_foot_clearance = torch.square((base_height - foot_z_pos)-desired_foot_height) * foot_vel
        # print("rew_foot_clearance",rew_foot_clearance.size())
        return torch.sum(rew_foot_clearance, dim=1)
    
    
    def _reward_feet_distance(self):
        foot_y_pos = torch.abs((self.foot_pos.view(self.num_envs,4,3)[:,:,1]).view(self.num_envs,-1))
        return torch.sum((foot_y_pos - 0.),dim=-1)    

    def _reward_lie(self):
        lie_dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_lie_dof_pos), dim=1) * (torch.abs(self.base_rpy[:,0])<=0.3)  
        return torch.exp(-lie_dof_pos_error/self.cfg.rewards.tracking_sigma) 

    def _reward_stand(self):
        # stand_dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_stand_dof_pos), dim=1) # * (torch.abs(self.base_rpy[:,0])<=0.3)  
        stand_dof_pos_error = torch.sum(torch.abs(self.dof_pos[:,::3] - 0.), dim=1) # * (torch.abs(self.base_rpy[:,0])<=0.3)  
        return torch.exp(-stand_dof_pos_error/self.cfg.rewards.tracking_sigma)     

    def _reward_lie_orientation(self):

        orientation_error = torch.sum(torch.abs(self.projected_gravity[:,2].unsqueeze(-1) - self.default_lie_orientation[:,2].unsqueeze(-1)), dim=1)

        return torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma) 
    
    def _reward_lie_orientation_rpy(self):
        orientation_error = torch.sum(torch.abs(self.base_rpy - 0.), dim=1)
        return torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma)     

    def _reward_lie_orientation_y(self):
        orientation_error = torch.sum(torch.abs(torch.abs(self.base_rpy[:,0].unsqueeze(-1)) - torch.pi) + torch.abs(self.base_rpy[:,2].unsqueeze(-1) - 0.), dim=1)

        return torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma)   
    
    def _reward_lie_down(self):
        lie_dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_lie_dof_pos), dim=1) * (torch.abs(self.base_rpy[:,0])>2.7)  
        return torch.exp(-lie_dof_pos_error/self.cfg.rewards.tracking_sigma)   

    def _reward_lie_pos(self):
        body_height = self.root_states[:, :3] - self.env_origins[:,:3]
        # print(body_height)
        lie_dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_lie_dof_pos), dim=1) * (torch.abs(body_height[:,2])>0.4)  
        r = torch.exp(-0.1*lie_dof_pos_error)   
        # print(r)
        return r
    
    def _reward_lie_pos_roll(self):
        # print(body_height)
        lie_dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_lie_dof_pos), dim=1) 
        r =  torch.exp(-0.1*lie_dof_pos_error) * torch.abs(self.base_ang_vel[:,0]) 
        # print(r)
        return r
    
    def _reward_lie_pos_pitch(self):
        # print(body_height)
        lie_dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.default_lie_dof_pos), dim=1) 
        r =  torch.exp(-0.1*lie_dof_pos_error) * torch.abs(self.base_ang_vel[:,1]) 
        # print(r)
        return r
    
    def _reward_lie_orientation_down(self):
        # Penalize non flat base orientation
        # orientation_error = torch.sum(torch.abs(torch.abs(self.base_rpy[:,:2]) - self.default_lie_orientation_down), dim=1)
        orientation_error = torch.sum(torch.abs(self.projected_gravity - self.default_lie_orientation_down), dim=1)
        return torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma)    

    def _reward_tracking_joint_vel(self,current_joint_vel,desired_joint_vel):
        error = torch.sum(torch.square(current_joint_vel - desired_joint_vel), dim=1)
        # print("_reward_tracking_joint_vel",error)
        return torch.exp(-error*0.001)         
    
    def _reward_joint_pos(self):
        error = torch.sum(torch.square((self.dof_pos[:,3*self.leg_id:3*self.leg_id+3] - self.default_joint_pos[:,3*self.leg_id:3*self.leg_id+3])), dim=1)
        # print("_reward_tracking_joint_vel",error)
        return torch.exp(-5*error) 

    def _reward_foot_contact(self):
        error = torch.mean(torch.abs(self.contact_force_value[:,self.leg_id].unsqueeze(-1)), dim=1)
        return error          

    def _reward_foot_full_contact(self):
        r = torch.sum((self.contact_force_value>1.).float(), dim=1) # *(self.root_states[:, 0]<self.commands[:, 6])
        return r
    
    def _reward_tracking_joint(self,current_joint,desired_joint):
        error = torch.sum(torch.square(current_joint - desired_joint), dim=1)
        # print("_reward_tracking_joint",error)
        return torch.exp(-error*0.1)    


    def _reward_tracking_rp_vel(self,current_rp_vel,desired_rp_vel):
        error = torch.sum(torch.square(current_rp_vel - desired_rp_vel), dim=1)
        # print("_reward_tracking_rp_vel",error)
        return torch.exp(-error*0.1)      

    def _reward_tracking_projected_gravity(self,current_projected_gravity,desired_projected_gravity):
        error = torch.sum(torch.square(current_projected_gravity - desired_projected_gravity), dim=1)
        # print("_reward_tracking_rp_vel",error)
        return torch.exp(-error*0.1)     

    def _reward_tracking_rp(self,current_rp,desired_rp):
        error = torch.sum(torch.square(current_rp - desired_rp), dim=1)
        return torch.exp(-error*0.1)   

    def _reward_tracking_linear_vel(self,current_v,desired_v):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(current_v - desired_v), dim=1)
        return torch.exp(-lin_vel_error*2.)    

    def _reward_tracking_z_pos(self,current_z,desired_z):
        # Tracking of linear velocity commands (xy axes)
        z_error = torch.sum(torch.square(current_z - desired_z), dim=1)
        return torch.exp(-z_error*0.1)   

    def _reward_imitation(self):
        frames = self.amp_data.get_full_frame_batch(self.num_envs)


        r1 = self._reward_tracking_joint(self.dof_pos,
                                        self.amp_data.get_joint_pose_batch(frames))   
        r2 = self._reward_tracking_joint_vel(self.dof_vel,
                                        self.amp_data.get_joint_vel_batch(frames))   

          
        return 1*(r1+r2).mean()  

    
    def get_env_mass_central_fluncations(self):

        xyz = self.root_states[:, :3] - self.robot_pos_zero
        
        rpy = self.base_rpy
        foot_force_feedback = self.contact_force_value # dim:(env_num, 4)
        mass_central = torch.cat((xyz,rpy),dim=-1)
        return mass_central, foot_force_feedback

    def get_env_task_entity_identity(self):
        foot_force_feedback = self.contact_force_value # dim:(env_num, 4)

        mass_central = torch.cat((self.base_lin_vel,self.base_ang_vel),dim=-1)

        desired_foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        
        return mass_central, foot_force_feedback,desired_foot_pos

 
    def reset_command(self,commands):
        # print(commands)
        self.command_ranges["lin_vel_x"] = [commands[0],commands[0]+0.0001]
        self.command_ranges["lin_vel_y"] = [commands[1],commands[1]+0.0001]
        self.command_ranges["lin_vel_z"] = [commands[2],commands[2]+0.0001]
        self.command_ranges["ang_vel_roll"] = [commands[3],commands[3]+0.0001]
        self.command_ranges["ang_vel_pitch"] = [commands[4],commands[4]+0.0001]
        self.command_ranges["ang_vel_yaw"] = [commands[5],commands[5]+0.0001]
        self._resample_commands(env_ids=[0])

    def get_task_param(self,name=None,sequential_id=0,WalkingSlowCommands=None):
        # print(sequential_id)
        if name=="walking_slow":
            commands = torch.tensor(WalkingSlowCommands, device=self.device, dtype=torch.float)
            body_move = np.array([WalkingSlowCommands for _ in range(40)])

        else:
            if name==None:
                # name = skills_name_normal[np.random.randint(0,len(skills_name_normal))][:-3]
                # print("name",name)
                sequential_id = int(sequential_id)
                name = skills_name_normal[self.task_params[self.case_id][sequential_id%self.sequential_num]][:-3]


            if self.terrain_name in ['ComplexTerrain_NewEnv']:
                body_move,commands = self.body_move_dict[name+"_oe"]

            elif self.terrain_name in ['ComplexTerrain_Sequential']:
                body_move,commands = self.body_move_dict[name+"_oe"]
                commands = commands*1.2
                for com_id in range(len(commands)):
                    if torch.abs(commands[com_id])>0.:
                        body_move[:,com_id] = body_move[:,com_id]*1.2

            elif self.terrain_name in ['ComplexTerrain_Sequential_Case_1']:

                body_move,commands = self.body_move_dict[name+"_oe"]
                commands = commands*1.
                for com_id in range(len(commands)):
                    if torch.abs(commands[com_id])>0.:
                        body_move[:,com_id] = body_move[:,com_id]*1.
                            

        self.command_ranges["lin_vel_x"] = [commands[0],commands[0]+0.01]
        self.command_ranges["lin_vel_y"] = [commands[1],commands[1]+0.01]

        self.command_ranges["lin_vel_z"] = [commands[2],commands[2]+0.01]
        self.command_ranges["ang_vel_roll"] = [commands[3],commands[3]+0.01]
        self.command_ranges["ang_vel_pitch"] = [commands[4],commands[4]+0.01]
        self.command_ranges["ang_vel_yaw"] = [commands[5],commands[5]+0.01]
        self._resample_commands(env_ids=[0])

        # body_move = np.concatenate((body_move,body_move),axis=-1)

        return body_move,name
    

    
    def Command_change(self):
        current_state = self.root_states[:,:2]

        if self.cfg.terrain.case == 0:
            desired_point_1 = torch.tensor([self.cfg.terrain.terrain_length*(self.cfg.terrain.num_cols-1)-0.05,
                                        self.cfg.terrain.terrain_width*1*0.5],
                                        device=self.device, requires_grad=False, dtype=torch.float)
                    
        elif self.cfg.terrain.case == 1:
            key_point = torch.tensor(self.terrain.key_point,
                                        device=self.device, requires_grad=False, dtype=torch.float)

                    
            desired_point_1 = key_point
            
        elif self.cfg.terrain.case == 2:
            # todo: get key desired point
            # init point [3,9]

            # the first key point
            desired_point_1 = torch.tensor([9,9.5], 
                                        device=self.device, requires_grad=False, dtype=torch.float)
        


        max_vel = 0.5
        max_yaw_vel = 0.01

        distance = desired_point_1 - current_state
        x = distance.squeeze().cpu().numpy()[0]
        y = distance.squeeze().cpu().numpy()[1]
        x,y = float(x),float(y)
        distance_point_x = torch.linspace(0,x,100)
        distance_vel_x = (torch.diff(distance_point_x)/self.dt)[0]
        distance_vel_x = torch.sign(distance_vel_x)*torch.clip(torch.abs(distance_vel_x),max=max_vel).to(self.device)

        distance_point_y = torch.linspace(0,y,100)
        distance_vel_y = (torch.diff(distance_point_y)/self.dt)[0]
        distance_vel_y = torch.sign(distance_vel_y)*torch.clip(torch.abs(distance_vel_y),max=max_vel).to(self.device)

        x,y = distance_vel_x, distance_vel_y

        yaw = self.base_rpy[:,2].squeeze().cpu().numpy()-2*torch.pi
        # print("yaw",yaw)

        yaw = float(yaw)
        distance_point_yaw = torch.linspace(0,yaw,100)
        distance_vel_yaw = (torch.diff(distance_point_yaw)/self.dt)[0]
        distance_vel_yaw = torch.sign(distance_vel_yaw)*torch.clip(torch.abs(distance_vel_yaw),max=max_yaw_vel).to(self.device)
        yaw_vel = distance_vel_yaw


        self.reset_command([x,y,0,0,0,yaw_vel])

        return False    
    
    def ActionCorrection(self,action):
        dof_pos_np = self.dof_pos.cpu().numpy()

        # print("dof_pos_np",dof_pos_np)

        J_matrix = []
        J_matrix_inv = []
        # print("self._model",self._model)
        # print("self._data",self._data)
        for i in range(self.cfg.env.num_envs):
            matrix = pin.computeJointJacobians(self._model, self._data, dof_pos_np[i, :])
            # print("matrix",matrix)
            # matrix_inv = np.linalg.pinv(matrix)
            
            # matrix_inv = np.linalg.inv(matrix.T.dot(matrix)).dot(matrix.T)
            from scipy import linalg
            
            matrix_inv = linalg.pinv(matrix)

            J_matrix.append(matrix)
            J_matrix_inv.append(matrix_inv)

        J_matrix = torch.tensor(np.array(J_matrix), device=self.device, requires_grad=False, dtype=torch.float).transpose(1,2)
        J_matrix_inv = torch.tensor(np.array(J_matrix_inv), device=self.device, requires_grad=False, dtype=torch.float)
  
        desired_xyz = torch.cat((self.commands[:, :2],torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)),dim=-1)
        desired_rpy = torch.cat((self.commands[:, 4].unsqueeze(-1),self.commands[:, 5].unsqueeze(-1),self.commands[:, 2].unsqueeze(-1)),dim=-1)

        x_d = -torch.cat((desired_xyz,desired_rpy),dim=-1) + torch.cat((self.base_lin_vel,self.base_ang_vel),dim=-1)
        x = x_d * self.cfg.sim.dt
        # print("x",x)

        delta_param = 2.


        desired_dof_vel = torch.bmm(J_matrix_inv,torch.cat((desired_xyz,desired_rpy),dim=-1).unsqueeze(-1)).squeeze() - delta_param*torch.bmm(J_matrix_inv,x.unsqueeze(-1)).squeeze()
        

        self.current_desired_dof_pos = self.current_desired_dof_pos + desired_dof_vel*self.cfg.sim.dt

        s = self.dof_vel - desired_dof_vel
        # print("s",s)
        self.s = s

        Q = torch.diag_embed(action)
        
        lambda_param = 0.01


        alpha_rate = -lambda_param*self.alpha+0.02*torch.bmm(Q,s.unsqueeze(-1)).squeeze()

        self.alpha = self.alpha+alpha_rate * self.cfg.sim.dt
        
        self.alpha = (0.0+2.0*torch.clip(self.alpha,min=0.95,max=1.05))/2.
        
      
        return action*self.alpha
        # return action
    
   
    def GetEnergyConsumption(self):
        """Get the amount of energy used in last one time step.

        Returns:
        Energy Consumption based on motor velocities and torques (Nm^2/s).
        """

        return np.abs(
          self.torques.cpu().numpy()*self.dof_vel.cpu().numpy())
    
    def GetCOT(self):
        power = np.sum(self.GetEnergyConsumption(),axis=-1)
        # basemass = self.GetBaseMassesFromURDF()[0]
        # legmass = self.GetLegMassesFromURDF()
        mass = 12.
        g =9.8
        weight = mass * g
        Vel = np.linalg.norm(self.base_lin_vel.cpu().numpy(),axis=-1)
        return power / (weight * Vel)
    

