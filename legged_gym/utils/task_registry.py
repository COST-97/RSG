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

import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, AMPOnPolicyRunner, SkillGraphOnPolicyRunner, BOOnPolicyRunner,BOOnPolicyRunnerSequential

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None, curr_start_num=None,epi_length_s=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)

        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        
        # train
        # env_cfg.env.num_envs = 15000
        
        curriculum_start_num = curr_start_num
        env_cfg.env.num_envs = args.num_envs
        if curriculum_start_num==None:
            curriculum_start_num= args.max_iterations//2
            # curriculum_start_num= args.max_iterations//2

        if epi_length_s is not None:
            env_cfg.env.episode_length_s  = epi_length_s

        # test
        # env_cfg.env.num_envs = 100

        # set_seed(env_cfg.seed)
        # zhy
        set_seed(args.seed)

        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless,
                            skills_descriptor_id=args.skills_descriptor_id,
                            terrain_id=args.terrain_id,
                            leg_lift_id=args.leg_lift_id,
                            isActionCorrection=args.isActionCorrection,
                            case_id=args.case_id,
                            curriculum_start_num=curriculum_start_num,
                            isObservationEstimation=args.isObservationEstimation,
                            isEnvBaseline=args.isEnvBaseline,
                            isBOEnvBaseline=args.isBOEnvBaseline,
                            # seed=args.seed
                            )
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")

        # train_cfg.runner.max_iterations = 500
        # train_cfg.runner.max_iterations = 350

        train_cfg.runner.save_interval = 500
        # train_cfg.runner.save_interval = 200

        if args.isObservationEstimation:
            train_cfg.runner.experiment_name = train_cfg.runner.experiment_name + "_oe"
              
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)



        if log_root=="default":
            if args.actor_critic_class is not None:
                if (args.skills_descriptor_id is not None) and (args.terrain_id is not None):
                    if args.leg_lift_id is not None:
                        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                                                train_cfg.runner.experiment_name, 
                                                args.actor_critic_class+"_EnvID_"+str(args.terrain_id)+"_SkillID_"+str(args.skills_descriptor_id)
                                                +"_LegLiftID_"+str(args.leg_lift_id)                                               
                                                )                    
                    else:
                        # if args.isActionCorrection:    
                        #     log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                        #                         train_cfg.runner.experiment_name, 
                        #                         args.actor_critic_class+"_EnvID_"+str(args.terrain_id)+"_SkillID_"+str(args.skills_descriptor_id)+"_ActionCorrection")
                        # else:  
                        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 
                                                train_cfg.runner.experiment_name, 
                                                args.actor_critic_class+"_EnvID_"+str(args.terrain_id)+"_SkillID_"+str(args.skills_descriptor_id))
                                                    
                else:    
                    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.actor_critic_class)
            else:
                log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        # print(train_cfg.runner_class_name)
        runner_class = eval(train_cfg.runner_class_name)
        train_cfg_dict = class_to_dict(train_cfg)

        if args.actor_critic_class is not None:
            runner = runner_class(env, train_cfg_dict, log_dir, actor_critic_class=args.actor_critic_class, device=args.rl_device)
        else:
            runner = runner_class(env, train_cfg_dict, log_dir, device=args.rl_device)

        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()