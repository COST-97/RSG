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
from typing import Dict, List, Tuple, Optional

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import CompositePPO, CompositeBO, CompositeES
# TODO:
from rsl_rl.modules import Actor, Critic, CompositeActor, NewCompositeActor
from rsl_rl.env import VecEnv

import numpy as np

torch.pi = torch.acos(torch.zeros(1)).item() * 2

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
}

task_command = {
"Task1":[0.3,0.,0.,0.,0.,0.],
"Task2":[0.4,0.,0.,0.,0.,-0.4], #"Task2":"forward_right",
"Task3":[0.4,0.,0.,0.,0.,0.4],

"Task4":[-0.2,0.,0.,0.,0.,0.], #"Task4":"backward_walking",
"Task5":[-0.5,0.,0.,0.,0.,0.4],

"Task6":[-0.4,0.,0.,0.,0.,-0.4], #"Task6":"backward_left_walking",

"Task7":[0.,-0.25,0.,0.,0.,0.],
"Task8":[0.,0.25,0.,0.,0.,0.], #"Task8":"sidestep_left",

"Task9":[0.,0.,0.,0.,0.,-0.5], #"Task9":"spin_clockwise",
"Task10":[0.,0.,0.,0.,0.,0.5], #"Task10":"spin_counterclockwise",
"Task11":[0.6,0.,0.,0.,0.,0.], # "Task11":"gallop",

"Task12":[0.6,0.,0.,0.,0.,0.], # "Task12":"forward_walking_fast",
"Task13":[0.2,0.,0.,0.,0.,0.],
"Task14":[0.2,0.,0.,0.,0.,0.],

"Task15":[0.,0.,2.,0.,0.,0.],
"Task16":[0.,0.,2.,0.,0.,0.],

"Task17":[-1.,0.,2.,0.,0.,0.],
"Task18":[1.,0.,2.,0.,0.,0.],
"Task19":[0.,0.75,2.,0.,0.,0.],
"Task20":[0.,-0.75,2.,0.,0.,0.],
}


    
def dist_fn(
    env: Optional[
        Tuple[
            # Env 1: mass_central, foot_force
            Tuple[torch.Tensor, torch.Tensor],
            # Env 2: mass_central, foot_force
            Tuple[torch.Tensor, torch.Tensor],
        ]
    ],
    task: Optional[
        Tuple[
            # Task 1: body_move, legs_move
            Tuple[torch.Tensor, torch.Tensor],
            # Task 2: body_move, legs_move
            Tuple[torch.Tensor, torch.Tensor],
        ]
    ],
) -> float:
    assert env is not None or task is not None

    env_dist = 0.0
    if env is not None:
        (mc1, ff1), (mc2, ff2) = env[0], env[1]
        env_dist += (mc1 - mc2).abs().mean() + (ff1 - ff2).abs().mean()

    task_dist = 0.0
    if task is not None:
        (bm1, lm1), (bm2, lm2) = task[0], task[1]
        task_dist += (bm1 - bm2).abs().mean() + (lm1 - lm2).abs().mean()

    return (env_dist + task_dist).item()


class SkillGraphOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.actor_cfg = train_cfg["actor"]
        self.critic_cfg = train_cfg["critic"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_class = eval(self.cfg["actor_name"]) # Actor
        critic_class = eval(self.cfg["critic_name"]) # Critic
        actor: actor = actor_class(self.env.num_obs, self.env.num_actions, self.device, **self.actor_cfg)
        critic: critic = critic_class(num_critic_obs, self.device, **self.critic_cfg)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        # print(self.alg_cfg['num_learning_epochs'])
        if alg_class.is_ES:
            self.alg = alg_class(actor, self.env, device=self.device, **self.alg_cfg)
        else:
            self.alg = alg_class(actor, critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

        # initialize skill graph
        self.sg = SkillGraph()
        self.sg.add_dist_fn(dist_fn)
        self.sg.load(path="logs/skill_graph_model")

        # set query env and query task
        self.N = 8
        
        self.mass_central_array = np.zeros((self.N,6))
        self.foot_force_feedback_array = np.zeros((self.N,4))
        
        self.get_env_info()

        self.body_move, self.desired_foot_pos = self.env.get_task_desired_mass_central_move(self.N,steps_num=50)
        self.query_task = {
                # "body_move": np.random.uniform(-50, 50, size=(N, 6)),
                # "legs_move": np.random.uniform(-50, 50, size=(N, 4, 3)),
                "body_move": self.body_move.cpu().numpy(), # desired xyz rpy (N,6)
                "legs_move": self.desired_foot_pos.cpu().numpy(), # desired foot pos
            } 

        print("self.query_env:\n",self.query_env)
        print("self.query_task:\n",self.query_task)

        # self.skills_list = []

        skills, (adjacency_matrix, adjacency_matrix_raw) = self.sg.knn(
            # k=train_cfg["actor"]["num_base_actor"],
            k=10,
            query_env=self.query_env,
            query_task=self.query_task,
            )
        
        # self.skills_list.append(skills)

        print("="*50)
        print("adjacency_matrix:\n",adjacency_matrix)
        print("base skills:")
        for skill in skills:
            print(skill[2][0].desc)
            # print(skill[2][0].model_weight)
        print("="*50)

        self.alg.actor.load_base_actors(skills, adjacency_matrix, adjacency_matrix_raw)



    def get_env_info(self):
        for e in self.sg.envs.values():
            if e.desc=="Indoor Flat Floor":
                self.mass_central_array = np.array(e.mass_central)
                self.foot_force_feedback_array = np.array(e.foot_force)

        self.query_env = {
                "foot_force": self.foot_force_feedback_array, # foot force (N,4)
                "mass_central": self.mass_central_array, # xyz rpy
            }

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        # switch to train mode (for dropout for example)
        self.alg.actor.train()
        self.alg.critic.train() 

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            # print("it",it)
            start = time.time()
            
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # print("i",i)
                    # zhy todo:

                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos,_,_ = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    # print("reward",rewards)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    self.get_env_info()

                    if dones:
                        self.mass_central_array = np.zeros((self.N,6))
                        self.foot_force_feedback_array = np.zeros((self.N,4))

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                      
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0


                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            ep_infos.clear()

            self.env.set_step_counter(it)
            self.alg.set_step_counter(it)
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if self.alg.actor.is_composite:
            mean_std = self.alg.actor.distribution.stddev.mean()
        else:
            mean_std = self.alg.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        # TODO: save base policy!
        torch.save({
            'actor_state_dict': self.alg.actor.state_dict(),
            'critic_state_dict': self.alg.critic.state_dict(),
            'actor_optimizer_state_dict': self.alg.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.alg.critic_optimizer.state_dict(),
            # 'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict['actor_state_dict'])
        self.alg.critic.load_state_dict(loaded_dict['critic_state_dict'])
        if load_optimizer:
            self.alg.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer_state_dict'])
            self.alg.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])
            #  self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        # switch to evaluation mode (dropout for example)
        self.alg.actor.eval() 
        if device is not None:
            self.alg.actor.to(device)
        return self.alg.actor.act_inference


MODEL_WEIGHT_FOLDER= "./new_data/skills"

class BOOnPolicyRunnerSequentialCase1:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.actor_cfg = train_cfg["actor"]
        # self.critic_cfg = train_cfg["critic"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs

        actor_class = eval(self.cfg["actor_name"]) # Actor
        # critic_class = eval(self.cfg["critic_name"]) # Critic
        actor: actor = actor_class(self.env,
                                   self.env.num_obs, 
                                   self.env.num_actions, 
                                   self.device, 
                                   **self.actor_cfg)
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        # print(self.alg_cfg['num_learning_epochs'])
        
        self.alg = alg_class(actor, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()


    def get_env_info(self):
        for e in self.sg.envs.values():
            if e.desc=="Indoor Flat Floor":
                self.mass_central_array = np.array(e.mass_central)
                self.foot_force_feedback_array = np.array(e.foot_force)

        self.query_env = {
                "foot_force": self.foot_force_feedback_array, # foot force (N,4)
                "mass_central": self.mass_central_array, # xyz rpy
            }

    def skill_extraction(self,env_param,task_param):
        self.Similar_Value_high = 0.9
        self.Similar_Value_low = 0.7

        skills, scores, (e_scores, t_scores) = self.env.sg.kgc(env_property=env_param,task_property=task_param.flatten())
        skill_weight = []
        scores_weight = []

        skill_desc_list = []
        Scores_list = []
        Env_scores_list = []
        Task_scores_list = []
        for i in range(len(skills)):
            if Scores<self.Similar_Value_low:
                break
    
            weight = skills[i].get_model_weight(MODEL_WEIGHT_FOLDER)
            skill_weight.append(weight)
            scores_weight.append(scores[i])

            Scores = round(float(scores[i]),3)
            Env_scores = round(float(e_scores[i]),3)
            Task_scores = round(float(t_scores[i]),3)

            skill_desc_list.append(skills[i].desc)
            Scores_list.append(Scores)
            Env_scores_list.append(Env_scores)
            Task_scores_list.append(Task_scores)


            if Scores>self.Similar_Value_high:
                break
        
        print("env_param:",env_param,skill_desc_list,"Scores:",Scores)

        self.env._resample_commands(env_ids=[0])
        self.alg.actor.load_base_actors(skill_weight,scores_weight)
        self.alg.reset=True    
        
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):        
        
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.skill_list = []
        skill_id = 1

        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        episode_num = 0
        episode_return = 0.
        episode_return_list = []


        env_param = self.env.get_env_param()
        task_param = self.env.get_task_param()
        
        task_param_next = None

        self.skill_extraction(env_param,task_param)

        for it in range(self.current_learning_iteration, tot_iter):

            # print("it",it)
            start = time.time()
            returns_list = []

            # Rollout
            for i in range(self.num_steps_per_env):
                env_param_next = self.env.get_env_param()
        
                if it%10==0:
                    task_param_next = self.env.get_task_param()

      
                if (env_param_next!=env_param).any() or (task_param_next is not None and task_param!= task_param_next).any():
                 
                    self.skill_extraction(env_param_next,task_param_next)
                    env_param = env_param_next
                    task_param = task_param_next

                actions = self.alg.act(obs, critic_obs)

                obs, privileged_obs, rewards, dones, infos,_,_ = self.env.step(actions)

                critic_obs = privileged_obs if privileged_obs is not None else obs
                obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                # print("reward",rewards)
                self.alg.process_env_step(rewards, dones, infos)
                
                episode_return+=rewards

                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])

                    # print("rewards",rewards.size())   

                    cur_reward_sum += rewards

                    returns_list.append(rewards.cpu().numpy())

                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    
                    if dones>0:
                        episode_num+=1
                        episode_return_list.append(float(episode_return.squeeze().cpu().numpy()))
                        # print("episode_num",episode_num)
                        episode_return = 0.


                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            self.alg.step=0

            stop = time.time()
            learn_time = stop - start
            

            ep_infos.clear()
            
            self.env.set_step_counter(it)
            self.alg.set_step_counter(it)
            
        self.current_learning_iteration += num_learning_iterations
        # self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

        if len(episode_return_list)==0:
            performance=0
        else:    
            performance = episode_return_list
            print("Performance:",np.round(performance))
                    

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if self.alg.actor.is_composite:
            mean_std = self.alg.actor.distribution.stddev.mean()
        else:
            mean_std = self.alg.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        # TODO: save base policy!
        torch.save({
            'actor_state_dict': self.alg.actor.state_dict(),
            # 'critic_state_dict': self.alg.critic.state_dict(),
            # 'actor_optimizer_state_dict': self.alg.actor_optimizer.state_dict(),
            # 'critic_optimizer_state_dict': self.alg.critic_optimizer.state_dict(),
            # 'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict['actor_state_dict'])
        # self.alg.critic.load_state_dict(loaded_dict['critic_state_dict'])
        # if load_optimizer:
        #     self.alg.actor_optimizer.load_state_dict(loaded_dict['actor_optimizer_state_dict'])
        #     self.alg.critic_optimizer.load_state_dict(loaded_dict['critic_optimizer_state_dict'])
            #  self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        # switch to evaluation mode (dropout for example)
        self.alg.actor.eval() 
        if device is not None:
            self.alg.actor.to(device)
        return self.alg.actor.act_inference
