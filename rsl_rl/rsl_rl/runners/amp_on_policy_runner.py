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

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import pickle

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, LegActorCritic, JointActorCritic, VAE
from rsl_rl.modules.vqvae import VQVAE 
from rsl_rl.modules.transformervae import TransformerVAE 

from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
# from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.storage.replay_buffer import ReplayBuffer

# from WBC.WBC_Ctrl import WBC_Ctrl,Input
# from WBC.KinWBC import KinWBC

class AMPOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                  actor_critic_class="ActorCritic",
                 device='cpu',
                #  IsObservationEstimation=False
                #  amp_replay_buffer_size=100000,
                 ):
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs

        # actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        
        print("="*50)
        print(actor_critic_class)
        print("="*50)

        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs
        
        print("num_actor_obs",num_actor_obs) # 44*6=264

        self.IsObservationEstimation=self.env.cfg.env.IsObservationEstimation

        if actor_critic_class=="ActorCritic":
            # print("num_actor_obs_true",num_actor_obs_true)
            num_ObsEst_obs=None
            self.vel_dim=None
            self.ObsEstModel=None

            if self.IsObservationEstimation and self.env.cfg.env.state_est_dim>0:
                # self.cfg['vel_dim'] = 3
                self.cfg['vel_dim'] = self.env.cfg.env.state_est_dim
                
                if self.env.cfg.env.ObsEstModelName =="VAE" or self.env.cfg.env.ObsEstModelName == "TransformerVAE":
                    self.cfg['z_dim'] = 16
                elif self.env.cfg.env.ObsEstModelName =="VQVAE":
                    self.cfg['z_dim'] = 16

                self.vel_dim =self.cfg['vel_dim']
                z_dim= self.cfg['z_dim']
                num_actor_obs_true = self.env.num_obs + z_dim 

                if self.env.cfg.env.useTrackingError_two:
                    num_actor_obs_true+=12

                if self.env.include_history_steps is not None:
                    if self.env.cfg.env.useTrackingError:
                        include_history_steps = self.env.include_history_steps-2
                    else:
                        include_history_steps = self.env.include_history_steps-1
                    # command 3 last_action 12
                    num_ObsEst_obs = (self.env.num_obs - self.vel_dim -self.env.cfg.env.current_addition_dim ) * include_history_steps
                else:
                    num_ObsEst_obs = self.env.num_obs - self.vel_dim -self.env.cfg.env.current_addition_dim
                
                print("state_est_dim",self.vel_dim) 
                print("num_ObsEst_obs",num_ObsEst_obs)  # 195 
                print("num_actor_obs_true",num_actor_obs_true) # 54
                
                nn_dim = 128
                nn_dim2 = 64
                
                if self.env.cfg.env.ObsEstModelName =="VAE":
                    self.ObsEstModel = VAE(img_shape=num_ObsEst_obs,
                                    vel_dim=self.vel_dim,
                                    latent_dim=z_dim,
                                    obs_decode_num=num_ObsEst_obs// include_history_steps,
                                    
                                    nn_dim = nn_dim, 
                                    nn_dim2 = nn_dim2,
                                    # nn_dim = 512, 
                                    # nn_dim2 = 256,

                                    activation='elu',
                                    useRewardDone=self.env.cfg.env.useRewardDone,
                                    useTrackingError=self.env.cfg.env.useTrackingError,
                                    useTrackingError_two=self.env.cfg.env.useTrackingError_two
                                    ).to(device)
                
                elif self.env.cfg.env.ObsEstModelName =="VQVAE":
                    num_hiddens = 1024 # 512 # 128 # 512 # 128 
                    num_residual_hiddens = 32
                    num_residual_layers = 2

                    # This value is not that important, usually 64 works.
                    # This will not change the capacity in the information-bottleneck.
                    # embedding_dim = 64
                    embedding_dim = self.cfg['z_dim'] # 4 bad

                    # The higher this value, the higher the capacity in the information bottleneck.
                    # num_embeddings = 512
                    # num_embeddings = 4096 # 512 bad  # 256 bad
                    num_embeddings = 8192 #1024

                    # commitment_cost should be set appropriately. It's often useful to try a couple
                    # of values. It mostly depends on the scale of the reconstruction cost
                    # (log p(x|z)). So if the reconstruction cost is 100x higher, the
                    # commitment_cost should also be multiplied with the same amount.
                    
                    # commitment_cost = 0.25
                    commitment_cost = 0.05

                    decay = 0.99
                    self.ObsEstModel = VQVAE(
                                            num_ObsEst_obs,
                                            # include_history_steps, 
                                             embedding_dim, 
                                             num_embeddings, 
                                             num_hiddens,
                                                    num_residual_layers, 
                                                    num_residual_hiddens,
                                                      commitment_cost, 
                                                      decay,
                                                      vel_dim=self.vel_dim,
                                                      obs_decode_num=num_ObsEst_obs// include_history_steps,
                                                      ).to(device)

                elif self.env.cfg.env.ObsEstModelName =="TransformerVAE":
                    sequence_length = include_history_steps
                  
                    ntokens = num_ObsEst_obs // include_history_steps
                    
                    obs_decode_num=num_ObsEst_obs// include_history_steps
                    """
                    # Model parameters
                    ntokens = len_vocab
                    e_dim = 512
                    ff_dim = 4 * e_dim
                    nheads = 8
                    nTElayers = 4
                    nTDlayers = 4
                    z_dim = 32
                    """
                    nheads = 4
                    e_dim = nheads*64
                    ff_dim = 2 * e_dim
                    nTElayers = 2
                    nTDlayers = 2
                    z_dim = self.cfg['z_dim']
                    vel_dim = self.cfg['vel_dim']

                    self.ObsEstModel = TransformerVAE(
                    ntokens, e_dim, z_dim, nheads, 
                    ff_dim, nTElayers, nTDlayers,
                    sequence_length=sequence_length,
                    sequence_target_length=1,
                    obs_decode_num = obs_decode_num,
                    vel_dim = vel_dim
                    ).to(device)

                self.PrivilegeInfoEncoder=None
                if self.env.cfg.env.usePrivilegeLabel:
                    import torch.nn as nn
                    layers = []
                    mlp_input_dim_a = self.env.num_privileged_obs - self.env.num_obs
                    layers.append(nn.Linear(mlp_input_dim_a, nn_dim))
                    layers.append(nn.ELU())
                    layers.append(nn.Linear(nn_dim, nn_dim2))
                    layers.append(nn.ELU())
                    layers.append(nn.Linear(nn_dim2, z_dim))
                    self.PrivilegeInfoEncoder = nn.Sequential(*layers).to(device)

                    num_critic_obs = self.env.num_obs + z_dim

                actor_critic = ActorCritic(num_actor_obs=num_actor_obs_true,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        device=self.device,
                                                        **self.policy_cfg)
            
            else:
                actor_critic = ActorCritic(num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        device=self.device,
                                                        **self.policy_cfg)
                
            # train_cfg['runner']['amp_task_reward_lerp'] = 0.3
            train_cfg['algorithm']['entropy_coef'] = 0.005
            train_cfg['algorithm']['gamma'] = 0.99

        elif actor_critic_class=="LegActorCritic":
            actor_critic = LegActorCritic( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        device=self.device,
                                                        **self.policy_cfg) 
            # train_cfg['runner']['amp_task_reward_lerp'] = 0.3 
            train_cfg['algorithm']['entropy_coef'] = 0.015 #good 0.02
            train_cfg['algorithm']['gamma'] = 0.9

            # print(actor_critic)                                                                                     
        elif actor_critic_class=="JointActorCritic":
            actor_critic = JointActorCritic( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        device=self.device,
                                                        **self.policy_cfg)
            # train_cfg['runner']['amp_task_reward_lerp'] = 0.3

            train_cfg['algorithm']['entropy_coef'] = 0.025 # good 0.03
            train_cfg['algorithm']['gamma'] = 0.85
        
        elif actor_critic_class=="ActorCriticRecurrent":
            actor_critic = ActorCriticRecurrent(
                                       num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        device=self.device,
                                                        **self.policy_cfg)
            
            # print(actor_critic)
        # amp_data = AMPLoader(
        #     device, time_between_frames=self.env.dt, preload_transitions=True,
        #     num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
        #     motion_files=self.cfg["amp_motion_files"])
        # print(actor_critic_class)
        # print(train_cfg['runner']['amp_task_reward_lerp'])


        train_cfg['runner']['amp_reward_lerp'] = self.env.get_imitation_reward_weight()    

        self.H = 2
        self.observation_amp_dim = self.env.observation_amp_dim
        self.discriminator = AMPDiscriminator(
            self.env.observation_amp_dim * self.H,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            # train_cfg['runner']['amp_task_reward_lerp']
            train_cfg['runner']['amp_reward_lerp']
            ).to(self.device)
        
        # amp_data = ReplayBuffer(
        #     discriminator.input_dim // 2, amp_replay_buffer_size, device)


        # Num_skills = train_cfg['runner']['num_skills']
        # #print(train_cfg['runner']['amp_motion_files'])
        # f_read = open(train_cfg['runner']['amp_motion_files'][0], 'rb')
        # dict2 = pickle.load(f_read)
        # states_array = dict2["states"][:Num_skills]
        # next_states_array = dict2["next_states"][:Num_skills]
        # f_read.close()

        # for i in range(Num_skills):
        #     states = torch.from_numpy(states_array[i]).to(device)
        #     next_states = torch.from_numpy(next_states_array[i]).to(device)
        #     amp_data.insert(states, next_states)

        # print('='*50)
        # print(train_cfg['runner']['amp_motion_files'][0])
        # print("expert num_samples:",amp_data.num_samples)
        # print('='*50)
    
        amp_normalizer = Normalizer(self.env.observation_amp_dim)   


        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0])))

        self.alg: PPO = alg_class(env,
                                  actor_critic, 
                                    self.discriminator, 
                                    self.env.amp_data, 
                                    amp_normalizer, 
                                    device=self.device, 
                                    min_std=min_std, 

                                    IsObservationEstimation=self.IsObservationEstimation,
                                    num_ObsEst_obs=num_ObsEst_obs,
                                    vel_dim=self.vel_dim,
                                    ObsEstModel=self.ObsEstModel,
                                    PrivilegeInfoEncoder=self.PrivilegeInfoEncoder,
                                    # isMSELoss = self.cfg["isMSELoss"],
                                    **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # if self.env.cfg.env.useWBC:
        #     self.input = Input()
        #     _ = KinWBC()
        #     self.wbc_ctrl = WBC_Ctrl(self.input)

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)

        if self.env.cfg.env.useWBC:
            self.last_pBody_RPY_des = self.env.base_rpy.cpu().numpy()
            self.last_pBody_des = np.repeat(np.array([[0.,0.,0.25]]),self.env.num_envs, axis=0)
            foot_pos = self.env.foot_positions_in_base_frame(self.env.dof_pos)
            
            last_pFoot_des = foot_pos.view(self.env.num_envs,4,3).cpu().numpy()
            self.last_pFoot_des = np.stack((last_pFoot_des[:,1],last_pFoot_des[:,0],last_pFoot_des[:,3],last_pFoot_des[:,2]),axis=1)

            self.last_bodyPosition = np.repeat(np.array([[0.,0.,0.25]]),self.env.num_envs, axis=0)


        # amp_obs_H = torch.zeros((self.env.num_envs,self.observation_amp_dim, self.H), dtype=torch.float, device=self.device)
        # amp_obs_H[:,:,-1] = amp_obs

        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        it_old = None
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # print("obs",obs.size())

                    current_joint_pos = obs[:,6:18]
                    # current_joint_pos_vel = obs[:,18:30]

                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    desired_jpos_and_vel_list = []
                    if (self.env.cfg.env.useWBC and (it>=self.env.curriculum_start_num)):
                    # if (self.env.cfg.env.useWBC and (it>=0)):

                        vel_pred = self.alg.vel_ObservationEstimate.cpu().numpy()
                        foot_pos_pred = np.reshape(vel_pred[:,8:20],(self.env.num_envs,4,3))
                        foot_pos_pred = np.stack((foot_pos_pred[:,1],foot_pos_pred[:,0],foot_pos_pred[:,3],foot_pos_pred[:,2]),axis=1)

                        desired_jpos_and_vel_list = []
                        for num in range(self.env.num_envs):
                            if dones[num]:
                                self.last_pBody_RPY_des[num] = self.env.base_rpy[num].cpu().numpy()
                                self.last_pBody_des[num] = np.array([0.,0.,0.25])
                                foot_pos = self.env.foot_positions_in_base_frame(self.env.dof_pos)
                                
                                last_pFoot_des = foot_pos.view(self.env.num_envs,4,3).cpu().numpy()
                                self.last_pFoot_des[num] = np.stack((last_pFoot_des[num,1],last_pFoot_des[num,0],last_pFoot_des[num,3],last_pFoot_des[num,2]),axis=0)

                                self.last_bodyPosition[num] = np.array([0.,0.,0.25])

                            desired_jpos, desired_jpos_vel = self.get_desired_jpos_from_WBC(self.input, self.wbc_ctrl, self.env, foot_pos_pred[num],num)
                            desired_jpos_and_vel_list.append(torch.concat((desired_jpos,desired_jpos_vel)))
                        
                        desired_jpos_and_vel_list = torch.stack(desired_jpos_and_vel_list,dim=0).to(self.device)
                        # print(desired_jpos_and_vel_list)
                        # print(desired_jpos_and_vel_list.size())
                        
                        q_current = (current_joint_pos/self.env.obs_scales.dof_pos+self.env.default_dof_pos).detach()
                        q_label = desired_jpos_and_vel_list[:,:12].detach()

                        # q_vel_current = current_joint_pos_vel/self.env.obs_scales.dof_vel
                        # q_vel_label = desired_jpos_and_vel_list[:,12:].detach()

                        # loss_WBC = -1.* torch.mean(torch.square(q_current - q_label))
                        loss_WBC = -0.5* torch.mean(torch.square(q_current - q_label))
                        
                        # loss_WBC_vel = -0.001*torch.mean(torch.square(q_vel_current - q_vel_label))
                        # rewards += (loss_WBC+loss_WBC_vel)
                        rewards += loss_WBC
                        rewards = torch.clip(rewards[:], min=0.)

                        if it!=it_old:
                            print("Reward loss_WBC: ",loss_WBC.cpu().numpy())
                            # print("Reward loss_WBC_vel: ",loss_WBC_vel.cpu().numpy())
                            it_old=it

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards = self.alg.discriminator.predict_amp_reward(
                        # torch.reshape(amp_obs_H,(self.env.num_envs,-1)),
                        amp_obs, 
                        next_amp_obs_with_term, 
                        rewards, 
                        normalizer=self.alg.amp_normalizer)[0]

                    amp_obs = torch.clone(next_amp_obs)
                    # amp_obs_H = torch.cat((amp_obs_H,amp_obs.unsqueeze(2)),-1)[:,:,1:]
                    


                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term,
                                              self.env.reward_list,desired_jpos_and_vel_list
                                              )
                    
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
            
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

            self.alg.set_step_counter(it)
            self.env.set_step_counter(it)
            self.alg.discriminator.set_step_counter(it)
        
        self.current_learning_iteration += num_learning_iterations

        # print(self.current_learning_iteration)

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
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
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
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
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
        if self.IsObservationEstimation and self.env.cfg.env.state_est_dim>0:
            if self.PrivilegeInfoEncoder is not None:
                torch.save({
                    'model_state_dict': self.alg.actor_critic.state_dict(),
                     'PrivilegeInfoEncoder_state_dict': self.alg.PrivilegeInfoEncoder.state_dict(),
                    'observation_estimation_model_state_dict': self.alg.ObsEstModel.state_dict(),
                    'optimizer_state_dict': self.alg.optimizer.state_dict(),
                    'discriminator_state_dict': self.alg.discriminator.state_dict(),
                    'amp_normalizer': self.alg.amp_normalizer,
                    'iter': self.current_learning_iteration,
                    'infos': infos,
                    }, path)
            else:    
                torch.save({
                    'model_state_dict': self.alg.actor_critic.state_dict(),
                    'observation_estimation_model_state_dict': self.alg.ObsEstModel.state_dict(),
                    'optimizer_state_dict': self.alg.optimizer.state_dict(),
                    'discriminator_state_dict': self.alg.discriminator.state_dict(),
                    'amp_normalizer': self.alg.amp_normalizer,
                    'iter': self.current_learning_iteration,
                    'infos': infos,
                    }, path)
        else:    
            torch.save({
                'model_state_dict': self.alg.actor_critic.state_dict(),
                'optimizer_state_dict': self.alg.optimizer.state_dict(),
                'discriminator_state_dict': self.alg.discriminator.state_dict(),
                'amp_normalizer': self.alg.amp_normalizer,
                'iter': self.current_learning_iteration,
                'infos': infos,
                }, path)

    def load(self, path, load_optimizer=True):
        # print(path)
        loaded_dict = torch.load(path)
        # print(loaded_dict['model_state_dict'])

        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])

        if self.IsObservationEstimation and self.env.cfg.env.state_est_dim>0:
            self.alg.ObsEstModel.load_state_dict(loaded_dict['observation_estimation_model_state_dict'])            
            if self.PrivilegeInfoEncoder is not None:
                self.alg.PrivilegeInfoEncoder.load_state_dict(loaded_dict['PrivilegeInfoEncoder_state_dict']) 

        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_desired_jpos_from_WBC(self,input,wbc_ctrl,env, foot_pos_pred,num):
        input.bodyOrientation = env.base_quat[num].cpu().numpy()
        
        # pos_xy = (env.root_states[0, :2]).cpu().numpy()
        # pos_z = torch.mean((env.root_states[:, 2:3] - env.measured_heights),dim=-1)[0].cpu().numpy()
        # input.bodyPosition = np.concatenate((pos_xy, [pos_z]))

        # print(input.bodyPosition)

        input.bodyVelocity = np.concatenate((env.base_lin_vel[num].cpu().numpy(),env.base_ang_vel[num].cpu().numpy()))
        
        input.bodyPosition = self.last_bodyPosition[num] + input.bodyVelocity[:3]*env.dt
        self.last_bodyPosition[num] = input.bodyPosition 

        dof_pos = -1. * env.dof_pos[num].cpu().numpy()
    
        input.joint_q = np.concatenate((dof_pos[3:6],dof_pos[0:3],dof_pos[9:12],dof_pos[6:9]))
        # print(" input.joint_q", input.joint_q)
        
        dof_vel = -1. * env.dof_vel[num].cpu().numpy()
        input.joint_qd = np.concatenate((dof_vel[3:6],dof_vel[0:3],dof_vel[9:12],dof_vel[6:9]))


        input.leg_data_q = np.reshape(input.joint_q,(4,3))
        input.leg_data_qd = np.reshape(input.joint_qd,(4,3))


        input.contact_state = (env.contact_force_value>1.)[num].cpu().numpy()
        
        # print("input.contact_state",input.contact_state)

        input.vBody_Ori_des = np.array([0,0,env.commands[num, 2].cpu().numpy()])
        input.pBody_RPY_des = self.last_pBody_RPY_des[num] + input.vBody_Ori_des*env.dt
        self.last_pBody_RPY_des[num] =  input.pBody_RPY_des

        # print(" input.pBody_RPY_des", input.pBody_RPY_des)

        input.vBody_des = np.array([env.commands[num, 0].cpu().numpy(),
                                    env.commands[num, 1].cpu().numpy(),
                                    0.])
        
        input.pBody_des = self.last_pBody_des[num] + input.vBody_des*env.dt
        self.last_pBody_des[num] = input.pBody_des

        # print("input.pBody_des",input.pBody_des)
        # print("foot_pos_pred",foot_pos_pred)
        input.pFoot_des = foot_pos_pred 
        input.vFoot_des = (input.pFoot_des - self.last_pFoot_des[num])/env.dt
        self.last_pFoot_des[num] = input.pFoot_des

        # print("foot_pos_pred....",input.pFoot_des)
        # print("input.vFoot_des",input.vFoot_des)

        wbc_ctrl.run(input)

        desired_jpos = -1.*wbc_ctrl.jpos_cmd
        desired_jpos_vel = -1.*wbc_ctrl.jvel_cmd
        desired_jpos = torch.from_numpy(np.concatenate((desired_jpos[3:6],
                                                    desired_jpos[0:3],
                                                    desired_jpos[9:12],
                                                    desired_jpos[6:9]))).to(env.device)
        desired_jpos_vel = torch.from_numpy(np.concatenate((desired_jpos_vel[3:6],
                                                        desired_jpos_vel[0:3],
                                                        desired_jpos_vel[9:12],
                                                        desired_jpos_vel[6:9]))).to(env.device)

        return desired_jpos, desired_jpos_vel

