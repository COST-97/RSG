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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import rsl_rl.algorithms.pytorch_stats_loss as stats_loss

from rsl_rl.modules import ActorCritic, VAE
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer

class AMPPPO:
    actor_critic: ActorCritic
    def __init__(self,
                 env,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 isMSELoss=True,
                 min_std=None,
                 IsObservationEstimation=False,
                 num_ObsEst_obs=None,
                 vel_dim=None,
                 ObsEstModel=None,
                 PrivilegeInfoEncoder=None
                 ):

        self.env =env
        self.device = device

        self.isMSELoss = isMSELoss

        self.step_counter_last = -1
        self.step_counter = 0

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()

        self.H = 2
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // self.H, amp_replay_buffer_size, device)

        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        # Optimizer for policy and discriminator.
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {'params': self.discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 'name': 'amp_head'},
            # {'params': self.ObsEstModel.parameters(),
            #  'weight_decay': 10e-4, 'name': 'ObsEstModel'},
             ]

        self.IsObservationEstimation = IsObservationEstimation
        self.ObsEstModel=ObsEstModel
        self.vel_dim=vel_dim

        self.PrivilegeInfoEncoder = PrivilegeInfoEncoder

        if IsObservationEstimation:
            self.reward_dim = self.env.cfg.env.reward_dim
            if self.env.cfg.env.state_est_dim>0:
                if self.env.cfg.env.useRealData:
                    import h5py
                    f = h5py.File('data_state_estimation.h5', 'r')
                    self.input_data_real = f["input_data"]
                    self.output_data_real = f["output_data"]
                    # print(self.input_data_real)
                    # print(self.output_data_real)
                    self.input_data_real = torch.tensor(np.array(self.input_data_real),dtype=torch.float32).to(self.device)
                    self.output_data_real = torch.tensor(np.array(self.output_data_real),dtype=torch.float32).to(self.device)
                    # self.input_data_real = torch.from_numpy(self.input_data_real,dtype=torch.float32).to(self.device)
                    # self.output_data_real = torch.from_numpy(self.output_data_real,dtype=torch.float32).to(self.device)

                    # print("input_data_size:",self.input_data_real.size())
                    # print("output_data_size:",self.output_data_real.size())

                    if self.env.cfg.env.state_est_dim == 3:
                        self.output_data_real = self.output_data_real[:,:3]
                    elif self.env.cfg.env.state_est_dim == 12:
                        self.output_data_real = self.output_data_real[:,:-12]                
                    elif self.env.cfg.env.state_est_dim == 20:
                        self.output_data_real = torch.cat((self.output_data_real[:,:-12-4],self.output_data_real[:,-12:]),dim=-1)

                    data_real = []
                    recon_label_batch_real = []
                    vel_label_real = []

                    include_history_steps = self.env.include_history_steps
                    for i_real in range(self.input_data_real.size()[0]//include_history_steps-1):
                        tmp = self.input_data_real[i_real*include_history_steps:(i_real+1)*include_history_steps]
                        tmp_output = self.output_data_real[i_real*include_history_steps:(i_real+1)*include_history_steps]
                        # data_real.append(tmp[-2::-1].view(1,-1))
                        obs = []
                        # for obs_id in reversed(sorted(obs_ids)):
                        for obs_id in range(include_history_steps-1):
                            obs.append(tmp[include_history_steps-2 - obs_id])
                        data_real.append(torch.cat(obs, dim=-1))
            
                        recon_label_batch_real.append(tmp[-1])
                        vel_label_real.append(tmp_output[-2])

                    self.data_real = torch.stack(data_real,dim=0)
                    self.recon_label_batch_real = torch.stack(recon_label_batch_real,dim=0)
                    self.vel_label_real = torch.stack(vel_label_real,dim=0)
                    # print("data_real:",self.data_real.size())
                    # print("recon_label_batch_real:",self.recon_label_batch_real.size())
                    # print("vel_label_real:",self.vel_label_real.size())

                self.num_observations=self.env.cfg.env.num_observations
                # self.vel_dim = self.env.cfg.env.vel_dim
                # self.z_dim = self.env.cfg.env.z_dim
                self.num_ObsEst_obs=num_ObsEst_obs

                # self.ObsEstModel = VAE(img_shape=num_ObsEst_obs,
                #                        vel_dim=self.vel_dim,
                #                        latent_dim=self.z_dim,
                #                        obs_decode_num=num_ObsEst_obs// (self.env.include_history_steps-1)
                #                        ).to(device)

                params.append({'params': self.ObsEstModel.parameters(),
                            'weight_decay': 10e-4, 'name': 'ObsEstModel'})
                
                if self.PrivilegeInfoEncoder is not None:
                    params.append({'params': self.PrivilegeInfoEncoder.parameters(),
                           'name': 'PrivilegeInfoEncoder'})

                # params_oe = [{'params': self.ObsEstModel.parameters(),
                #             'weight_decay': 10e-4, 'name': 'ObsEstModel'}]
                # self.optimizer_oe = optim.Adam(params_oe, lr=1e-3)
            
        # if self.isMSELoss:
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # else:
            # self.optimizer = optim.RMSprop(params, lr=learning_rate)

        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device,self.IsObservationEstimation,self.reward_dim)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()
    
    def set_step_counter(self,step_counter):
        self.step_counter = step_counter

    def ObservationEstimate(self,aug_obs,aug_critic_obs=None):
        # print("aug_obs",aug_obs.size()) # (-1,195)
        obs_list = []
        current_obs = []
        
        if self.env.cfg.env.useTrackingError:
            include_history_steps = self.env.include_history_steps-2
        else:
            include_history_steps = self.env.include_history_steps-1

        for obs_id in range(include_history_steps):
            obs_one = aug_obs[:, obs_id * (self.num_observations) : (obs_id + 1) * (self.num_observations)]
            # command 3 last_action 12
            obs_list.append(obs_one[:,:-self.vel_dim-self.env.cfg.env.current_addition_dim])
            if obs_id==0:
                current_obs = obs_one

        # print(1111111111111111)
        if self.env.cfg.env.ObsEstModelName =="VAE":         
            aug_obs = torch.cat(obs_list, dim=-1)
            z_decode, z_mu, z_logvar, vel = self.ObsEstModel.forward(aug_obs)
            
            # z_decode, z_mu_old, z_logvar, vel, z = self.ObsEstModel.forward(aug_obs)
            # z_mu = z
            # print("z_mu_old",z_mu_old[0])
            # print("z_logvar",z_logvar[0])
            
            # print("z_mu",z_mu[0])    
            
        elif self.env.cfg.env.ObsEstModelName =="VQVAE":
            aug_obs = torch.cat(obs_list, dim=-1)
            # aug_obs = torch.stack(obs_list, dim=1)
            self.ObsEstModel.eval()
            # print(2222222222)
            z_decode, z_mu, vel = self.ObsEstModel.forward(aug_obs)
            # z_mu = z_mu.view(z_mu.size()[0],-1)
        elif self.env.cfg.env.ObsEstModelName =="TransformerVAE":         
            aug_obs = torch.stack(obs_list, dim=1)
            z_decode, z_mu, z_logvar, vel = self.ObsEstModel.forward(aug_obs)

        # if self.step_counter>1.5*self.env.curriculum_start_num: 
        #     current_obs = torch.cat((current_obs[:,:-self.vel_dim],vel), dim=-1)
        # current_obs = torch.cat((current_obs[:,:-self.vel_dim],vel), dim=-1)

        if self.env.cfg.env.useTrackingError_two:
            aug_obs = torch.cat((current_obs,z_mu,self.ObsEstModel.getTrackingError_pred()), dim=-1).detach()
        else:
            # if (self.step_counter<self.env.curriculum_start_num) and (aug_critic_obs is not None): 
            #     privilege_embedding = self.PrivilegeInfoEncoder(aug_critic_obs[:,self.env.num_obs:]) 
            #     aug_obs = torch.cat((current_obs,privilege_embedding), dim=-1).detach()
            # else:
            #     aug_obs = torch.cat((current_obs,z_mu), dim=-1).detach()
            aug_obs = torch.cat((current_obs,z_mu), dim=-1).detach()

        # privilege_embedding = self.PrivilegeInfoEncoder(aug_critic_obs[:,self.env.num_obs:]) 
        # print("privilege_embedding",privilege_embedding) 
        # aug_obs = torch.cat((current_obs,privilege_embedding), dim=-1).detach()

        return aug_obs,vel,z_decode
    
    def PrivilegeInfoEstimate(self,aug_critic_obs):
        privilege_embedding = self.PrivilegeInfoEncoder(aug_critic_obs[:,self.env.num_obs:])
        aug_critic_obs = torch.concat((aug_critic_obs[:,:self.env.num_obs],privilege_embedding),dim=-1)
        return aug_critic_obs

    def act(self, obs, critic_obs, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        # Compute the actions and values
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        


        if self.IsObservationEstimation and self.env.cfg.env.state_est_dim>0:
            aug_obs,self.vel_ObservationEstimate,_ = self.ObservationEstimate(aug_obs,aug_critic_obs)

            if self.PrivilegeInfoEncoder is not None:
                # privilege_embedding = self.PrivilegeInfoEncoder(aug_critic_obs[:,self.env.num_obs:])
                # aug_critic_obs = torch.concat((aug_critic_obs[:,:self.env.num_obs],privilege_embedding),dim=-1)
                aug_critic_obs = self.PrivilegeInfoEstimate(aug_critic_obs)
                aug_critic_obs = aug_critic_obs.detach()


        self.actor_critic.act(aug_obs).detach()

        action_mean = self.actor_critic.action_mean.detach()
        # action_mean = self.env.CompensateUncertainty(action_mean)
        self.transition.action_mean = action_mean
        self.actor_critic.distribution.loc = action_mean

        self.transition.actions = self.actor_critic.distribution.sample()

        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, amp_obs, reward_list,desired_jpos_and_vel_list):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if 'time_outs' in infos:
            # print(infos['time_outs'])
            time_outs = infos['time_outs'].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)

        self.transition.reward_list=reward_list.clone()

        self.transition.desired_jpos_and_vel_list=desired_jpos_and_vel_list

        undesired_state = ((dones==True) & (infos['time_outs']==False)).type(torch.float32)
        # print(undesired_state)

        # dones True time_outs False return True
        self.transition.undesired_state=undesired_state
        
        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(
            self.amp_transition.observations, amp_obs)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        if self.PrivilegeInfoEncoder is not None:
            last_critic_obs = self.PrivilegeInfoEstimate(last_critic_obs)
    
        aug_last_critic_obs = last_critic_obs.detach()
        last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)   

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, reward_list_batch, undesired_state_batch, desired_jpos_and_vel_batch = sample
                
                aug_obs_batch = obs_batch.detach()
                aug_critic_obs_batch = critic_obs_batch.detach()
                
                if self.IsObservationEstimation and self.env.cfg.env.state_est_dim>0:
                    if self.env.cfg.env.useTrackingError:
                        obs = []
                        next_next_obs = aug_obs_batch[:,:self.num_observations-self.vel_dim]
                        next_desired_joint_pos = next_next_obs[:,-12:]*self.env.cfg.control.action_scale + self.env.default_dof_pos
                        
                        next_obs = aug_obs_batch[:,self.num_observations:
                                                 2*self.num_observations-self.vel_dim-self.env.cfg.env.current_addition_dim
                                                 ]
                        next_joint_pos = next_obs[:,6:18]/self.env.obs_scales.dof_pos+self.env.default_dof_pos
                        TrackingError_pred_label = next_desired_joint_pos - next_joint_pos
                        
                        for obs_id in range(2,self.env.include_history_steps):
                            obs_one = aug_obs_batch[:, obs_id * (self.num_observations) : (obs_id + 1) * (self.num_observations)]
                            if obs_id==2:
                                current_obs = obs_one
                                current_vel = obs_one[:,-self.vel_dim:]
                            obs.append(obs_one[:,:-self.vel_dim-self.env.cfg.env.current_addition_dim])
                        aug_obs_batch_train = torch.cat(obs, dim=-1)
                    
                    else:
                        obs = []
                        next_obs = aug_obs_batch[:,:self.num_observations-self.vel_dim-self.env.cfg.env.current_addition_dim]
                        
                        next_joint_pos = next_obs[:,6:18]

                        for obs_id in range(1,self.env.include_history_steps):
                            obs_one = aug_obs_batch[:, obs_id * (self.num_observations) : (obs_id + 1) * (self.num_observations)]
                            if obs_id==1:
                                current_obs = obs_one
                                current_vel = obs_one[:,-self.vel_dim:]
                            obs.append(obs_one[:,:-self.vel_dim-self.env.cfg.env.current_addition_dim])

                        if self.env.cfg.env.ObsEstModelName =="VAE" or "VQVAE":    
                            aug_obs_batch_train = torch.cat(obs, dim=-1)
                        elif self.env.cfg.env.ObsEstModelName =="TransformerVAE":
                            aug_obs_batch_train = torch.stack(obs, dim=1)
                        

                    # data = aug_obs_batch_train # 600,150
                    # recon_label_batch = next_obs # 600,30
                    # vel_label = current_vel # 600,20
                    if self.env.cfg.env.useRealData:
                        indices = torch.randint(low=0,high=self.data_real.size()[0],size=(aug_obs_batch_train.size()[0],))
                        data_r = self.data_real[indices]
                        recon_label_batch_r = self.recon_label_batch_real[indices]
                        vel_label_r = self.vel_label_real[indices]

                        data = torch.cat((aug_obs_batch_train,data_r),dim=0) # 1200,150
                        recon_label_batch = torch.cat((next_obs,recon_label_batch_r),dim=0) # 1200,30
                        vel_label = torch.cat((current_vel,vel_label_r),dim=0) # 1200,20
                    else:
                        data = aug_obs_batch_train # 600,150
                        recon_label_batch = next_obs # 600,30
                        vel_label = current_vel # 600,20

                        # reward_list_label = reward_list_batch.detach()
                        undesired_state_label = undesired_state_batch.detach()

                    if (self.env.cfg.env.ObsEstModelName =="VAE") or (self.env.cfg.env.ObsEstModelName =="TransformerVAE"):    
                        recon_batch, mu, logvar, vel = self.ObsEstModel(data)
                        
                    elif self.env.cfg.env.ObsEstModelName =="VQVAE":
                        self.ObsEstModel.train()
                        recon_batch, mu, vel = self.ObsEstModel(data)

                    
                    if self.env.cfg.env.useRewardDone:
                        reward_list_label=None
                        BCE,BCE_obs,BCE_reward,BCE_d, KLD, VEL_LOSS = self.ObsEstModel.loss_function(recon_batch, recon_label_batch, 
                                                                            mu, logvar,
                                                                            vel,vel_label,
                                                                            reward_list_label,
                                                                            undesired_state_label
                                                                            )
                        BCE, KLD, VEL_LOSS = BCE / len(data), KLD / len(data), VEL_LOSS/len(data)
                        BCE_obs,BCE_reward,BCE_d  = BCE_obs / len(data),  BCE_reward / len(data),  BCE_d / len(data)
                        C = 0.
                        self.beta =  0.01 # 10. # 1.KLD -->2.5 100. KLD-->0
                        alpha = 10. #10.  # 20 too large 1 too small
                        loss_obs_est = BCE_obs+10.*BCE_reward+10.*BCE_d + self.beta * torch.abs(KLD - C) + alpha*VEL_LOSS

                        if self.step_counter>self.step_counter_last:
                            self.step_counter_last=self.step_counter
                            print(
                                '| BCE', np.round(BCE.data.cpu().numpy(),4), 
                                '| BCE_obs', np.round(BCE_obs.data.cpu().numpy(),4), 
                                '| BCE_reward', np.round(BCE_reward.data.cpu().numpy(),4), 
                                '| BCE_d', np.round(BCE_d.data.cpu().numpy(),4), 

                                '| KL', np.round(KLD.data.cpu().numpy(), 4),
                                # '| C', C, 
                                # '| (KL-C) * beta', np.round(torch.abs(KLD - C).data.cpu().numpy() * self.beta, 4),
                                '| VEL_LOSS', np.round(VEL_LOSS.data.cpu().numpy(),4),
                                '| Loss', np.round(loss_obs_est.data.cpu().numpy(),4))
                    else:
                        if self.env.cfg.env.useTrackingError: 
                            # print("TrackingError_pred_label",TrackingError_pred_label.size())
                            BCE,BCE_TrackingError,KLD, VEL_LOSS = self.ObsEstModel.loss_function(recon_batch, recon_label_batch, 
                                                                                mu, logvar,
                                                                                vel,vel_label,
                                                                                TrackingError_pred_label=TrackingError_pred_label
                                                                                )
                            BCE,BCE_TrackingError, KLD, VEL_LOSS = BCE / len(data),BCE_TrackingError/len(data), KLD / len(data), VEL_LOSS/len(data)
                        
                            C = 0.
                            self.beta =  0.01 # 10. # 1.KLD -->2.5 100. KLD-->0
                            alpha = 10. #10.  # 20 too large 1 too small
                            loss_obs_est = BCE+ BCE_TrackingError + self.beta * torch.abs(KLD - C) + alpha*VEL_LOSS

                            if self.step_counter>self.step_counter_last:
                                self.step_counter_last=self.step_counter
                                print(
                                    '| BCE', np.round(BCE.data.cpu().numpy(),4), 
                                    '| BCE_TrackingError', np.round(BCE_TrackingError.data.cpu().numpy(),4), 

                                    '| KL', np.round(KLD.data.cpu().numpy(), 4),
                                    # '| C', C, 
                                    # '| (KL-C) * beta', np.round(torch.abs(KLD - C).data.cpu().numpy() * self.beta, 4),
                                    '| VEL_LOSS', np.round(VEL_LOSS.data.cpu().numpy(),4),
                                    '| Loss', np.round(loss_obs_est.data.cpu().numpy(),4))
                        else:
                            if (self.env.cfg.env.ObsEstModelName =="VAE") or (self.env.cfg.env.ObsEstModelName =="TransformerVAE"): 
                                BCE,KLD, VEL_LOSS = self.ObsEstModel.loss_function(recon_batch, recon_label_batch, 
                                                                                mu, logvar,
                                                                                vel,vel_label
                                                                                )
                                BCE, KLD, VEL_LOSS = BCE / len(data), KLD / len(data), VEL_LOSS/len(data)
                            
                                C = 0.
                                self.beta =  0.01 
                                alpha = 1. 
                                beta = 0.
                                loss_obs_est = BCE + self.beta * torch.abs(KLD - C) + alpha*VEL_LOSS
                              
                                if self.PrivilegeInfoEncoder is not None:
                                    privilege_embedding = self.PrivilegeInfoEncoder(critic_obs_batch[:,self.env.num_obs:])
                                    loss_embedding = beta*torch.sum(torch.square(mu - privilege_embedding.detach()))/len(data)
                                    loss_obs_est = loss_obs_est + loss_embedding
                                    # loss_obs_est =  loss_embedding
                      

                                if self.step_counter>self.step_counter_last:
                                    self.step_counter_last=self.step_counter

                                    if self.PrivilegeInfoEncoder is not None:
                                        print(
                                        '| BCE', np.round(BCE.data.cpu().numpy(),4), 
                                        '| KL', np.round(KLD.data.cpu().numpy(), 4),
                                        # '| C', C, 
                                        # '| (KL-C) * beta', np.round(torch.abs(KLD - C).data.cpu().numpy() * self.beta, 4),
                                        '| VEL_LOSS', np.round(VEL_LOSS.data.cpu().numpy(),4),
                                        '| Loss_embedding', np.round(loss_embedding.data.cpu().numpy(),4),
                                        '| Loss', np.round(loss_obs_est.data.cpu().numpy(),4))
                                    else:                                        
                                        print(
                                        '| BCE', np.round(BCE.data.cpu().numpy(),4), 
                                        '| KL', np.round(KLD.data.cpu().numpy(), 4),
                                        # '| C', C, 
                                        # '| (KL-C) * beta', np.round(torch.abs(KLD - C).data.cpu().numpy() * self.beta, 4),
                                        '| VEL_LOSS', np.round(VEL_LOSS.data.cpu().numpy(),4),
                                        '| Loss', np.round(loss_obs_est.data.cpu().numpy(),4))
                                    
                            elif self.env.cfg.env.ObsEstModelName =="VQVAE":
                                BCE,KLD, VEL_LOSS = self.ObsEstModel.loss_function(recon_batch, recon_label_batch, 
                                                                                vel,vel_label
                                                                                
                                                                                )
                                BCE, KLD, VEL_LOSS = BCE / len(data), KLD / len(data), VEL_LOSS/len(data)
                            
                                C = 0.
                                self.beta =  1. # 10. # 1.KLD -->2.5 100. KLD-->0
                                alpha = 1. #10.  # 20 too large 1 too small
                                loss_obs_est = BCE+ self.beta * torch.abs(KLD - C) + alpha*VEL_LOSS

                                if self.step_counter>self.step_counter_last:
                                    self.step_counter_last=self.step_counter
                                    print(
                                        '| BCE', np.round(BCE.data.cpu().numpy(),4), 
                                        '| KL', np.round(KLD.data.cpu().numpy(), 4),
                                        # '| C', C, 
                                        # '| (KL-C) * beta', np.round(torch.abs(KLD - C).data.cpu().numpy() * self.beta, 4),
                                        '| VEL_LOSS', np.round(VEL_LOSS.data.cpu().numpy(),4),
                                        '| Loss', np.round(loss_obs_est.data.cpu().numpy(),4))
                                    
                           

                    aug_obs_batch,_,_ = self.ObservationEstimate(aug_obs_batch,aug_critic_obs_batch)
        
                self.actor_critic.act(aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)

                if self.PrivilegeInfoEncoder is not None:
                    aug_critic_obs_batch = self.PrivilegeInfoEstimate(aug_critic_obs_batch)
                
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

          
                    
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))

                expert_loss = torch.nn.MSELoss()(
                    expert_d, torch.ones(expert_d.size(), device=self.device))
                
                policy_loss = torch.nn.MSELoss()(
                    policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                
                amp_loss = 0.5 * (expert_loss + policy_loss)
               


                grad_pen_loss = self.discriminator.compute_grad_pen(
                    *sample_amp_expert, lambda_=10)

                # Compute total loss.
                if self.IsObservationEstimation:
                    if self.env.skills_descriptor_id is None:
                        if (("roll" in self.env.files) or ("stand_up" in self.env.files)) and self.env.cfg.env.state_est_dim>0:
                            if "roll" in self.env.files:
                                a_pre = mu_batch*self.env.cfg.control.action_scale+self.env.default_lie_dof_pos
                                a_label = next_joint_pos/self.env.obs_scales.dof_pos+self.env.default_dof_pos
                            elif "stand_up" in self.env.files:
                                a_pre = mu_batch*self.env.cfg.control.action_scale+self.env.default_dof_pos
                                a_label = self.env.default_dof_pos

                            loss_smoothing = torch.mean(torch.square(a_pre - a_label))

                            # print("a_pre",a_pre)
                            # print("a_label",a_label)
                            # print("loss_smoothing",loss_smoothing)

                            loss = (
                            surrogate_loss +
                            self.value_loss_coef * value_loss -
                            self.entropy_coef * entropy_batch.mean() +
                            loss_obs_est + loss_smoothing
                            )

                        elif self.env.cfg.env.state_est_dim>0:
                            loss = (
                            surrogate_loss +
                            self.value_loss_coef * value_loss -
                            self.entropy_coef * entropy_batch.mean() +
                            loss_obs_est
                            )
                        else:
                            loss = (
                            surrogate_loss +
                            self.value_loss_coef * value_loss -
                            self.entropy_coef * entropy_batch.mean()
                            )       
                    else:                        
                        loss = (
                        surrogate_loss +
                        self.value_loss_coef * value_loss -
                        self.entropy_coef * entropy_batch.mean() +
                        amp_loss + grad_pen_loss + loss_obs_est
                        )

                else:                    
                    loss = (
                        surrogate_loss +
                        self.value_loss_coef * value_loss -
                        self.entropy_coef * entropy_batch.mean() +
                        amp_loss + grad_pen_loss
                        )

           

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred
