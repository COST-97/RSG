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

from re import M
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn import functional as F

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                        # env,
                        num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                        device="cpu",
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        # self.env = env
        # print("actor_hidden_dims",actor_hidden_dims)

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers).to(device)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers).to(device)

        # print(f"Actor MLP: {self.actor}")
        # print(f"Critic MLP: {self.critic}")

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # mean = self.actor(observations)
        mean = self.actor(observations) 
    
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
        # actions_mean = self.actor(observations)
        # return actions_mean

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

class LegActorCritic(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                          device="cpu",
                        **kwargs):
        if kwargs:
            print("LegActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__(num_actor_obs=num_actor_obs,
                         num_critic_obs=num_critic_obs,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         fixed_std=fixed_std)

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_output_dim_a = int(num_actions / 4)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor1 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor2 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor3 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor4 = nn.Sequential(*actor_layers).to(device)



    def update_distribution(self, observations):
        mean1 = self.actor1(observations)
        mean2 = self.actor2(observations)
        mean3 = self.actor3(observations)
        mean4 = self.actor4(observations)
        mean = torch.cat((mean1, mean2, mean3, mean4), 1)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)


    def act_inference(self, observations):
        mean1 = self.actor1(observations)
        mean2 = self.actor2(observations)
        mean3 = self.actor3(observations)
        mean4 = self.actor4(observations)
        mean = torch.cat((mean1, mean2, mean3, mean4), 1)
        return mean
        
class JointActorCritic(ActorCritic):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                          device="cpu",
                        **kwargs):
        if kwargs:
            print("JointActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__(num_actor_obs=num_actor_obs,
                         num_critic_obs=num_critic_obs,
                         num_actions=num_actions,
                         actor_hidden_dims=actor_hidden_dims,
                         critic_hidden_dims=critic_hidden_dims,
                         activation=activation,
                         init_noise_std=init_noise_std,
                         fixed_std=fixed_std)

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_output_dim_a = int(num_actions / 12)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor1 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor2 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor3 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor4 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor5 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor6 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor7 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor8 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor9 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor10 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor11 = nn.Sequential(*actor_layers).to(device)

        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor12 = nn.Sequential(*actor_layers).to(device)



    def update_distribution(self, observations):
        mean1 = self.actor1(observations)
        mean2 = self.actor2(observations)
        mean3 = self.actor3(observations)
        mean4 = self.actor4(observations)
        mean5 = self.actor5(observations)
        mean6 = self.actor6(observations)
        mean7 = self.actor7(observations)
        mean8 = self.actor8(observations)
        mean9 = self.actor3(observations)
        mean10 = self.actor10(observations)
        mean11 = self.actor11(observations)
        mean12 = self.actor12(observations)
        mean = torch.cat((mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9, mean10, mean11, mean12), 1)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)


    def act_inference(self, observations):
        mean1 = self.actor1(observations)
        mean2 = self.actor2(observations)
        mean3 = self.actor3(observations)
        mean4 = self.actor4(observations)
        mean5 = self.actor5(observations)
        mean6 = self.actor6(observations)
        mean7 = self.actor7(observations)
        mean8 = self.actor8(observations)
        mean9 = self.actor3(observations)
        mean10 = self.actor10(observations)
        mean11 = self.actor11(observations)
        mean12 = self.actor12(observations)
        mean = torch.cat((mean1, mean2, mean3, mean4, mean5, mean6, mean7, mean8, mean9, mean10, mean11, mean12), 1)
        return mean

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
class VAE(nn.Module):
    def __init__(self,img_shape,vel_dim=3,latent_dim = 16,obs_decode_num=39,
                 nn_dim = 128, nn_dim2 = 64,activation='elu',useRewardDone=False,useTrackingError=False,useTrackingError_two=False
                 ):
        super(VAE, self).__init__()
        print("VAE input_dim:",img_shape)
        print("VAE vel_dim:",vel_dim)
        print("VAE latent_dim:",latent_dim)
        print("VAE obs_decode_num:",obs_decode_num)
        
        self.useRewardDone = useRewardDone
        self.useTrackingError = useTrackingError
        self.useTrackingError_two = useTrackingError_two

        reward_decode_num = 15
        d_decode_num = 2
        
        self.img_shape=img_shape
        self.ELU = get_activation(activation)

        self.FocalLoss = FocalLoss(gamma=2, alpha=0.2, size_average=False)

        # encoder
        self.fc1 = nn.Linear(img_shape, nn_dim)
        self.fc2 = nn.Linear(nn_dim, nn_dim2)
        
        self.fc21 = nn.Linear(nn_dim2, latent_dim)
        self.fc22 = nn.Linear(nn_dim2, latent_dim)
        
        # vel predict
        # self.fc23 = nn.Linear(nn_dim2, vel_dim)
        self.fc30 = nn.Linear(latent_dim, nn_dim2)
        self.fc40 = nn.Linear(nn_dim2, nn_dim)
        self.fc50 = nn.Linear(nn_dim, vel_dim)

        # next obs predict 
        self.fc3 = nn.Linear(latent_dim, nn_dim2)
        self.fc4 = nn.Linear(nn_dim2, nn_dim)
        self.fc5 = nn.Linear(nn_dim, obs_decode_num)

        if self.useRewardDone:
                # next obs predict 
            self.fc31 = nn.Linear(latent_dim, nn_dim2)
            self.fc41 = nn.Linear(nn_dim2, nn_dim)
            self.fc51 = nn.Linear(nn_dim, obs_decode_num)

            # next reward predict 
            self.fc32 = nn.Linear(latent_dim, nn_dim2)
            self.fc42 = nn.Linear(nn_dim2, nn_dim)
            self.fc52 = nn.Linear(nn_dim, reward_decode_num)

            # next undesired state predict 
            self.fc33 = nn.Linear(latent_dim, nn_dim2)
            self.fc43 = nn.Linear(nn_dim2, nn_dim)
            self.fc53 = nn.Linear(nn_dim, d_decode_num)  

        if self.useTrackingError:
            # next Tracking Error predict 
            self.fc34 = nn.Linear(latent_dim, nn_dim2)
            self.fc44 = nn.Linear(nn_dim2, nn_dim)
            self.fc54 = nn.Linear(nn_dim, 12)

    def encode(self, x):
        # h1 = F.leaky_relu(self.fc1(x))
        # h2 = F.leaky_relu(self.fc2(h1))
        h1 = self.ELU(self.fc1(x))
        h2 = self.ELU(self.fc2(h1))

        return self.fc21(h2), self.fc22(h2) #,self.fc23(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def reparameterize_static(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return std.add_(mu)

    def decode(self, z):
        h30 = self.ELU(self.fc30(z))
        h40 = self.ELU(self.fc40(h30))
        vel_pred = self.fc50(h40)

        if self.useRewardDone:
            h31 = self.ELU(self.fc31(z))
            h41 = self.ELU(self.fc41(h31))
            obs_next =  self.fc51(h41)

            h32 = self.ELU(self.fc32(z))
            h42 = self.ELU(self.fc42(h32))
            self.reward_pred = self.fc52(h42)

            h33 = self.ELU(self.fc33(z))
            h43 = self.ELU(self.fc43(h33))
            self.d_pred = self.fc53(h43)

        else:
            h3 = self.ELU(self.fc3(z))
            h4 = self.ELU(self.fc4(h3))
            obs_next = self.fc5(h4) 
         
        
        if self.useTrackingError:
            h34 = self.ELU(self.fc34(z))
            h44 = self.ELU(self.fc44(h34))
            self.TrackingError_pred = self.fc54(h44)
        
        return obs_next,vel_pred

    def getTrackingError_pred(self):
        return self.TrackingError_pred

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.img_shape))
        z = self.reparameterize(mu, logvar)
        obs_next_pred,vel_pred = self.decode(z)
        return obs_next_pred, mu, logvar, vel_pred

    def loss_function(self,recon_x, x, mu, logvar, vel, vel_label,
                      reward_list_label=None, undesired_state_label=None,
                      TrackingError_pred_label=None
                      ):
        BCE_obs = F.mse_loss(recon_x, x, reduction='sum')
        if self.useTrackingError:
            # print(self.TrackingError_pred, TrackingError_pred_label)
            BCE_TrackingError = F.mse_loss(self.TrackingError_pred, TrackingError_pred_label, reduction='sum')

        if self.useRewardDone:
            BCE_reward = F.mse_loss(self.reward_pred, reward_list_label, reduction='sum')
            BCE_d = self.FocalLoss.forward(self.d_pred,undesired_state_label)

    #     BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_shape), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        VEL_LOSS = F.mse_loss(vel,vel_label, reduction='sum')
        if self.useRewardDone:
            return (BCE_obs+BCE_reward+BCE_d),BCE_obs,BCE_reward,BCE_d, KLD, VEL_LOSS
        else:
            if self.useTrackingError:
                return BCE_obs, BCE_TrackingError,KLD, VEL_LOSS
            else:
                return BCE_obs, KLD, VEL_LOSS

import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim=-1)
        logpt = logpt.gather(1,target.type(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1).type(torch.int64))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()