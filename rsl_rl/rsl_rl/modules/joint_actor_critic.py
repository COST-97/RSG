import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from actor_critic import ActorCritic, get_activation


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

        self.actors = []
        # Policy
        for _ in range(12):
            actor_layers = []
            actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dims)):
                if l == len(actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], mlp_output_dim_a))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actors.append(nn.Sequential(*actor_layers))



    def update_distribution(self, observations):
        means = [actor(observations) for actor in self.actors]
        mean = torch.cat((means[0], means[1], means[2], means[3],
                        means[4], means[5], means[6], means[7],
                        means[8], means[9], means[10], means[11]), 1)
        std = self.std.to(mean.device)
        self.distribution = Normal(mean, mean*0. + std)


    def act_inference(self, observations):
        means = [actor(observations) for actor in self.actors]
        mean = torch.cat((means[0], means[1], means[2], means[3],
                        means[4], means[5], means[6], means[7],
                        means[8], means[9], means[10], means[11]), 1)
        return mean