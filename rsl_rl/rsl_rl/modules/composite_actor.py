import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .new_actor_critic import Actor
from rsl_rl.utils import unpad_trajectories


class Composite(nn.Module):
    def __init__(self, dim, num, is_GNN_train, is_weight_train, K=2) -> None:
        super().__init__()

        self.K = K
        self.propagate_matrix = None

        self.is_GNN_train = is_GNN_train 
        self.GNN_w = nn.Parameter(torch.rand([dim, dim]), requires_grad=True)

        self.is_weight_train = is_weight_train
        self.weight_w = nn.Parameter(torch.rand([1,num]), requires_grad=True)

    def set_adjacency(self, adjacency):
        normalized_degree_matrix = torch.diag_embed(adjacency.sum(1).pow(-0.5))
        self.propagate_matrix = (normalized_degree_matrix @ adjacency @ normalized_degree_matrix).pow(self.K)

    def forward(self, X, type):
        # TODO: 激活函数？
        X = torch.matmul(self.propagate_matrix, X)
        if type == 'square_stddev':
            if self.is_GNN_train:
                X = torch.matmul(X, self.GNN_w * self.GNN_w)
                X = torch.tanh(X)
            if self.is_weight_train:
                X = torch.matmul(self.weight_w * self.weight_w, X)
            else:
                X = X.mean(dim=1, keepdim=True)
        else:
            if self.is_GNN_train:
                X = torch.matmul(X, self.GNN_w)
                X = torch.tanh(X)
            if self.is_weight_train:
                X = torch.matmul(self.weight_w, X)
            else:
                X = X.mean(dim=1, keepdim=True)
        return torch.tanh(X)


class CompositeActor(nn.Module):
    is_composite = True
    def __init__(
        self,  
        num_actor_obs,
        num_actions,
        device,
        num_base_actor,
        is_constant_std,
        is_GNN_train,
        is_weight_train,
        actor_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        **kwargs):
        super(CompositeActor, self).__init__()

        # base actors
        self.num_base_actor = num_base_actor
        self.body_actors = []
        self.legs_actors = []
        self.joints_actors = []

        self.device = device

        # set_base_policy
        # for skill_name in ['forward_walk', 'left_turn', 'right_turn']:
        #     for part in ['body', 'leg', 'actuator']:
        #         if part == 'body':
        #             # f_read = open(path + part + '.pkl', 'rb')
        #             # dict = pickle.load(f_read)
        #             temp_actor = Actor(num_actor_obs,
        #             num_actions,
        #             device,
        #             actor_hidden_dims)
        #             # temp_actor.actor.load_state_dict(dict)
        #             self.body_actors.append(temp_actor)
        #         elif part == 'leg':
        #             temp_legs = []
        #             for i in range(4):
        #                 # f_read = open(path + part + str(i) + '.pkl', 'rb')
        #                 # dict = pickle.load(f_read)
        #                 temp_actor = Actor(num_actor_obs,
        #                 3,
        #                 device,
        #                 actor_hidden_dims)
        #                 # temp_actor.actor.load_state_dict(dict)
        #                 temp_legs.append(temp_actor)
        #             self.legs_actors.append(temp_legs)
        #         else:
        #             temp_joints = []
        #             for i in range(12):
        #                 # f_read = open(path + part + str(i) + '.pkl', 'rb')
        #                 # dict = pickle.load(f_read)
        #                 temp_actor = Actor(num_actor_obs,
        #                 1,
        #                 device,
        #                 actor_hidden_dims)
        #                 # temp_actor.actor.load_state_dict(dict)
        #                 temp_joints.append(temp_actor)
        #             self.joints_actors.append(temp_joints)

        # Given 2 adjacency matrix: 
        # 3*3
        self.skill_Adjacency = None
        # 17*17
        # self.structure_Adjacency = structure_Adjacency
        # TODO: structure_Adjacency应该在初始化时传入？
        self.structure_Adjacency = torch.FloatTensor(
            [   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
                [1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0.],
                [1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0.],
                [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
                [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
                [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
                [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]]).to(device)

        # composite network (adjacency, dim, num)
        self.body_composite_net = Composite(12, self.num_base_actor, is_GNN_train, is_weight_train).to(device)
        self.leg_composite_net = Composite(3, self.num_base_actor, is_GNN_train, is_weight_train).to(device)
        self.joint_composite_net = Composite(1, self.num_base_actor, is_GNN_train, is_weight_train).to(device)
        
        self.hierarchical_composite_net = Composite(12, 17, is_GNN_train, is_weight_train).to(device)
        self.hierarchical_composite_net.set_adjacency(self.structure_Adjacency)

        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.is_constant_std = is_constant_std
        if is_constant_std:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def load_base_actors(self, skills, skill_adjacency):
        assert self.num_base_actor == len(skills)

        for skill in skills:
            body_actor = Actor(self.num_actor_obs, self.num_actions, self.device, self.actor_hidden_dims)
            body_actor.load_state_dict(skill[2][0].model_weight)
            self.body_actors.append(body_actor)
            
            temp_legs = []
            for i in range(4):
                leg_actor = Actor(self.num_actor_obs, int(self.num_actions/4), self.device, self.actor_hidden_dims)
                leg_actor.load_state_dict(skill[2][i+1].model_weight)
                temp_legs.append(leg_actor)
            self.legs_actors.append(temp_legs)

            temp_joints = []
            for i in range(12):
                joint_actor = Actor(self.num_actor_obs, int(self.num_actions/12), self.device, self.actor_hidden_dims)
                joint_actor.load_state_dict(skill[2][i+5].model_weight)
                temp_joints.append(joint_actor)
            self.joints_actors.append(temp_joints)

        self.skill_Adjacency = torch.FloatTensor(skill_adjacency).to(self.device)
        self.body_composite_net.set_adjacency(self.skill_Adjacency)
        self.leg_composite_net.set_adjacency(self.skill_Adjacency)
        self.joint_composite_net.set_adjacency(self.skill_Adjacency)


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
        for pi in self.body_actors:
            pi.update_distribution(observations)
        for pis in self.legs_actors:
            for pi in pis:
                pi.update_distribution(observations)
        for pis in self.joints_actors:
            for pi in pis:
                pi.update_distribution(observations)

        mean_feature = self.global_feature('mean')
        mean = self.hierarchical_composite_net(mean_feature, 'mean')
        mean = mean.squeeze(1)
        if self.is_constant_std:
            stddev = self.std.to(self.device)
        else:
            square_stddev_feature = self.global_feature('square_stddev')
            square_stddev = self.hierarchical_composite_net(square_stddev_feature, 'square_stddev')
            stddev = square_stddev.squeeze(1)

        self.distribution = Normal(mean, stddev)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        self.update_distribution(observations)
        action_feature = self.global_feature('action', observations)
        return self.hierarchical_composite_net(action_feature, 'action').squeeze(1)

    def F(self, pi, type, observations):
        if type == 'mean':
            return (pi.action_mean).detach()
        elif type == 'square_stddev':
            return (pi.action_std.pow(2)).detach()
        else:
            return (pi.act_inference(observations)).detach()


    def local_feature(self, actors, dim, type, observations):
        batchsize = actors[0].action_mean.shape[0]

        feature = torch.zeros(batchsize, self.num_base_actor, dim).to(self.device)
        index = 0

        for pi in actors:
            feature[:,index,:] = self.F(pi, type, observations)
            index += 1

        return feature


    def global_feature(self, type, observations=None):
        body_feature = self.local_feature(self.body_actors, 12, type, observations)
        body_embedding = self.body_composite_net(body_feature, type)

        batchsize = body_embedding.shape[0]
        leg_embedding = torch.zeros(batchsize, 4, 12).to(self.device)
        for i in range(4):
            currect_leg_actors = [x[i] for x in self.legs_actors]
            leg_feature = self.local_feature(currect_leg_actors, 3, type, observations)
            leg_embedding[:,i,3*i:3*i+3] = self.leg_composite_net(leg_feature, type).squeeze(1)

        joint_embedding = torch.zeros(batchsize, 12, 12).to(self.device)
        for i in range(12):
            currect_joint_actors = [x[i] for x in self.joints_actors]
            joint_feature = self.local_feature(currect_joint_actors, 1, type, observations)
            joint_embedding[:,i,i:i+1] = self.joint_composite_net(joint_feature, type).squeeze(1)

        return torch.cat((body_embedding, leg_embedding, joint_embedding), 1)

    # def sub_composite(self, dim, actors, actioncomposite):
    #     batchsize = actors[0].action_mean.shape[0]
    #     # TODO: device?
    #     device = torch.device('cuda:0')
    #     mean_feature = torch.zeros(batchsize, self.num_base_actor, dim).to(device)
    #     square_stddev_feature = torch.zeros(batchsize, self.num_base_actor, dim).to(device)
    #     index = 0

    #     for pi in actors:
    #         mean_feature[:,index,:] = pi.action_mean
    #         square_stddev_feature[:,index,:] = pi.action_std ** 2
    #         index += 1
        
    #     mean = actioncomposite(mean_feature)
    #     square_stddev = actioncomposite(square_stddev_feature)
    #     return mean, square_stddev


    # def global_composite(self):
    #     # TODO: device?
    #     device = torch.device('cuda:0')
    #     body_mean, body_square_stddev = self.sub_composite(12, self.body_actors, self.body_composite_net)
        
    #     batchsize = body_mean.shape[0]
    #     leg_mean = torch.zeros(batchsize, 4, 12).to(device)
    #     leg_square_stddev = torch.zeros(batchsize, 4, 12).to(device)
    #     for i in range(4):
    #         currect_leg_actors = [x[i] for x in self.legs_actors]
    #         currect_leg_mean, currect_leg_square_stddev = self.sub_composite(3, currect_leg_actors, self.leg_composite_net)
    #         leg_mean[:,i,3*i:3*i+3] = currect_leg_mean.squeeze(1)
    #         leg_square_stddev[:,i,3*i:3*i+3] = currect_leg_square_stddev.squeeze(1)

    #     joint_mean = torch.zeros(batchsize, 12, 12).to(device)
    #     joint_square_stddev = torch.zeros(batchsize, 12, 12).to(device)
    #     for i in range(12):
    #         currect_joint_actors = [x[i] for x in self.joints_actors]
    #         currect_joint_mean, currect_joint_square_stddev = self.sub_composite(1, currect_joint_actors, self.joint_composite_net)
    #         joint_mean[:,i,i:i+1] = currect_joint_mean.squeeze(1)
    #         joint_square_stddev[:,i,i:i+1] = currect_joint_square_stddev.squeeze(1)

    #     global_mean = torch.cat((body_mean, leg_mean, joint_mean), 1)
    #     global_square_stddev = torch.cat((body_square_stddev, leg_square_stddev, joint_square_stddev), 1)

    #     mean = self.hierarchical_composite_net(global_mean)
    #     square_stddev = self.hierarchical_composite_net(global_square_stddev)

    #     return mean, square_stddev