import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .new_actor_critic import Actor
# from rsl_rl.utils import unpad_trajectories


class GNNComposite(nn.Module):
    def __init__(self, dim, num, is_sg_GNN_train, K=2) -> None:
        super().__init__()
        # TODO: more layers are need!
        self.K = K
        self.dim = dim
        self.num = num
        self.propagate_matrix = None

        self.is_sg_GNN_train = is_sg_GNN_train 
        # self.GNN = nn.Parameter(torch.rand([dim, dim]), requires_grad=True)
        # self.GNN = nn.Parameter(torch.ones([dim, dim]) / dim, requires_grad=False)

        # self.GNN = nn.Parameter(torch.ones([dim]).uniform_(0.9,1.1), requires_grad=True) # 0.8~1.1
        # self.weight = nn.Parameter(torch.ones([1,num]).uniform_(0.9,1.1)/num, requires_grad=True) # 0.8~1.1
        # self.GNN = nn.Parameter(torch.ones([dim]).uniform_(0.99,1.01), requires_grad=True) # 0.8~1.1
        
        # self.GNN = nn.Parameter(torch.ones([dim]).uniform_(-0.1,0.1), requires_grad=True) # 0.8~1.1
        self.bias = nn.Parameter(torch.ones([dim]), requires_grad=True) # 0.8~1.1
        
        # self.weight = nn.Parameter(torch.ones([1,num]).uniform_(0.8,1.2)/num, requires_grad=True) # 0.8~1.1
        self.weight = nn.Parameter(torch.ones([1,num]), requires_grad=True) # 0.8~1.1
        
        self.soft_max = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def set_adjacency(self, adjacency):
        normalized_degree_matrix = torch.diag_embed(adjacency.sum(1).pow(-0.5))
        self.propagate_matrix = (normalized_degree_matrix @ adjacency @ normalized_degree_matrix).pow(self.K)
        self.propagate_matrix = self.soft_max(self.propagate_matrix)

    def init_weight(self,device):
        self.bias = nn.Parameter(torch.ones([self.dim],device=device), requires_grad=True) # 0.8~1.1
        
        # self.weight = nn.Parameter(torch.ones([1,self.num],device=device).uniform_(0.8,1.2)/self.num, requires_grad=True) # 0.8~1.1
        self.weight = nn.Parameter(torch.ones([1,self.num],device=device), requires_grad=True) # 0.8~1.1

    def get_weight(self):
        weight = torch.cat((self.weight,self.bias.unsqueeze(0)),-1)
        return weight.detach().cpu().numpy()
    
    def set_weight(self,weight,device):
        self.weight.data = nn.Parameter(torch.from_numpy(weight[:self.num]).unsqueeze(0).float()).to(device)
        self.bias.data = nn.Parameter(torch.from_numpy(weight[self.num:]).float()).to(device)

    def forward(self, X, type):
        # print(type)
        # print("GNNComposite X.size1 ",X.size()) #  torch.Size([1, 3, 12])

        # TODO: 激活函数？
        # print("self.propagate_matrix",self.propagate_matrix)
        X = torch.matmul(self.propagate_matrix, X)
        # print("GNNComposite X.size2 ",X.size()) #  torch.Size([1, 3, 12])
        # input()

        # self.weight = self.weight.clip_(min=0.0001,max=0.04)
        
        weight = self.soft_max(self.weight)
        # weight = self.weight
        
        # print("GNNComposite ",weight,self.GNN) # torch.Size([1, 17, 12])
        
        if type == 'square_stddev':
            if self.is_sg_GNN_train:
                # X = torch.matmul(X, self.GNN * self.GNN)
                # X = X*self.GNN * self.GNN
                # X = torch.tanh(X)
                
                X = torch.matmul( weight*weight, X)
                # X = X + self.GNN_std
                # X = X.clip_(min=0.0001,max=0.09)
                X = 0.09*self.sigmoid(X) + 0.0001
                # print(X)
        else:
            if self.is_sg_GNN_train:
                # print("GNNComposite X.size 1",X.size()) #  torch.Size([1, 1, 12])
                # print("print(self.GNN)",self.GNN)
                # X = torch.matmul(X, self.GNN)
                # X = X*self.GNN
                # X = torch.tanh(X)

                # self.GNN = 0.1*torch.tanh(self.GNN)

                # print("weight",weight)
                # print("bias",0.1*self.tanh(self.bias))
                X = torch.matmul( weight, X)
                X = X + 0.1*self.tanh(self.bias)

                self.weight_test = weight
                self.bias_test = 0.1*self.tanh(self.bias)

                # print(X)
                # print("GNNComposite X.size 2",X.size()) #  torch.Size([1, 1, 12])
        # X = X.mean(dim=1, keepdim=True)
        # X = torch.matmul( self.weight, X)
        return X


class HierarchicalComposite(nn.Module):
    def __init__(self, num, is_hierarchical_weight_train, normalization_weight) -> None:
        super().__init__()
        # self.normalization_weight = nn.Parameter(torch.ones([1, num]) / normalization_weight, requires_grad=is_hierarchical_weight_train)
        # self.normalization_weight = nn.Parameter(torch.ones([1, num]).uniform_(0.8,1.2)/ normalization_weight, requires_grad=is_hierarchical_weight_train)
        self.num = num
        self.is_hierarchical_weight_train=is_hierarchical_weight_train
        self.normalization_weight = nn.Parameter(torch.ones([1, num]), requires_grad=is_hierarchical_weight_train)
        
        print(self.normalization_weight.size())
        # self.bias = nn.Parameter(torch.ones([12]).uniform_(-0.05,0.05), requires_grad=True) # 0.8~1.1
        self.bias = nn.Parameter(torch.ones([12]), requires_grad=True) # 0.8~1.1
        self.soft_max = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def init_weight(self,device):
        self.bias = nn.Parameter(torch.ones([12],device=device), requires_grad=True) # 0.8~1.1
        
        # self.weight = nn.Parameter(torch.ones([1,num]).uniform_(0.8,1.2)/num, requires_grad=True) # 0.8~1.1
        self.normalization_weight = nn.Parameter(torch.ones([1,self.num],device=device), requires_grad=self.is_hierarchical_weight_train) # 0.8~1.1

    def forward(self, X, type):
        # print(type)
        # print("HierarchicalComposite X.size 1",X.size()) # torch.Size([1, 17, 12])
        # print(X)

        weight = self.soft_max(self.normalization_weight)
        # print("HierarchicalComposite ",weight,self.bias) # torch.Size([1, 17, 12])
        
        if type == 'square_stddev':
            X = torch.matmul(weight*weight, X)
            # X = self.normalization_weight * self.normalization_weight * X
            # X = torch.tanh(X)
            # X = X + self.bias_std
            X = 0.09*self.sigmoid(X) + 0.0001
        else:
            # print("weight",weight)
            X = torch.matmul(weight, X)
            # X = self.normalization_weight* X
            # X = torch.tanh(X)
            # self.bias.clip_(min=-0.05,max=0.05)
            # X = X + self.bias.clip_(min=-0.05,max=0.05)
            X = X + 0.05*self.tanh(self.bias)
            
            # X_ = X_.clip_(min=X-0.05,max=X+0.05)
        # print("HierarchicalComposite X.size 2",X.size()) # torch.Size([1, 1, 12])    
        return X


class NewCompositeActor(nn.Module):
    is_composite = True
    def __init__(
        self,  
        env,
        num_actor_obs,
        num_actions,
        device,
        num_base_actor,
        is_constant_std,
        is_sg_GNN_train,
        is_hierarchical_weight_train,
        is_body_composite,
        is_leg_composite,
        is_joint_composite,
        actor_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        **kwargs):
        super(NewCompositeActor, self).__init__()

        # base actors
        print('i am new! hello!')
        self.env = env
        self.num_base_actor = num_base_actor
        self.body_actors = []
        self.legs_actors = []
        self.joints_actors = []

        self.device = device

        # Given 2 adjacency matrix: 
        # 3*3
        self.skill_Adjacency = None
        # 17*17
        # self.structure_Adjacency = structure_Adjacency
        # TODO: structure_Adjacency应该在初始化时传入？
        # TODO: delete?
        # self.structure_Adjacency = torch.FloatTensor(
        #     [   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
        #         [1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        #         [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0.],
        #         [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0.],
        #         [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1.],
        #         [1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
        #         [1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #         [1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #         [1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0.],
        #         [1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
        #         [1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
        #         [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0.],
        #         [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.],
        #         [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
        #         [1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
        #         [1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.]]).to(device)

        # composite network (adjacency, dim, num)
        self.body_composite_net = GNNComposite(12, self.num_base_actor, is_sg_GNN_train).to(device)
        self.leg_composite_net = GNNComposite(3, self.num_base_actor, is_sg_GNN_train).to(device)
        self.joint_composite_net = GNNComposite(1, self.num_base_actor, is_sg_GNN_train).to(device)
        
        # TODO:需要改的只有分层合成对吗？
        weight = is_body_composite + is_leg_composite  + is_joint_composite
        num = is_body_composite * 1 + is_leg_composite * 4 + is_joint_composite * 12
        
        self.hierarchical_composite_net = HierarchicalComposite(weight, is_hierarchical_weight_train, weight).to(device)

        assert is_body_composite + is_leg_composite + is_joint_composite > 0
        self.is_body_composite  = is_body_composite
        self.is_leg_composite   = is_leg_composite
        self.is_joint_composite = is_joint_composite

        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.is_constant_std = is_constant_std
        if is_constant_std:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=True)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


        # zhy
        # self.isTorqueComposite = False

        self.skill_Adjacency = nn.Parameter(torch.FloatTensor([[1.,1.],[1.,1.]]),requires_grad=True).to(self.device)
        self.body_composite_net.set_adjacency(self.skill_Adjacency)
        self.leg_composite_net.set_adjacency(self.skill_Adjacency)
        self.joint_composite_net.set_adjacency(self.skill_Adjacency)

        path1 = r"logs_415/standup/ActorCritic_EnvID_0_SkillID_0/Apr13_19-50-25_/model_500.pt"
        path2 = r"logs_415/forward_walking_fast/ActorCritic_EnvID_0_SkillID_0/Apr13_09-24-47_/model_400.pt"

        path3 = r"logs_415/backward_left_walking/ActorCritic_EnvID_0_SkillID_0/Apr13_09-24-37_/model_400.pt"
        path4 = r"logs_415/backward_right_walking/ActorCritic_EnvID_0_SkillID_0/Apr13_09-26-53_/model_400.pt"
        path5 = r"logs_415/backward_walking/ActorCritic_EnvID_0_SkillID_0/Apr13_09-24-46_/model_400.pt"
        
        path6 = r"logs_415/forward_left/ActorCritic_EnvID_0_SkillID_0/Apr13_09-27-09_/model_400.pt"
        path7 = r"logs_415/forward_right/ActorCritic_EnvID_0_SkillID_0/Apr13_09-28-01_/model_400.pt"
        path8 = r"logs_415/gallop/ActorCritic_EnvID_0_SkillID_0/Apr13_09-24-39_/model_400.pt"

        path9 = r"logs_415/spin_counterclockwise/ActorCritic_EnvID_0_SkillID_0/Apr13_09-28-17_/model_400.pt"

        path10 = r"logs_415/forward_walking/ActorCritic_EnvID_0_SkillID_0/Apr14_11-30-10_/model_500.pt"

        path_list = [path1,path2,path3,path4,path5,path6,path7,path8,path9]

        if self.is_body_composite:
            for path in [path1,path2]:
                body_actor = Actor(self.num_actor_obs, self.num_actions, self.device, self.actor_hidden_dims)
                # path = r"/home/amax/zhy/SciRobt23/AMP_for_hardware/logs/gallop/ActorCritic_EnvID_7_SkillID_5_LegLiftID_0/Feb26_11-03-37_/model_3000.pt"
                loaded_dict = torch.load(path)
                print("=="*50)
                model_state = loaded_dict['model_state_dict']
                for key in list(model_state.keys()):
                    if "critic" in key:
                        model_state.pop(key)

                body_actor.load_state_dict(model_state)

                self.body_actors.append(body_actor)



    def load_base_actors(self, skills, skill_adjacency, skill_adjacency_raw):
        assert self.num_base_actor == len(skills)
        # TODO:
        self.skill_Adjacency = torch.FloatTensor(skill_adjacency_raw).to(self.device)
        print(self.skill_Adjacency)
        # self.skill_Adjacency = torch.FloatTensor(skill_adjacency).to(self.device)
        
        self.body_actors = []
        self.legs_actors = []
        self.joints_actors = []

        for skill in skills:
            if self.is_body_composite:
                body_actor = Actor(self.num_actor_obs, self.num_actions, self.device, self.actor_hidden_dims)

                body_actor.load_state_dict(skill[2][0].model_weight)
                
                # print(skill[2][0].model_weight)

                self.body_actors.append(body_actor)
            
            if self.is_leg_composite:
                temp_legs = []
                for i in range(4):
                    leg_actor = Actor(self.num_actor_obs, int(self.num_actions/4), self.device, self.actor_hidden_dims)
                    leg_actor.load_state_dict(skill[2][i+1].model_weight)
                    temp_legs.append(leg_actor)
                self.legs_actors.append(temp_legs)
            
            if self.is_joint_composite:
                temp_joints = []
                for i in range(12):
                    joint_actor = Actor(self.num_actor_obs, int(self.num_actions/12), self.device, self.actor_hidden_dims)
                    joint_actor.load_state_dict(skill[2][i+5].model_weight)
                    temp_joints.append(joint_actor)
                self.joints_actors.append(temp_joints)

        
        self.body_composite_net.set_adjacency(self.skill_Adjacency)
        self.leg_composite_net.set_adjacency(self.skill_Adjacency)
        self.joint_composite_net.set_adjacency(self.skill_Adjacency)

        self.body_composite_net.init_weight(self.device)
        self.leg_composite_net.init_weight(self.device)
        self.joint_composite_net.init_weight(self.device)

        self.hierarchical_composite_net.init_weight(self.device)

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
        if self.is_body_composite:
            for pi in self.body_actors:
                pi.update_distribution(observations)

        if self.is_leg_composite:
            for pis in self.legs_actors:
                for pi in pis:
                    pi.update_distribution(observations)
        if self.is_joint_composite:
            for pis in self.joints_actors:
                for pi in pis:
                    pi.update_distribution(observations)

        mean_feature = self.global_feature('mean')

        # print("mean_feature",mean_feature.size())
        mean = self.hierarchical_composite_net(mean_feature, 'mean')
        

        mean = mean.squeeze(1)
        if self.is_constant_std:
            # print("self.std",self.std)
            stddev = self.std.to(self.device)
        else:
            square_stddev_feature = self.global_feature('square_stddev')
            square_stddev = self.hierarchical_composite_net(square_stddev_feature, 'square_stddev')
            stddev = square_stddev.pow(0.5).squeeze(1)

        # print("mean",mean.size())
        # print("stddev",stddev.size())
        
        self.distribution = Normal(mean, stddev)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
        # return self.distribution.mean
    
    def get_actions_log_prob(self, actions):
        # return self.body_actors[0].distribution.log_prob(actions).sum(dim=-1)
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        self.update_distribution(observations)
        action_feature = self.global_feature('action', observations)
        return self.hierarchical_composite_net(action_feature, 'action').squeeze(1)

    # def F(self, pi, type, observations):
    #     if type == 'mean':
    #         return (pi.action_mean).detach()
    #     elif type == 'square_stddev':
    #         return (pi.action_std.pow(2)).detach()
    #     else:
    #         return (pi.act_inference(observations)).detach()


    # def local_feature(self, actors, dim, type, observations):
    #     batchsize = actors[0].action_mean.shape[0]

    #     feature = torch.zeros(batchsize, self.num_base_actor, dim).to(self.device)
    #     index = 0

    #     for pi in actors:
    #         feature[:,index,:] = self.F(pi, type, observations)
    #         index += 1

    #     return feature

    #zhy 3/4
    def F(self, pi, type, observations):
        if type == 'mean':
            action_mean = (pi.action_mean)
            # if self.isTorqueComposite:
            #     actions_scaled = action_mean * self.env.cfg.control.action_scale
            #     desired_pos = actions_scaled + self.env.default_dof_pos
            #     action_mean = self.env.randomized_p_gains*(desired_pos - self.env.dof_pos) - self.env.randomized_d_gains*self.env.dof_vel
            
            return action_mean
        
        elif type == 'square_stddev':
            return (pi.action_std.pow(2))
        else:
            return (pi.act_inference(observations))


    def local_feature(self, actors, dim, type, observations):
        batchsize = actors[0].action_mean.shape[0]

        feature = torch.zeros(batchsize, self.num_base_actor, dim, requires_grad=False).to(self.device)
        index = 0

        for pi in actors:
            feature[:,index,:] = self.F(pi, type, observations)
            index += 1

        return feature

    def global_feature(self, type, observations=None):
        if self.is_body_composite:
            body_feature = self.local_feature(self.body_actors, 12, type, observations)
            # print("body_feature",body_feature.requires_grad) # false
            body_embedding = self.body_composite_net(body_feature, type)

        if self.is_leg_composite:
            batchsize = self.legs_actors[0][0].action_mean.shape[0]
            leg_embedding = torch.zeros(batchsize, 1, 12).to(self.device)
            for i in range(4):
                currect_leg_actors = [x[i] for x in self.legs_actors]
                leg_feature = self.local_feature(currect_leg_actors, 3, type, observations)
                leg_embedding[:,0,3*i:3*i+3] = self.leg_composite_net(leg_feature, type).squeeze(1)

        if self.is_joint_composite:
            batchsize = self.joints_actors[0][0].action_mean.shape[0]
            joint_embedding = torch.zeros(batchsize, 1, 12).to(self.device)
            for i in range(12):
                currect_joint_actors = [x[i] for x in self.joints_actors]
                joint_feature = self.local_feature(currect_joint_actors, 1, type, observations)
                joint_embedding[:,0,i:i+1] = self.joint_composite_net(joint_feature, type).squeeze(1)

        # TODO: how to cat???????????????????????
        if self.is_body_composite + self.is_leg_composite + self.is_joint_composite == 3:
            return torch.cat((body_embedding, leg_embedding, joint_embedding), 1)
        if self.is_body_composite + self.is_leg_composite + self.is_joint_composite == 2:
            if not self.is_body_composite:
                return torch.cat((leg_embedding, joint_embedding), 1)
            if not self.is_leg_composite:
                return torch.cat((body_embedding, joint_embedding), 1)
            if not self.is_joint_composite:
                return torch.cat((leg_embedding, body_embedding), 1)
        if self.is_body_composite + self.is_leg_composite + self.is_joint_composite == 1:
            if self.is_body_composite:
                return body_embedding
            if self.is_leg_composite:
                return leg_embedding
            if self.is_joint_composite:
                return joint_embedding
