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

        self.bias = nn.Parameter(torch.zeros([dim]), requires_grad=True) # 0.8~1.1
        
    
        self.weight = nn.Parameter(0.5*torch.ones([1,num]), requires_grad=True) # 0.8~1.1
        
        self.soft_max = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def set_adjacency(self, adjacency):

        self.weight.data = adjacency

    def init_weight(self,device):
        self.bias = nn.Parameter(torch.zeros([self.dim],device=device), requires_grad=True) # 0.8~1.1
        self.weight = nn.Parameter(torch.ones([1,self.num],device=device), requires_grad=True) # 0.8~1.1

    def get_weight(self):
        weight = torch.cat((self.weight,self.bias.unsqueeze(0)),-1)
        return weight.detach().cpu().numpy()
    
    def set_weight(self,weight,device):
        # print("weight",weight)
        self.weight.data = nn.Parameter(torch.from_numpy(weight[:self.num]).unsqueeze(0).float()).to(device)
        self.bias.data = nn.Parameter(torch.from_numpy(weight[self.num:]).float()).to(device)


    def forward(self, X, type):
        
        weight = self.soft_max(self.weight) # 1,3

        
        if type == 'square_stddev':
            if self.is_sg_GNN_train:

                
                X = torch.matmul( weight*weight, X)
                X = 0.09*self.sigmoid(X) + 0.0001
        else:
            if self.is_sg_GNN_train:

                X = torch.matmul( weight, X)
                X = X + 0.1*self.tanh(self.bias)

                self.weight_test = weight
                self.bias_test = 0.1*self.tanh(self.bias)

        return X

class NewCompositeActor(nn.Module):
    is_composite = True
    def __init__(
        self,  
        env,
        num_actor_obs,
        num_actions,
        device,
        # num_base_actor,
        is_constant_std,
        is_sg_GNN_train,


        actor_hidden_dims=[512, 256, 128],
        init_noise_std=1.0,
        **kwargs):
        super(NewCompositeActor, self).__init__()


        self.env = env
        # self.num_base_actor = num_base_actor
        self.device = device
        self.is_sg_GNN_train= is_sg_GNN_train

        self.IsObservationEstimation=self.env.cfg.env.IsObservationEstimation

        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.is_constant_std = is_constant_std
        if is_constant_std:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions), requires_grad=True)

        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False


        
    def set_skill(self,path_list):

        self.body_actors = []
        for path in path_list:
            body_actor = Actor(self.num_actor_obs, self.num_actions, self.device, self.actor_hidden_dims)
            loaded_dict = torch.load(path)
            print("=="*50)
            model_state = loaded_dict['model_state_dict']
            for key in list(model_state.keys()):
                if "critic" in key:
                    model_state.pop(key)

            body_actor.load_state_dict(model_state)
            self.body_actors.append(body_actor)

        self.num_base_actor = len(self.body_actors)

        self.body_composite_net = GNNComposite(12, self.num_base_actor, self.is_sg_GNN_train).to(self.device)
        if self.num_base_actor==1:
            self.skill_Adjacency = nn.Parameter(torch.FloatTensor([[1.]]),requires_grad=True).to(self.device)
        else:
            self.skill_Adjacency = nn.Parameter(torch.FloatTensor([[1.,1.]]),requires_grad=True).to(self.device)
        self.body_composite_net.set_adjacency(self.skill_Adjacency)
        self.body_composite_net.init_weight(self.device)




    def ObservationEstimate(self,aug_obs,ObsEstModel):

        obs_list = []
        current_obs = []
        for obs_id in range(self.env.include_history_steps-1):
            obs_one = aug_obs[:, obs_id * (self.num_obs) : (obs_id + 1) * (self.num_obs)]
            obs_list.append(obs_one[:,:-self.vel_dim])
            if obs_id==0:
                current_obs = obs_one
                
        aug_obs = torch.cat(obs_list, dim=-1)

        z_decode, z_mu, z_logvar, vel = ObsEstModel.forward(aug_obs)

  
        aug_obs = torch.cat((current_obs,z_mu), dim=-1).detach()
        return aug_obs,vel


    def load_base_actors(self, skills, scores_weight,isWalkingSlow=False):
        # assert self.num_base_actor == len(skills)
        
        self.num_base_actor = len(skills)
        

        
        self.body_actors = []
        self.body_actors_obs_est = []
        
        for skill in skills:
            if self.IsObservationEstimation:

                
                self.num_obs = self.env.num_obs
                if isWalkingSlow:
                    self.num_obs +=3


                self.vel_dim =3 #self.cfg['vel_dim']
                z_dim= 16 #self.cfg['z_dim']
                num_actor_obs_true = self.num_obs + z_dim 

                if self.env.include_history_steps is not None:
                    num_ObsEst_obs = (self.num_obs - self.vel_dim) * (self.env.include_history_steps-1)
                else:
                    num_ObsEst_obs = self.num_obs - self.vel_dim

   

                ObsEstModel = VAE(img_shape=num_ObsEst_obs,
                                vel_dim=self.vel_dim,
                                latent_dim=z_dim,
                                obs_decode_num=num_ObsEst_obs// (self.env.include_history_steps-1),
                                nn_dim = 128, 
                                nn_dim2 = 64,
                                activation='elu'
                                ).to(self.device)

                body_actor = Actor(num_actor_obs_true, 
                                    self.num_actions, 
                                    self.device, 
                                    self.actor_hidden_dims)

    
                print("=="*50)
       
                
                model_state_obs = skill["model_state_dict"]
                for key in list(model_state_obs.keys()):
                    if "critic" in key:
                        model_state_obs.pop(key)

                body_actor.load_state_dict(model_state_obs)

                self.body_actors.append(body_actor)
                
                model_state_obs_est = skill['observation_estimation_model_state_dict']
                ObsEstModel.load_state_dict(model_state_obs_est)

                self.body_actors_obs_est.append(ObsEstModel)
                # print(body_actor)
                # print(ObsEstModel)
            else:
                body_actor = Actor(self.num_actor_obs, self.num_actions, self.device, self.actor_hidden_dims)
                # body_actor.load_state_dict(skill[2][0].model_weight)
                body_actor.load_state_dict(skill)

                self.body_actors.append(body_actor)

        self.body_composite_net = GNNComposite(12, self.num_base_actor, self.is_sg_GNN_train).to(self.device)

        # self.body_composite_net.set_adjacency(self.skill_Adjacency)
        self.body_composite_net.set_adjacency(torch.tensor(scores_weight,device=self.device))

        self.body_composite_net.init_weight(self.device)

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
        if self.IsObservationEstimation:
            for i in range(len(self.body_actors)):
                ObsEstModel = self.body_actors_obs_est[i]
                observations_aug,vel = self.ObservationEstimate(observations,ObsEstModel)
                self.vel=vel
                self.body_actors[i].update_distribution(observations_aug)

        else:
            for pi in self.body_actors:
                pi.update_distribution(observations)

        mean_feature = self.global_feature('mean')

        mean = mean_feature

        mean = mean.squeeze(1)
        if self.is_constant_std:
            # print("self.std",self.std)
            stddev = self.std.to(self.device)
        else:
            square_stddev_feature = self.global_feature('square_stddev')

            stddev = square_stddev_feature.pow(0.5).squeeze(1)

        
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
        # action_feature = self.global_feature('mean', observations)
        # return action_feature.squeeze(1)
        return self.distribution.mean
    
    
    def F(self, pi, type, observations):
        if type == 'mean':
            action_mean = (pi.action_mean)

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
        
        body_feature = self.local_feature(self.body_actors, 12, type, observations)
        # print("body_feature",body_feature.requires_grad) # false
        body_embedding = self.body_composite_net(body_feature, type)
        return body_embedding

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
                 nn_dim = 128, nn_dim2 = 64,activation='elu'
                 ):
        super(VAE, self).__init__()

        self.img_shape=img_shape
        self.ELU = get_activation(activation)

        self.fc1 = nn.Linear(img_shape, nn_dim)
        self.fc2 = nn.Linear(nn_dim, nn_dim2)
        
        self.fc21 = nn.Linear(nn_dim2, latent_dim)
        self.fc22 = nn.Linear(nn_dim2, latent_dim)
        
        self.fc23 = nn.Linear(nn_dim2, vel_dim)

        self.fc3 = nn.Linear(latent_dim, nn_dim2)
        self.fc4 = nn.Linear(nn_dim2, nn_dim)
        self.fc5 = nn.Linear(nn_dim, obs_decode_num)

    def encode(self, x):
        # h1 = F.leaky_relu(self.fc1(x))
        # h2 = F.leaky_relu(self.fc2(h1))
        h1 = self.ELU(self.fc1(x))
        h2 = self.ELU(self.fc2(h1))

        return self.fc21(h2), self.fc22(h2),self.fc23(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def reparameterize_static(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return std.add_(mu)

    def decode(self, z):
        h3 = self.ELU(self.fc3(z))
        h4 = self.ELU(self.fc4(h3))
        return self.fc5(h4)
#         return torch.tanh(self.fc5(h4))

    def forward(self, x):
        mu, logvar, vel = self.encode(x.view(-1, self.img_shape))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, vel

    def loss_function(self,recon_x, x, mu, logvar, vel, vel_label):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
    #     BCE = F.binary_cross_entropy(recon_x, x.view(-1, img_shape), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        VEL_LOSS = F.mse_loss(vel,vel_label, reduction='sum')
        return BCE, KLD, VEL_LOSS
