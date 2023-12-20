import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.obs_dim = obs_dim
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device)
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0

        self.H = 2
    
    def insert(self, states, next_states):
        """Add new states to memory."""
        
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def insert_expert_data(self, states, next_states):
        """Add new states to memory."""
        
        self.states_source_value = states

        states = self.change_joint_order(states)
        next_states = self.change_joint_order(next_states)

        self.states_source = states

        states = states[:,:self.obs_dim]
        next_states = next_states[:,:self.obs_dim]

        # print("get_linear_vel_batch")
        # print(states[:,24:27])
        # print("get_angular_vel_batch")
        # print(states[:,27:30])

        # import matplotlib.pylab as plt

        # vel = np.concatenate((states[:,24:27].cpu().numpy(),states[:,27:30].cpu().numpy()),axis=-1)

        # plt.figure()
        # for i in range(6):
        #     plt.subplot(2,3,i+1)
        #     plt.plot(vel[:,i])
        # # plt.savefig("linear_vel.png")   
        # plt.show()  

        # input()


        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size


    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device),
                   self.next_states[sample_idxs].to(self.device))

    # def feed_forward_generator(self, num_mini_batch, mini_batch_size):
    #     for _ in range(num_mini_batch):
    #         sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            
    #         states = torch.zeros(mini_batch_size, self.obs_dim*self.H).to(self.device)
    #         next_states = torch.zeros(mini_batch_size, self.obs_dim*self.H).to(self.device)

    #         for i in range(mini_batch_size):
    #             if sample_idxs[i]-self.H>=0:
    #                 states_ = self.states[sample_idxs[i]-self.H:sample_idxs[i]]
    #                 states[i] = torch.reshape(states_, (1,-1))

    #                 next_states_ = self.next_states[sample_idxs[i]-self.H:sample_idxs[i]]
    #                 next_states[i] = torch.reshape(next_states_, (1,-1))

    #         yield (states, next_states)

    def get_full_frame_batch(self, num_frames):
        idxs = np.random.choice(
               self.states_source.shape[0], size=num_frames)
        return self.states_source[idxs]

    def change_joint_order(self,states):
        """
        FR,FL,RR,RL-->FL,FR,RL,RR

        """
        joint_pos = states[:,0:12]
        joint_vel = states[:,30:42]

        FR = joint_pos[:,0:3]
        FL = joint_pos[:,3:6]
        RR = joint_pos[:,6:9]
        RL = joint_pos[:,9:12] 

        joint_pos = torch.cat((FL,FR,RL,RR),dim=1)   

        FR_ = joint_vel[:,0:3]
        FL_ = joint_vel[:,3:6]
        RR_ = joint_vel[:,6:9]
        RL_ = joint_vel[:,9:12] 

        joint_vel = torch.cat((FL_,FR_,RL_,RR_),dim=1)   

        return torch.cat((joint_pos,states[:,12:30],joint_vel,states[:,42:]),dim=1)   

    def get_joint_pose_batch(self,frames):
        return frames[:,0:12]

    def get_joint_vel_batch(self,frames):
        return frames[:,30:42]

    def get_z_pos_batch(self,frames):
        return frames[:,42:43]

    def get_rpy_batch(self,frames):
        return frames[:,43:46]

    def get_linear_vel_batch(self,frames):
        # print("get_linear_vel_batch")
        # print(frames[:,24:27])
        return frames[:,24:27]

    def get_angular_vel_batch(self,frames):
        # print("get_angular_vel_batch")
        # print(frames[:,27:30])
        return frames[:,27:30]    

    def get_test_joint_pos(self,t_list):
        return  self.states_source_value[t_list,:12]   