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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import torch

class Hockey():

    def reset_idx(self, env_ids):
        self._reset_root_state(env_ids)

    def _reset_root_state(self, env_ids):
        robot_indices = self.root_indices[env_ids]
        # print(self.env.root_state[robot_indices, 0])
        self.env.root_state[robot_indices, :3] = self.default_base_pose[:3] + self.env.env_origins[env_ids]

        # isaacgym/root_state: 0~2: x y z,
        # 3~6: qx, qy qz, w,
        # 7~9: lin vel,
        # 9~13: ang vel
        rand_x = torch.tensor(np.random.uniform(-.5, .5, size=len(env_ids)), device=self.device)
        self.env.root_state[robot_indices, 0] += rand_x
        # print("rooty", self.env.root_state[robot_indices, 1])
        rand_y = torch.tensor(np.random.uniform(-.5, .5, size=len(env_ids)), device=self.device)
        self.env.root_state[robot_indices, 1] += rand_y
        # print(self.env.root_state[robot_indices, 0:2])

        self.env.root_state[robot_indices, 3:7] = self.default_base_pose[3:7]
        self.env.root_state[robot_indices, 7:] = 0.  # [7:10]: lin vel, [10:13]: ang vel

        # vel -x
        rand_vel = torch.tensor(np.random.uniform(-9.5, -6.5, size=len(env_ids)), device=self.device)
        self.env.root_state[robot_indices, 7] += rand_vel
        # vel -y
        rand_vel = torch.tensor(np.random.uniform(-1.5, 1.5, size=len(env_ids)), device=self.device)
        self.env.root_state[robot_indices, 8] += rand_vel



class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width

        self.terrain_name_list = [
                                'Indoor Floor', #0
                            'Grassland', #6
                            'Grass and Pebble', #9
                            'Steps', #10
                            'Grass and Sand', #11
                  
                            ]
        
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        # print("self.proportions",self.proportions)

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        # print(cfg.num_rows, cfg.num_cols)

        # self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.env_origins = np.zeros((cfg.num_cols,cfg.num_rows, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        # print( self.width_per_env_pixels)
        # print( self.length_per_env_pixels)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        # print("self.tot_rows",self.tot_rows)

        self.height_field_raw = np.zeros((self.tot_cols, self.tot_rows), dtype=np.int16)
        
        self.env_param = {}
        
        if cfg.curriculum: # base skill
            self.curriculum()

        elif cfg.selected:
            self.selected_terrain()

        elif cfg.terrain_name=="ComplexTerrain_Sequential_Case_1":
            self.sequential_case1()
                
        else: # complex terrain
            # self.env_param = {"id":[],"flat":[],"slope":[]}
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            # print("ijk:")
            # print(k,i,j)

            self.terrain_ind = (j,i)

            choice = np.random.uniform(0, 1)
            # difficulty = np.random.choice([0.5, 0.75, 0.9])
            difficulty = np.random.uniform(0, 1)
            
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
            

            
    def curriculum(self):
        # flat_list = []
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):

                difficulty = i / self.cfg.num_rows
                
                # difficulty = (self.cfg.num_rows-1) / self.cfg.num_rows

                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

                # flat,flat_list = CalculateFlatness(terrain.height_field_raw)
                # flat_list.append(flat)
                # print("flat",min(flat_list),",",max(flat_list))   
                # input()     

    def sequential_case1(self):
        # flat_list = []
        self.key_point = []

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
                if self.cfg.case==0:
                    if j==0 and i==0:
                
                        slope = 0.2
                        terrain.slope = slope              
                        terrain_utils.sloped_terrain(terrain, slope=slope)  # Uphill
                    elif j==1 and i ==0:
                        terrain.slope = 0
                        self.discrete_obstacles_case1_terrain(terrain, max_height=1.2, min_size=0.4, max_size=1,
                                                            num_rects=100)
                        
                    elif j==2 and i ==0:
                        terrain.slope = 0  
                        self.discrete_obstacles_case2_terrain(terrain, max_height=0.7,platform_size=1.)

                
                elif self.cfg.case==1:
                    if j==0 and i==0:
                        terrain.slope = 0  
                        self.discrete_obstacles_case20_terrain(terrain, max_height=2.6,platform_size=1.)     
                    elif j==1 and i ==0:
                        terrain.slope = 0

                       
                        self.discrete_obstacles_case21_terrain(terrain, max_height=2.6,platform_size=1.) 
                        
                      
                elif self.cfg.case==2:
                    self.complex_terrain(terrain)


                self.add_terrain_to_map(terrain, i, j)

                # flat,flat_list = CalculateFlatness(terrain.height_field_raw)
                flat_list=0
                self.env_param[str(j)+str(i)] = [np.mean(flat_list),terrain.slope]  

        self.key_point = np.array(self.key_point)*terrain.horizontal_scale

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.length_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    
    def make_terrain(self, choice, difficulty=1.):
        terrain = terrain_utils.SubTerrain("terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        
        # slope = difficulty * 0.4
        
        slope = np.random.uniform(low=0.15,high=0.25)
        # slope = np.random.uniform(low=self.cfg.slope[0],high=self.cfg.slope[1])
        slope = slope * difficulty
        step_height = 0.0 + 0.12 * difficulty
   
        
        amplitude = 0.0 + 0.6 * difficulty
        discrete_obstacles_height = 0.01 + difficulty * 0.15
        stepping_stones_size = 2.
        stone_distance = 0.8

        terrain_name = self.cfg.terrain_name

        if terrain_name == 'ComplexTerrain_NewEnv':
            slope = np.random.uniform(low=self.cfg.slope[0],high=self.cfg.slope[1])
            terrain_height = np.random.uniform(low=self.cfg.terrain_height[0],high=self.cfg.terrain_height[1])
            
            terrain.slope = slope   

            terrain_utils.sloped_terrain(terrain, slope=slope)  # Uphill
            terrain_utils.random_uniform_terrain(terrain, min_height=-terrain_height, max_height=terrain_height, step=0.005,
                                                    downsampled_scale=0.2)
            # terrain_name = self.terrain_name_list[np.random.randint(0,len(self.terrain_name_list))]


        elif terrain_name == 'ComplexTerrain_NewEnv_opt':
            slope = np.random.uniform(low=-0.35,high=0.3)
            terrain_height = np.random.uniform(low=0.01,high=0.1)
            
            terrain.slope = slope   

            terrain_utils.sloped_terrain(terrain, slope=slope)  # Uphill
            terrain_utils.random_uniform_terrain(terrain, min_height=-terrain_height, max_height=terrain_height, step=0.005,
                                                    downsampled_scale=0.2)
            # terrain_name = self.terrain_name_list[np.random.randint(0,len(self.terrain_name_list))]


        elif self.cfg.terrain_name == 'ComplexTerrain_Sequential':
            slope = np.random.uniform(low=0.01,high=0.2)
            slope = slope * difficulty
            step_height = 0.0 + 0.06 * difficulty
            amplitude = 0.0 + 0.3 * difficulty
            discrete_obstacles_height = 0.01 + difficulty * 0.06
            stepping_stones_size = 2.
            stone_distance = 0.6
            terrain_name = "HighSteps" # self.terrain_name_list[np.random.randint(0,len(self.terrain_name_list))]


        elif self.cfg.terrain_name == 'ComplexTerrain_Baseline':
            slope = difficulty * 0.8
            # step_height = 0.05 + 0.15 * difficulty
            step_height = 0.01 + 0.15 * difficulty

            discrete_obstacles_height = 0.01 + difficulty * 0.1
            
         
            stepping_stones_size = 0.1 * (1.05 - difficulty)

            stone_distance = 0.05 if difficulty==0 else 0.1
            
            # gap_size = 1. * difficulty
            # pit_depth = 1. * difficulty
            gap_size = 1. * difficulty
            pit_depth = 1. * difficulty

            if choice < self.proportions[0]:
                if choice < self.proportions[0]/ 2:
                    slope *= -1
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            elif choice < self.proportions[1]:
                terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
            elif choice < self.proportions[3]:
                if choice<self.proportions[2]:
                    step_height *= -1

                step_width=0.2
                # step_width = np.random.uniform(low=0.2, high=0.42)
                terrain_utils.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height, platform_size=3.)
                

            elif choice < self.proportions[4]:
                num_rectangles = 20
                rectangle_min_size = 1.
                rectangle_max_size = 2.
                terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
            elif choice < self.proportions[5]:
                terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
            elif choice < self.proportions[6]:
                gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
            else:
                pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        

        if terrain_name == 'Marble Slope Uphill':
            terrain.slope = slope  

            terrain_utils.sloped_terrain(terrain, slope=slope)  # Uphill

        elif terrain_name == 'Marble Slope Downhill':
            terrain.slope = -slope*1.2  

            terrain_utils.sloped_terrain(terrain, slope=-slope*1.2)  # Downhill
        
        elif terrain_name == 'Grassland':
            terrain.slope = 0.
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.06*difficulty, max_height=0.06*difficulty, step=0.005,
                                                 downsampled_scale=0.2)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.1*difficulty, max_height=0.16*difficulty, step=0.01,
            #                                      downsampled_scale=0.2)
            
        elif terrain_name == 'Indoor Floor':
            terrain.slope = 0.
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.0001, max_height=0.0001, step=0.005,
                                                 downsampled_scale=0.2)
            # terrain_utils.random_uniform_terrain(terrain, min_height=-0.1*difficulty, max_height=0.16*difficulty, step=0.01,
            #                                      downsampled_scale=0.2)
            
            
        elif terrain_name == 'Grassland Slope Uphill':
            terrain.slope=slope*0.4
            terrain_utils.sloped_terrain(terrain, slope=slope*0.4)  # Uphill
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.04*difficulty, max_height=0.04*difficulty, step=0.005,
                                                 downsampled_scale=0.2)

        elif terrain_name == 'Grassland Slope Downhill':
            terrain.slope=-slope
            terrain_utils.sloped_terrain(terrain, slope=-slope)  # Uphill
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.04*difficulty, max_height=0.04*difficulty, step=0.005,
                                                 downsampled_scale=0.2)
            
        elif terrain_name == 'Grass and Pebble':

            terrain.slope=0
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.04*difficulty, max_height=0.04*difficulty, step=0.005,
                                                 downsampled_scale=0.2)
            
            self.discrete_obstacles_terrain(terrain, min_height=0.01*difficulty, max_height=0.14*difficulty, min_size=0.1, max_size=0.4,
                                                          num_rects=1000,
                                                          platform_size=1)

        elif terrain_name == 'Steps':
            terrain.slope=0
            self.discrete_obstacles_terrain(terrain, min_height=0.06*difficulty, max_height=0.1*difficulty, min_size=0.4, max_size=1,
                                                          num_rects=100,
                                                          platform_size=1)

        elif terrain_name == 'Grass and Sand':
            terrain.slope=0

            terrain_utils.random_uniform_terrain(terrain, min_height=-0.04*difficulty, max_height=0.04*difficulty, step=0.005,
                                                 downsampled_scale=0.2)
            terrain_utils.wave_terrain(terrain, num_waves=1, amplitude=0.15)                                     
        
        elif terrain_name == 'Grass and Mud':
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.06*difficulty, max_height=0.06*difficulty, step=0.005,
                                                 downsampled_scale=0.2)

        elif terrain_name == 'Brushwood':
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.06*difficulty, max_height=0.06*difficulty, step=0.005,
                                                 downsampled_scale=0.2)
        elif terrain_name == 'Hills':
            # terrain_utils.wave_terrain(terrain, num_waves=3, amplitude=amplitude)
            terrain.slope = 0.5   #-5~5
            terrain_utils.wave_terrain(terrain, num_waves=1, amplitude=amplitude) 


        elif terrain_name == 'Discrete Obstacles':
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            slope = 0
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, 
                                                    rectangle_max_size, num_rectangles, platform_size=1.)
        
        elif terrain_name == 'Stepping Stones':
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, 
                                                    max_height=0., platform_size=2.,depth=-0.06)

        elif terrain_name == 'UpStairs':
            step_height *= -1  # 0.33
            terrain.slope = 0.4
            # print("terrain_name",terrain_name)
            # print(step_height)
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.3, step_height=step_height, platform_size=2.)


        elif terrain_name == 'DownStairs':
            step_height = step_height*1.1
            terrain.slope = 0.26 #(step_height/step_width)
            
            # print("terrain_name",terrain_name)
            # print(step_height)

            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.5, step_height=step_height, platform_size=2.)


        elif terrain_name == 'HighSteps':
            step_height = 0.1
            terrain.slope = 0.26 #(step_height/step_width)
            
            # print("terrain_name",terrain_name)
            # print(step_height)

            terrain_utils.pyramid_stairs_terrain(terrain, step_width=1.2, step_height=step_height, platform_size=1.5)

        elif terrain_name == 'DiscreteObstacles':
            slope = np.random.uniform(low=self.cfg.slope[0],high=self.cfg.slope[1])
            terrain_height = np.random.uniform(low=self.cfg.terrain_height[0],high=self.cfg.terrain_height[1])
            
            num_rectangles = 20
            rectangle_min_size = 2.
            rectangle_max_size = 4.
            slope = 0
            discrete_obstacles_height = 0.2
            terrain_utils.obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, 
                                                    rectangle_max_size, num_rectangles, platform_size=1.)
        
            
        elif terrain_name == 'ComplexTerrain_NewTask':
            slope = np.random.uniform(low=self.cfg.slope[0],high=self.cfg.slope[1])
            terrain_height = np.random.uniform(low=self.cfg.terrain_height[0],high=self.cfg.terrain_height[1])
            
            num_rectangles = 20
            rectangle_min_size = 2.
            rectangle_max_size = 4.
            slope = 0
            discrete_obstacles_height = 0.2
            terrain_utils.obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, 
                                                    rectangle_max_size, num_rectangles, platform_size=1.)

        
        return terrain           

    def make_complex_terrain(self, terrain, choice, difficulty):
        slope = difficulty * 0.4
        # step_height = 0.05 + 0.18 * difficulty
        step_height = 0.01 + 0.12 * difficulty
        # discrete_obstacles_height = 0.05 + difficulty * 0.2

        discrete_obstacles_height = 0.01 + difficulty * 0.15

        # stepping_stones_size = 1.5 * (1.05 - difficulty)
        # stone_distance = 0.05 if difficulty==0 else 0.1
        # stone_distance = 0.05 
        stepping_stones_size = 1.2
        stone_distance = 0.1

        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        
        # if self.terrain_ind[0] == 0 and self.terrain_ind[1] == 0:
        if choice < self.proportions[0]:
            slope = 0.
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.01, max_height=0.01, step=0.005,
                                                 downsampled_scale=0.2)
                        
 

        elif choice < self.proportions[1]:
            slope = 0.25
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.1)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.06, max_height=0.06, step=0.005, downsampled_scale=0.2)
        
        elif choice < self.proportions[3]:
            slope = 0
            self.discrete_obstacles_terrain(terrain, min_height=0.06*difficulty, max_height=0.1*difficulty, min_size=0.4, max_size=1,
                                                          num_rects=100,
                                                          platform_size=1)
            
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            slope = 0
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, 
                                                    rectangle_max_size, num_rectangles, platform_size=1.)
        elif choice < self.proportions[5]:
            slope = 0
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, 
                                                    max_height=0., platform_size=1.,depth=-0.12)
        else:
            slope = 0.15
            terrain_utils.wave_terrain(terrain, num_waves=3, amplitude=slope) 
        terrain.slope = slope    
        return terrain                       


    def discrete_obstacles_terrain(self, terrain, min_height, max_height, min_size, max_size, num_rects, platform_size=1.):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height = max_height / terrain.vertical_scale
        min_height = min_height / terrain.vertical_scale

        # min_size = int(min_size / terrain.horizontal_scale)
        # max_size = int(max_size / terrain.horizontal_scale)
        min_size = min_size / terrain.horizontal_scale
        max_size = max_size / terrain.horizontal_scale
        platform_size = int(platform_size / terrain.horizontal_scale)

        (i, j) = terrain.height_field_raw.shape
        # height_range = [max_height // 2, max_height]
        # width_range = range(min_size, max_size, 4)
        # length_range = range(min_size, max_size, 4)
        # width_range = range(min_size, max_size)
        # length_range = range(min_size, max_size)

        for _ in range(num_rects):
            # width = np.random.choice(width_range)
            # length = np.random.choice(length_range)
            width = np.random.uniform(low=min_size,high=max_size)
            length = np.random.uniform(low=min_size,high=max_size)
            start_i = np.random.choice(range(0, i - int(width), 4))
            start_j = np.random.choice(range(0, j - int(length), 4))
            # terrain.height_field_raw[start_i:start_i + width, start_j:start_j + length] = np.random.choice(height_range)
            terrain.height_field_raw[start_i:start_i + int(width), start_j:start_j + int(length)] = np.random.uniform(low=min_height,high=max_height)

        x1 = (terrain.width - platform_size) // 2
        x2 = (terrain.width + platform_size) // 2
        y1 = (terrain.length - platform_size) // 2
        y2 = (terrain.length + platform_size) // 2
        terrain.height_field_raw[x1:x2, y1:y2] = 0
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row #0
        j = col #0
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels

        self.height_field_raw[start_y:end_y, start_x: end_x] = terrain.height_field_raw


    
        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)

        if self.cfg.terrain_name == 'ComplexTerrain_Sequential_Case_1':
            if self.cfg.case==1:
                env_origin_y = 4. #init x pos
                env_origin_x = 6.5 #init y pos
            
                x1 = int(env_origin_x / terrain.horizontal_scale)
                x2 = int((env_origin_x+0.5) / terrain.horizontal_scale)

                y1 = int(env_origin_y / terrain.horizontal_scale)
                y2 = int((env_origin_y+1) / terrain.horizontal_scale)   
        
            elif self.cfg.case==2:
                env_origin_y = 3 #init x pos
                env_origin_x = 9.5 #init y pos
            
                x1 = int(env_origin_x / terrain.horizontal_scale)
                x2 = int((env_origin_x+1) / terrain.horizontal_scale)

                y1 = int(env_origin_y / terrain.horizontal_scale)
                y2 = int((env_origin_y+1) / terrain.horizontal_scale)   

        # env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        env_origin_z = np.max(terrain.height_field_raw[y1:y2, x1:x2])*terrain.vertical_scale

        self.env_origins[j, i] = [env_origin_y, env_origin_x, env_origin_z]

    def discrete_obstacles_case1_terrain(self, terrain, max_height, min_size, max_size, num_rects):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height = int(max_height / terrain.vertical_scale)
        min_size = int(min_size / terrain.horizontal_scale)
        max_size = int(max_size / terrain.horizontal_scale)
    

        (i, j) = terrain.height_field_raw.shape
        # height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
        height_range = [max_height-0.1, max_height-0.06, max_height-0.04, max_height]
        
        width_range = range(min_size, max_size, 4)
        length_range = range(min_size, max_size, 4)
        
        terrain.height_field_raw[:, :] = height_range[0]

        for _ in range(num_rects):
            width = np.random.choice(width_range)
            length = np.random.choice(length_range)
            start_i = np.random.choice(range(0, i-width, 4))
            start_j = np.random.choice(range(0, j-length, 4))
            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

        return terrain

    def discrete_obstacles_case2_terrain(self, terrain, max_height, platform_size=0.5):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height = int(max_height / terrain.vertical_scale)


        (i, j) = terrain.height_field_raw.shape
        # height_range = [-max_height, -max_height // 2, max_height // 2, max_height]

        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        y1 = terrain.length // 2 - platform_size
        y2 = terrain.length // 2 + platform_size

        distance = int(0.4 / terrain.horizontal_scale / 2)


        # Num=3
        # for j_id in range(Num):
            # terrain.height_field_raw[j_id*(j//Num):(j_id+1)*(j//Num),y1:y2] = max_height*(1-j_id/Num)
        terrain.height_field_raw[distance:distance*6,y1:y2] = max_height
        terrain.height_field_raw[7*distance:distance*13,y1:y2] = max_height*0.8
        terrain.height_field_raw[14*distance:distance*20,y1:y2] = max_height*0.6
        
        return terrain
    

    def discrete_obstacles_case20_terrain(self, terrain, max_height,platform_size=0.5):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height_scale = int(max_height / terrain.vertical_scale)


        # (i, j) = terrain.height_field_raw.shape
        # height_range = [-max_height, -max_height // 2, max_height // 2, max_height]

        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        y1 = terrain.length - 4*platform_size
        # y2 = terrain.length // 2 + platform_size
        y2 = terrain.length - 2*platform_size

        # Num=3
        # for j_id in range(Num):
            # terrain.height_field_raw[j_id*(j//Num):(j_id+1)*(j//Num),y1:y2] = max_height*(1-j_id/Num)
        for i in range(0,12,2):
            terrain.height_field_raw[platform_size*i:platform_size*(i+1),y2:y2+platform_size] = int((max_height+1.5) / terrain.vertical_scale)

        terrain.height_field_raw[:,y1:y2] = max_height_scale

        # terrain.height_field_raw[:,y1:y1+1] = int(max_height+0.1 / terrain.vertical_scale)
        # terrain.height_field_raw[:,y2-1:y2] = int(max_height+0.1 / terrain.vertical_scale)
        
        return terrain

    def discrete_obstacles_case21_terrain(self, terrain, max_height, platform_size=0.5):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height_scale = int(max_height / terrain.vertical_scale)


        # (i, j) = terrain.height_field_raw.shape
        # height_range = [-max_height, -max_height // 2, max_height // 2, max_height]

        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        y1 = terrain.length - 4*platform_size
        y2 = terrain.length - 2*platform_size
        
        x_half = terrain.width - 2*platform_size
        # y_half = terrain.length // 2
        y_half = terrain.length - 2*platform_size

        # Num=3
        # for j_id in range(Num):
            # terrain.height_field_raw[j_id*(j//Num):(j_id+1)*(j//Num),y1:y2] = max_height*(1-j_id/Num)
        # print(y1)
        for i in range(0,12,2):
            terrain.height_field_raw[platform_size*i:platform_size*(i+1),y2:y2+platform_size] = int((max_height+1.5) / terrain.vertical_scale)

        terrain.height_field_raw[:x_half - 6*platform_size,y1:y2] = max_height_scale
        terrain.height_field_raw[x_half - 6*platform_size:,y1:y2] = int((max_height+1.) / terrain.vertical_scale)
        
        # terrain.height_field_raw[:,y1:y1+1] = int(max_height+0.1 / terrain.vertical_scale)
        # terrain.height_field_raw[:,y2-1:y2] = int(max_height+0.1 / terrain.vertical_scale)

        distance = int(0.4 / terrain.horizontal_scale / 2)

        # case 2
        terrain.height_field_raw[0:2*platform_size,
                            y_half - 4*platform_size:y_half-2*platform_size] = int((max_height-0.4) / terrain.vertical_scale)
        
        terrain.height_field_raw[2*platform_size:4*platform_size,
                                y_half - 4*platform_size:y_half-2*platform_size] = int((max_height-0.8) / terrain.vertical_scale)
        
        terrain.height_field_raw[4*platform_size:,
                                y_half - 4*platform_size:y_half-2*platform_size] = int((max_height-0.8) / terrain.vertical_scale)
        

        terrain.height_field_raw[x_half - 3*platform_size:x_half - 1*platform_size,
                                y_half - 6*platform_size:y_half-4*platform_size] = int((max_height-1.2) / terrain.vertical_scale)
                
        terrain.height_field_raw[:x_half - 3*platform_size,
                                y_half - 6*platform_size:y_half-4*platform_size] = int((max_height-1.6) / terrain.vertical_scale)
        
        terrain.height_field_raw[:4*platform_size,
                                y_half-8*platform_size:y_half-6*platform_size] = int((max_height-1.6) / terrain.vertical_scale)
        
        terrain.height_field_raw[:x_half - 2*platform_size,
                                y_half-10*platform_size:y_half-8*platform_size] = int((max_height-1.6) / terrain.vertical_scale)
        
        terrain.height_field_raw[x_half - 2*platform_size:,
                                y_half-10*platform_size:y_half-8*platform_size] = int((max_height+1.) / terrain.vertical_scale)


        max_height = 0.25 / terrain.vertical_scale
        
        
        terrain_scale = int(self.env_length / 8) # 6
        # for x in range(int(5.0 * terrain_scale /terrain.horizontal_scale), int(6.5 * terrain_scale /terrain.horizontal_scale)):
        for x in range(int(1. * terrain_scale /terrain.horizontal_scale), int(4.5 * terrain_scale /terrain.horizontal_scale)):
            
            terrain.height_field_raw[x, np.arange(int(0. * terrain_scale /terrain.horizontal_scale), int(2. * terrain_scale /terrain.horizontal_scale))] = int((max_height-2.3 + max_height) * (x - int(1. * terrain_scale /terrain.horizontal_scale)) / int(2. * terrain_scale /terrain.horizontal_scale))
        
        return terrain    

    def discrete_obstacles_case22_terrain(self, terrain, max_height, platform_size=0.5):
        """
        Generate a terrain with gaps

        Parameters:
            terrain (terrain): the terrain
            max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
            min_size (float): minimum size of a rectangle obstacle [meters]
            max_size (float): maximum size of a rectangle obstacle [meters]
            num_rects (int): number of randomly generated obstacles
            platform_size (float): size of the flat platform at the center of the terrain [meters]
        Returns:
            terrain (SubTerrain): update terrain
        """
        # switch parameters to discrete units
        max_height = int(max_height / terrain.vertical_scale)


        (i, j) = terrain.height_field_raw.shape
        # height_range = [-max_height, -max_height // 2, max_height // 2, max_height]

        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        y1 = terrain.length // 2 - platform_size
        y2 = terrain.length // 2 + platform_size
        
        x1 = terrain.width // 2 - platform_size
        x2 = terrain.width // 2 + platform_size

 
        distance = int(0.4 / terrain.horizontal_scale / 2)

        terrain.height_field_raw[x1:x2,y1:y2] = max_height
     
        
        return terrain    


    def complex_terrain(self,terrain):
        terrain_scale = int(self.env_length / 8) # 6
        # print("terrain_scale",terrain_scale)
        
        key_point_list = []

        # platform 1
        key_point_list.append([3.,1.5])

        terrain.height_field_raw[int(0.5 * terrain_scale / terrain.horizontal_scale): int(3.0 * terrain_scale / terrain.horizontal_scale),
                                    int(1.5 * terrain_scale /terrain.horizontal_scale): int(6.5 * terrain_scale / terrain.horizontal_scale)] = 2.0 / terrain.vertical_scale

        downsampled_scale = 0.2

        min_height = int(-0.05 / terrain.vertical_scale)
        max_height = int(0.05 / terrain.vertical_scale)
        step = int(0.005 / terrain.vertical_scale)

        heights_range = np.arange(min_height, max_height + step, step)
        height_field_downsampled = np.random.choice(heights_range, (
            int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
                terrain.length * terrain.horizontal_scale / downsampled_scale)))

        x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
        y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

        f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

        x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
        y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
        z_upsampled = np.rint(f(y_upsampled, x_upsampled))

        for x in range(int(1.5 * terrain_scale / terrain.horizontal_scale),
                        int(3.0 * terrain_scale / terrain.horizontal_scale)):
            terrain.height_field_raw[x, np.arange(int(1.5 * terrain_scale / terrain.horizontal_scale),
                                                    int(6.5 * terrain_scale / terrain.horizontal_scale))] += \
                z_upsampled.astype(np.int16)[x, np.arange(int(1.5 * terrain_scale / terrain.horizontal_scale),
                                                            int(6.5 * terrain_scale / terrain.horizontal_scale))]



        # add big steps on platform 1

        terrain.height_field_raw[int(1.0 * terrain_scale/ terrain.horizontal_scale): int(1.5 * terrain_scale / terrain.horizontal_scale),
                                    int(2.0 * terrain_scale/terrain.horizontal_scale): int(6.5 * terrain_scale / terrain.horizontal_scale)] = 1.6 / terrain.vertical_scale

        terrain.height_field_raw[int(0.5 * terrain_scale/ terrain.horizontal_scale): int(1.0 * terrain_scale / terrain.horizontal_scale),
                                    int(2.0 * terrain_scale/terrain.horizontal_scale): int(6.5 * terrain_scale / terrain.horizontal_scale)] = 1.2 / terrain.vertical_scale

        terrain.height_field_raw[int(0 * terrain_scale/ terrain.horizontal_scale): int(0.5 * terrain_scale / terrain.horizontal_scale),
                                    int(2.0 * terrain_scale/terrain.horizontal_scale): int(6.5 * terrain_scale / terrain.horizontal_scale)] = 0.8 / terrain.vertical_scale


        # add a slope on platform 1
        for y in range(int(2.0 * terrain_scale / terrain.horizontal_scale), int(3.5 * terrain_scale / terrain.horizontal_scale)):
            terrain.height_field_raw[np.arange(int(0.0 * terrain_scale / terrain.horizontal_scale), int(0.5 * terrain_scale / terrain.horizontal_scale)), y] += \
                int(1.2 / terrain.vertical_scale * (int(3.5 * terrain_scale / terrain.horizontal_scale) - y) / int(1.5 * terrain_scale / terrain.horizontal_scale))
        
        terrain.height_field_raw[int(0 * terrain_scale/ terrain.horizontal_scale): int(0.5 * terrain_scale / terrain.horizontal_scale),
                                    int(1.5 * terrain_scale/terrain.horizontal_scale): int(2.0 * terrain_scale / terrain.horizontal_scale)] = 2.0 / terrain.vertical_scale




        # platform 2
        terrain.height_field_raw[int(5.0 * terrain_scale / terrain.horizontal_scale): int(7.5 * terrain_scale / terrain.horizontal_scale),
                                    int(1.5 * terrain_scale /terrain.horizontal_scale): int(6.5 * terrain_scale / terrain.horizontal_scale)] = 2.0 / terrain.vertical_scale


        # add a slope on platform 2
        max_height = 0.4 / terrain.vertical_scale

        for x in range(int(5.0 * terrain_scale /terrain.horizontal_scale), int(6.5 * terrain_scale /terrain.horizontal_scale)):
            terrain.height_field_raw[x, np.arange(int(1.5 * terrain_scale /terrain.horizontal_scale), int(4.0 * terrain_scale /terrain.horizontal_scale))] += int(max_height * (x - int(5.0 * terrain_scale /terrain.horizontal_scale)) / int(1.5 * terrain_scale /terrain.horizontal_scale))
        
        for x in range(int(6.5 * terrain_scale /terrain.horizontal_scale), int(7.5 * terrain_scale /terrain.horizontal_scale)):
            terrain.height_field_raw[x, np.arange(int(1.5 * terrain_scale /terrain.horizontal_scale), int(6.5 * terrain_scale /terrain.horizontal_scale))] += int(max_height)

        # add steps on platform 2
        for x in range(int(6.0 * terrain_scale /terrain.horizontal_scale), int(6.5 * terrain_scale /terrain.horizontal_scale)):
            terrain.height_field_raw[x, np.arange(int(4.0 * terrain_scale /terrain.horizontal_scale), int(6.5 * terrain_scale /terrain.horizontal_scale))] += int(max_height / 2)


        # flat bridge
        terrain.height_field_raw[int(3.0 * terrain_scale / terrain.horizontal_scale): int(5.0 * terrain_scale / terrain.horizontal_scale),
                                    int(2.0 * terrain_scale /terrain.horizontal_scale): int(2.5 * terrain_scale / terrain.horizontal_scale)] = 2.0 / terrain.vertical_scale

        # wave bridge
        terrain.height_field_raw[int(3.0 * terrain_scale / terrain.horizontal_scale): int(5.0 * terrain_scale / terrain.horizontal_scale),
                                    int(5.5 * terrain_scale /terrain.horizontal_scale): int(6.0 * terrain_scale / terrain.horizontal_scale)] = 2.0 / terrain.vertical_scale
        
        for x in range(int(3.0 * terrain_scale / terrain.horizontal_scale), int(5.0 * terrain_scale / terrain.horizontal_scale)):
            terrain.height_field_raw[x, np.arange(int(2.0 * terrain_scale / terrain.horizontal_scale), int(2.5 * terrain_scale / terrain.horizontal_scale))] += \
                int(0.05 / terrain.vertical_scale * np.sin((x - int(3.0 * terrain_scale / terrain.horizontal_scale)) / int(0.2 / terrain.horizontal_scale)))

        return terrain

def CalculateFlatness(height_field_raw):
    flat_all=0.
    flat_list = []
    y,x = np.shape(height_field_raw)
    for yi in range(1,y-1):
        for xi in range(1,x-1):
            flat=0.
            for a in [-1,0,1]:
                for b in [-1,0,1]:
                    flat+= abs(height_field_raw[yi,xi] - height_field_raw[yi+a,xi+b])
            flat_all+= (flat/8.)
            flat_list.append(flat/8.)

    flat_all = flat_all/(x*y)
    return flat_all,flat_list     

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
  