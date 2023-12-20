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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
# from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
# from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .a1.a1_amp_roll_config import A1AMPCfg_ro, A1AMPCfgPPO_ro

from .a1.a1_amp_pitch_backward_config import A1AMPCfg_pb, A1AMPCfgPPO_pb
from .a1.a1_amp_pitch_forward_config import A1AMPCfg_pf, A1AMPCfgPPO_pf
from .a1.a1_amp_roll_left_config import A1AMPCfg_rl, A1AMPCfgPPO_rl
from .a1.a1_amp_roll_right_config import A1AMPCfg_rr, A1AMPCfgPPO_rr

from .a1.a1_amp_up_config import A1AMPCfg_up, A1AMPCfgPPO_up
from .a1.a1_amp_up1_config import A1AMPCfg_up1, A1AMPCfgPPO_up1
from .a1.a1_amp_up_forward_config import A1AMPCfg_up_f, A1AMPCfgPPO_up_f
from .a1.a1_amp_up_backward_config import A1AMPCfg_up_b, A1AMPCfgPPO_up_b
from .a1.a1_amp_up_left_config import A1AMPCfg_up_l, A1AMPCfgPPO_up_l
from .a1.a1_amp_up_right_config import A1AMPCfg_up_r, A1AMPCfgPPO_up_r

from .a1.a1_amp_forward_mass_config import A1AMPCfg_fc, A1AMPCfgPPO_fc
from .a1.a1_amp_forward_walking_config import A1AMPCfg_fw, A1AMPCfgPPO_fw

from .a1.a1_amp_forward_walking_fast_config import A1AMPCfg_fw_f, A1AMPCfgPPO_fw_f

from .a1.a1_amp_forward_walking_lstm_config import A1AMPCfg_fw_lstm, A1AMPCfgPPO_fw_lstm

from .a1.a1_amp_standup_config import A1AMPCfg_sd, A1AMPCfgPPO_sd
from .a1.a1_amp_backward_left_walking_config import A1AMPCfg_blw, A1AMPCfgPPO_blw
from .a1.a1_amp_backward_right_walking_config import A1AMPCfg_brw, A1AMPCfgPPO_brw
from .a1.a1_amp_backward_walking_config import A1AMPCfg_bw, A1AMPCfgPPO_bw
from .a1.a1_amp_forward_noise_config import A1AMPCfg_bow, A1AMPCfgPPO_bow
from .a1.a1_amp_forward_left_config import A1AMPCfg_forward_left, A1AMPCfgPPO_forward_left
from .a1.a1_amp_forward_right_config import A1AMPCfg_forward_right, A1AMPCfgPPO_forward_right
from .a1.a1_amp_gallop_config import A1AMPCfg_gallop, A1AMPCfgPPO_gallop
from .a1.a1_amp_gallop_fast_config import A1AMPCfg_gallop_fast, A1AMPCfgPPO_gallop_fast
from .a1.a1_amp_gallop_faster_config import A1AMPCfg_gallop_faster, A1AMPCfgPPO_gallop_faster

from .a1.a1_amp_sidestep_left_config import A1AMPCfg_sidestep_left, A1AMPCfgPPO_sidestep_left
from .a1.a1_amp_sidestep_right_config import A1AMPCfg_sidestep_right, A1AMPCfgPPO_sidestep_right
from .a1.a1_amp_spin_clockwise_config import A1AMPCfg_spin_clockwise, A1AMPCfgPPO_spin_clockwise
from .a1.a1_amp_spin_clockwise_oe_config import A1AMPCfg_spin_clockwise_oe, A1AMPCfgPPO_spin_clockwise_oe

from .a1.a1_amp_spin_counterclockwise_config import A1AMPCfg_sp_counterclockwise, A1AMPCfgPPO_sp_counterclockwise
from .a1.a1_amp_kick_fl_config import A1AMPCfg_kick_fl, A1AMPCfgPPO_kick_fl
from .a1.a1_amp_kick_fr_config import A1AMPCfg_kick_fr, A1AMPCfgPPO_kick_fr
from .a1.a1_amp_kick_rl_config import A1AMPCfg_kick_rl, A1AMPCfgPPO_kick_rl
from .a1.a1_amp_kick_rr_config import A1AMPCfg_kick_rr, A1AMPCfgPPO_kick_rr


# from .a1.a1_amp_complex_terrain_baseline_command_config import A1AMPCfg_ct, A1AMPCfgPPO_ct
from .a1.a1_amp_soccer_config import A1AMPCfg_soccer, A1AMPCfgPPO_soccer

from .a1.a1_amp_complex_terrain_sg_config import A1AMPCfg_sg, A1AMPCfgPPO_sg
from .a1.a1_amp_complex_terrain_baseline_config import A1AMPCfg_base, A1AMPCfgPPO_base
from .a1.a1_amp_complex_terrain_sequential_baseline_config import A1AMPCfg_base_sequ, A1AMPCfgPPO_base_sequ

from .a1.a1_amp_complex_terrain_baseline_command_config import A1AMPCfg_baseline_com, A1AMPCfgPPO_baseline_com
from .a1.a1_amp_complex_terrain_baseline_pure_RL_config import A1AMPCfg_baseline_pure_RL, A1AMPCfgPPO_baseline_pure_RL


import os

from legged_gym.utils.task_registry import task_registry

# task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "a1_amp_roll", LeggedRobot, A1AMPCfg_ro(), A1AMPCfgPPO_ro() )

task_registry.register( "a1_amp_pitch_forward", LeggedRobot, A1AMPCfg_pf(), A1AMPCfgPPO_pf() )
task_registry.register( "a1_amp_pitch_backward", LeggedRobot, A1AMPCfg_pb(), A1AMPCfgPPO_pb() )
task_registry.register( "a1_amp_roll_left", LeggedRobot, A1AMPCfg_rl(), A1AMPCfgPPO_rl() )
task_registry.register( "a1_amp_roll_right", LeggedRobot, A1AMPCfg_rr(), A1AMPCfgPPO_rr() )

task_registry.register( "a1_amp_up", LeggedRobot, A1AMPCfg_up(), A1AMPCfgPPO_up() )
task_registry.register( "a1_amp_up1", LeggedRobot, A1AMPCfg_up1(), A1AMPCfgPPO_up1() )
task_registry.register( "a1_amp_up_forward", LeggedRobot, A1AMPCfg_up_f(), A1AMPCfgPPO_up_f() )
task_registry.register( "a1_amp_up_backward", LeggedRobot, A1AMPCfg_up_b(), A1AMPCfgPPO_up_b() )
task_registry.register( "a1_amp_up_left", LeggedRobot, A1AMPCfg_up_l(), A1AMPCfgPPO_up_l() )
task_registry.register( "a1_amp_up_right", LeggedRobot, A1AMPCfg_up_r(), A1AMPCfgPPO_up_r() )

task_registry.register( "a1_amp_forward_mass", LeggedRobot, A1AMPCfg_fc(), A1AMPCfgPPO_fc() )
task_registry.register( "a1_amp_forward_walking", LeggedRobot, A1AMPCfg_fw(), A1AMPCfgPPO_fw() )

task_registry.register( "a1_amp_forward_walking_fast", LeggedRobot, A1AMPCfg_fw_f(), A1AMPCfgPPO_fw_f() )

task_registry.register( "a1_amp_forward_walking_lstm", LeggedRobot, A1AMPCfg_fw_lstm(), A1AMPCfgPPO_fw_lstm() )

task_registry.register( "a1_amp_standup", LeggedRobot, A1AMPCfg_sd(), A1AMPCfgPPO_sd() )

task_registry.register( "a1_amp_backward_left", LeggedRobot, A1AMPCfg_blw(), A1AMPCfgPPO_blw())
task_registry.register( "a1_amp_backward_right", LeggedRobot, A1AMPCfg_brw(), A1AMPCfgPPO_brw())
task_registry.register( "a1_amp_backward", LeggedRobot, A1AMPCfg_bw(), A1AMPCfgPPO_bw() )

task_registry.register( "a1_amp_forward_noise", LeggedRobot, A1AMPCfg_bow(), A1AMPCfgPPO_bow() )
task_registry.register( "a1_amp_forward_left", LeggedRobot, A1AMPCfg_forward_left(), A1AMPCfgPPO_forward_left() )
task_registry.register( "a1_amp_forward_right", LeggedRobot, A1AMPCfg_forward_right(), A1AMPCfgPPO_forward_right() )

task_registry.register( "a1_amp_gallop", LeggedRobot, A1AMPCfg_gallop(), A1AMPCfgPPO_gallop() )
task_registry.register( "a1_amp_gallop_fast", LeggedRobot, A1AMPCfg_gallop_fast(), A1AMPCfgPPO_gallop_fast() )
task_registry.register( "a1_amp_gallop_faster", LeggedRobot, A1AMPCfg_gallop_faster(), A1AMPCfgPPO_gallop_faster() )


task_registry.register( "a1_amp_sidestep_left", LeggedRobot, A1AMPCfg_sidestep_left(), A1AMPCfgPPO_sidestep_left() )
task_registry.register( "a1_amp_sidestep_right", LeggedRobot, A1AMPCfg_sidestep_right(), A1AMPCfgPPO_sidestep_right() )
task_registry.register( "a1_amp_spin_clockwise", LeggedRobot, A1AMPCfg_spin_clockwise(), A1AMPCfgPPO_spin_clockwise() )
task_registry.register( "a1_amp_spin_clockwise_oe", LeggedRobot, A1AMPCfg_spin_clockwise_oe(), A1AMPCfgPPO_spin_clockwise_oe() )

task_registry.register( "a1_amp_spin_counterclockwise", LeggedRobot, A1AMPCfg_sp_counterclockwise(), A1AMPCfgPPO_sp_counterclockwise() )

task_registry.register( "a1_amp_kick_fl", LeggedRobot, A1AMPCfg_kick_fl(), A1AMPCfgPPO_kick_fl() )
task_registry.register( "a1_amp_kick_fr", LeggedRobot, A1AMPCfg_kick_fr(), A1AMPCfgPPO_kick_fr() )
task_registry.register( "a1_amp_kick_rl", LeggedRobot, A1AMPCfg_kick_rl(), A1AMPCfgPPO_kick_rl() )
task_registry.register( "a1_amp_kick_rr", LeggedRobot, A1AMPCfg_kick_rr(), A1AMPCfgPPO_kick_rr() )


# task_registry.register( "a1_amp_ct", LeggedRobot, A1AMPCfg_ct(), A1AMPCfgPPO_ct() )
task_registry.register( "a1_amp_soccer", LeggedRobot, A1AMPCfg_soccer(), A1AMPCfgPPO_soccer() )

task_registry.register( "a1_amp_ct_sg", LeggedRobot, A1AMPCfg_sg(), A1AMPCfgPPO_sg() )
task_registry.register( "a1_amp_ct_b", LeggedRobot, A1AMPCfg_base(), A1AMPCfgPPO_base() )
task_registry.register( "a1_amp_ct_b_sequential", LeggedRobot, A1AMPCfg_base_sequ(), A1AMPCfgPPO_base_sequ() )

task_registry.register( "a1_amp_ct_b_com", LeggedRobot, A1AMPCfg_baseline_com(), A1AMPCfgPPO_baseline_com() )

task_registry.register( "a1_amp_ct_b_pure_rl", LeggedRobot, A1AMPCfg_baseline_pure_RL(), A1AMPCfgPPO_baseline_pure_RL() )