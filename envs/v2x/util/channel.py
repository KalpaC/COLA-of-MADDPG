# channel_buffer 2023/10/22 22:29
import math

import numpy as np
from .v2v_calculator import V2V_Calculator
from .v2i_calculator import V2I_Calculator


class Channel:
    def __init__(self,
                 vehicles,
                 delta_distance,
                 veh_amount,
                 neighbor_amount,
                 resource_blocks,
                 carry_frequency,
                 veh_ant_height,
                 V2V_decorrelation_distance,
                 V2V_shadow_std,
                 bs_ant_height,
                 V2I_decorrelation_distance,
                 V2I_shadow_std,
                 bs_position_x,
                 bs_position_y
                 ):
        self.n_veh = veh_amount
        self.n_RB = resource_blocks
        self.n_des = neighbor_amount

        self.V2V_shadow_std = V2V_shadow_std
        self.V2I_shadow_std = V2I_shadow_std
        self.V2V_pathloss = np.empty((veh_amount, veh_amount)) + 50 * np.identity(veh_amount)
        self.V2I_pathloss = np.empty(veh_amount)

        self.V2V_Shadowing = np.random.normal(0, self.V2V_shadow_std, (self.n_veh, self.n_veh))
        self.V2I_Shadowing = np.random.normal(0, self.V2I_shadow_std, self.n_veh)

        self.V2V_channels_abs = np.empty((veh_amount, veh_amount))
        self.V2I_channels_abs = np.empty(veh_amount)

        self.V2V_channels_with_fastfading = np.empty((veh_amount, veh_amount, resource_blocks))
        self.V2I_channels_with_fastfading = np.empty(veh_amount, resource_blocks)


        self.v2vc = V2V_Calculator(carry_frequency, veh_ant_height, V2V_decorrelation_distance, V2V_shadow_std)
        self.v2ic = V2I_Calculator(veh_ant_height, bs_ant_height, V2I_decorrelation_distance, V2I_shadow_std,
                                   [bs_position_x, bs_position_y])

        self.vehicles = vehicles
        self.delta_distance = delta_distance


    def renew(self):
        self.V2V_pathloss = np.zeros((self.n_veh, self.n_veh)) + 50 * np.identity(self.n_veh)
        self.V2I_pathloss = np.zeros(self.n_veh)
        # self.V2V_channels_abs = np.zeros((self.n_veh, self.n_veh))
        # self.V2I_channels_abs = np.zeros(self.n_veh)
        for i in range(self.n_veh):
            for j in range(i + 1, self.n_veh):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.v2vc.get_shadowing(
                    self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = self.v2vc.get_path_loss(
                    self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing
        self.V2I_Shadowing = self.v2ic.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.v2ic.get_path_loss(self.vehicles[i].position)
        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_fastfading(self):
        # 计算并更新得到新的包含fastfading的信道信息
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape)
                   + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape)
                   + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))
