# channel_buffer 2023/10/22 22:29
import logging
import math

import numpy as np
from .v2v_calculator import V2V_Calculator
from .v2i_calculator import V2I_Calculator
from .formula import to_w
from utils import get_track_logger


class Channel:
    def __init__(self,
                 vehicles,
                 delta_distance,
                 bs_position_x=375,
                 bs_position_y=649.5,
                 bs_ant_height=25,
                 bs_ant_gain=8,
                 bs_noise_figure=5,
                 veh_amount=4,
                 veh_ant_height=1.5,
                 veh_ant_gain=3,
                 veh_noise_figure=9,
                 carry_frequency=2,
                 resource_blocks=4,
                 bandwidth=1,
                 sig2_dB=-114,
                 V2V_decorrelation_distance=10,
                 V2V_shadow_std=3,
                 neighbor_amount=1,
                 V2I_decorrelation_distance=50,
                 V2I_shadow_std=8,
                 seed=7777,
                 ):
        self.n_veh = veh_amount
        self.n_RB = resource_blocks
        self.n_des = neighbor_amount
        self.bs_ant_gain = bs_ant_gain
        self.bs_noise_figure = bs_noise_figure
        self.veh_ant_gain = veh_ant_gain
        self.veh_noise_figure = veh_noise_figure
        self.bandwidth = bandwidth
        self.sig2_dB = sig2_dB
        np.random.seed(seed)

        self.V2V_shadow_std = V2V_shadow_std
        self.V2I_shadow_std = V2I_shadow_std
        self.V2V_pathloss = np.empty((veh_amount, veh_amount)) + 50 * np.identity(veh_amount)
        self.V2I_pathloss = np.empty(veh_amount)

        self.V2V_Shadowing = np.random.normal(0, self.V2V_shadow_std, (self.n_veh, self.n_veh))
        self.V2I_Shadowing = np.random.normal(0, self.V2I_shadow_std, self.n_veh)

        self.V2V_channels_abs = np.empty((veh_amount, veh_amount))
        self.V2I_channels_abs = np.empty(veh_amount)

        self.V2V_channels_with_fastfading = np.empty((veh_amount, veh_amount, resource_blocks))
        self.V2I_channels_with_fastfading = np.empty((veh_amount, resource_blocks))

        self.v2vc = V2V_Calculator(carry_frequency, veh_ant_height, V2V_decorrelation_distance, V2V_shadow_std)
        self.v2ic = V2I_Calculator(veh_ant_height, bs_ant_height, V2I_decorrelation_distance, V2I_shadow_std,
                                   [bs_position_x, bs_position_y])

        self.V2V_rate_cache = None
        self.V2I_rate_cache = None
        self.V2V_Interference_cache = None

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
        # 数据过期，刷新cache
        self._clear_cache()

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
        # 数据过期，刷新cache
        self._clear_cache()

    def _clear_cache(self):
        self.V2V_rate_cache = None
        self.V2I_rate_cache = None

    def get_V2V_rate(self, is_active=None, block_id=None, V2V_power_dB=None, V2I_power_dB=None):
        if self.V2V_rate_cache is not None:
            return self.V2V_rate_cache
        V2V_Interference = np.zeros((len(self.vehicles), self.n_des))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_des))
        block_id[(np.logical_not(is_active))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(block_id == i)
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].neighbors[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = to_w(
                    V2V_power_dB[indexes[j, 0], indexes[j, 1]]
                    - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i]
                    + 2 * self.veh_ant_gain - self.veh_noise_figure
                )
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += to_w(
                    V2I_power_dB
                    - self.V2V_channels_with_fastfading[i, receiver_j, i]
                    + 2 * self.veh_ant_gain - self.veh_noise_figure
                )
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].neighbors[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += to_w(
                        V2V_power_dB[indexes[k, 0], indexes[k, 1]]
                        - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]
                        + 2 * self.veh_ant_gain - self.veh_noise_figure
                    )
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += to_w(
                        V2V_power_dB[indexes[j, 0], indexes[j, 1]]
                        - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]
                        + 2 * self.veh_ant_gain - self.veh_noise_figure
                    )
        V2V_Interference = V2V_Interference + to_w(self.sig2_dB)
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, V2V_Interference))
        # 缓存
        self.V2V_Interference_cache = V2V_Interference.copy()
        self.V2V_rate_cache = V2V_Rate.copy()
        return V2V_Rate

    def get_V2I_rate(self, is_active=None, block_id=None, V2V_power_dB=None, V2I_power_dB=None):
        if self.V2I_rate_cache is not None:
            return self.V2I_rate_cache
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_des):
                if not is_active[i, j]:
                    continue
                V2I_Interference[block_id[i][j]] += to_w(V2V_power_dB[i, j]
                                                         - self.V2I_channels_with_fastfading[i, block_id[i, j]]
                                                         + self.veh_ant_gain + self.bs_ant_gain - self.bs_noise_figure)
        V2I_Interference = V2I_Interference + to_w(self.sig2_dB)
        V2I_Signals = to_w(V2I_power_dB
                           - self.V2I_channels_with_fastfading.diagonal()
                           + self.veh_ant_gain
                           + self.bs_ant_gain
                           - self.bs_noise_figure)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, V2I_Interference))
        self.V2I_rate_cache = V2I_Rate.copy()
        return V2I_Rate
