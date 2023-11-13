# mode_4_in_3gpp 2023/10/19 17:18
import logging
import math

import numpy as np
import torch

from utils import get_track_logger
from .. import MultiAgentEnv
from exceptions import IllegalArgumentException
from .util import V2V_Calculator
from .util import V2I_Calculator
from .util import Channel


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []


class Mode_4_in_3GPP(MultiAgentEnv):
    def __init__(
            self,
            grid_x_length=250,
            grid_y_length=433,
            lane_width=3.5,
            lane_amount=2,
            grid_x_amount=3,
            grid_y_amount=3,
            bs_position_x=375,
            bs_position_y=649.5,
            bs_ant_height=25,
            bs_ant_gain=8,
            bs_noise_figure=5,
            veh_amount=4,
            veh_ant_height=1.5,
            veh_ant_gain=3,
            veh_noise_figure=9,
            veh_velocity="10-15,m/s",
            carry_frequency=2,
            resource_blocks=4,
            bandwidth=1,
            V2V_power_levels=(23, 10, 5, -100),
            sig2_dB=-114,
            V2V_decorrelation_distance=10,
            V2V_shadow_std=3,
            neighbor_amount=1,
            V2I_decorrelation_distance=50,
            V2I_shadow_std=8,
            payload_size=1060,
            time_budget=0.1,
            V2I_power=23,
            time_fast=0.001,
            seed=7777
    ):
        # map_args:
        self.n_actions = resource_blocks * len(V2V_power_levels)  # 动作空间由(rb_id, power_level)组成
        self.n_agents = veh_amount * neighbor_amount  # 该环境假设每辆车与附近的neighbor_amount个车建立V2V通信，并各作为一个agent
        # 地图由重复的方格组成，默认在其中心位置有唯一基站。
        self.grid_x_length = grid_x_length  # 方格横向长度
        self.grid_y_length = grid_y_length  # 方格纵向长度
        self.lane_width = lane_width  # 车道宽度
        self.lane_amount = lane_amount  # 单向车道数量
        self.grid_x_amount = grid_x_amount  # 横向方格数量
        self.grid_y_amount = grid_y_amount  # 纵向方格数量
        # bs_args:
        self.bs_position_x = bs_position_x  # 基站位置横坐标
        self.bs_position_y = bs_position_y  # 基站位置纵坐标
        self.bs_ant_height = bs_ant_height  # 基站天线高度
        self.bs_ant_gain = bs_ant_gain  # 基站天线增益分贝
        self.bs_noise_figure = bs_noise_figure  # 天线处噪声功率
        # veh_args:
        self.veh_amount = veh_amount  # 车辆数量
        self.veh_ant_height = veh_ant_height  # 车辆天线高度
        self.veh_ant_gain = veh_ant_gain  # 车辆天线增益
        self.veh_noise_figure = veh_noise_figure  # 车辆天线噪声
        self.veh_velocity = veh_velocity  # 车辆速度，注意格式
        # V2X_args
        self.carry_frequency = carry_frequency  # 载波频率，单位为GHz
        self.resource_blocks = resource_blocks  # 资源块数量/子信道数量
        self.bandwidth = bandwidth * int(1e6)  # 信道带宽B
        self.V2V_power_levels = np.asarray(V2V_power_levels)  # 可选离散功率列表
        self.sig2_dB = sig2_dB  # 似乎是噪声方差
        #  V2V_args
        self.V2V_decorrelation_distance = V2V_decorrelation_distance  # 用于计算pathloss的值
        self.V2V_shadow_std = V2V_shadow_std  # 阴影衰落标准差
        self.neighbor_amount = neighbor_amount  # 每辆车通信邻居数量
        #  V2I_args
        self.V2I_decorrelation_distance = V2I_decorrelation_distance
        self.V2I_shadow_std = V2I_shadow_std
        self.payload_size = payload_size * 8  # 载荷bit数
        self.time_budget = time_budget  # 单一报文的时间预算
        self.V2I_power = V2I_power  # V2I信道固定功率
        self.time_fast = time_fast  # 快速衰落时间
        self.seed = seed  # 用于环境的种子
        np.random.seed(self.seed)

        self.episode_limit = int(time_budget // time_fast)  # 最多时间步数量

        # compute useful args
        self.map_x_length = self.grid_x_length * self.grid_x_amount
        self.map_y_length = self.grid_y_length * self.grid_y_amount
        self.time_slow = self.time_budget  # s

        # utils
        self.channel = None
        self._init_map()
        self.reset_game()

    def reset_game(self):
        self._init_vehicles()
        self._reset_payload()
        self._reset_time()
        self._reactive_ants()
        # 将信道信息的更新、管理功能抽象到Channel中
        self.channel = Channel(self.vehicles,
                               self.delta_distance,
                               self.bs_position_x,
                               self.bs_position_y,
                               self.bs_ant_height,
                               self.bs_ant_gain,
                               self.bs_noise_figure,
                               self.veh_amount,
                               self.veh_ant_height,
                               self.veh_ant_gain,
                               self.veh_noise_figure,
                               self.carry_frequency,
                               self.resource_blocks,
                               self.bandwidth,
                               self.sig2_dB,
                               self.V2V_decorrelation_distance,
                               self.V2V_shadow_std,
                               self.neighbor_amount,
                               self.V2I_decorrelation_distance,
                               self.V2I_shadow_std,
                               self.seed,
                               )
        # 进行一次计算，生成全部成员
        sample_action = np.ones((self.veh_amount, self.neighbor_amount), dtype=np.int)
        block_id, power = self._get_agents_actions(sample_action)
        self.channel.get_V2V_rate(self.is_active, block_id, power, self.V2I_power)
        self.channel.get_V2I_rate(self.is_active, block_id, power, self.V2I_power)

    def _init_map(self):
        self.lanes = {
            'u': [i * self.grid_x_length + self.lane_width / 2 + self.lane_width * j
                  for i in range(self.grid_x_amount)
                  for j in range(self.lane_amount)],
            'd': [(i + 1) * self.grid_x_length - self.lane_width / 2 - self.lane_width * j
                  for i in range(self.grid_x_amount)
                  for j in range(self.lane_amount - 1, -1, -1)],
            'l': [i * self.grid_y_length + self.lane_width / 2 + self.lane_width * j
                  for i in range(self.grid_y_amount)
                  for j in range(self.lane_amount)],
            'r': [(i + 1) * self.grid_y_length - self.lane_width / 2 - self.lane_width * j
                  for i in range(self.grid_y_amount)
                  for j in range(self.lane_amount - 1, -1, -1)]
        }
        self.up_lanes = self.lanes['u']
        self.down_lanes = self.lanes['d']
        self.left_lanes = self.lanes['l']
        self.right_lanes = self.lanes['r']

        self.directions = list(self.lanes.keys())

    def _init_vehicles(self):
        self.vehicles = []

        def add_vehicle():
            direction = np.random.choice(self.directions)
            road = np.random.randint(0, len(self.lanes[direction]))
            if direction == 'u' or direction == 'd':
                x = self.lanes[direction][road]
                y = np.random.rand() * self.map_y_length
            else:
                x = np.random.rand() * self.map_x_length
                y = self.lanes[direction][road]
            position = [x, y]
            self.vehicles.append(Vehicle(position, direction, get_velocity()))

        def get_velocity():
            opts = self.veh_velocity.split(',')
            try:
                x = opts[0].split('-')
                if len(x) == 1:
                    v = float(x[0])
                    if opts[1].lower() == 'km/s':
                        v /= 3.6
                    return v
                else:
                    low, high = (float(x[0]), float(x[1]))
                    if opts[1].lower() == 'km/s':
                        low /= 3.6
                        high /= 3.6
                    return low + np.random.rand() * (high - low)

            except Exception:
                raise IllegalArgumentException('veh_velocity', self.veh_velocity,
                                               "Expected {>0},{km/s or m/s}")

        for i in range(self.veh_amount):
            add_vehicle()
        self.delta_distance = np.array([c.velocity * self.time_slow for c in self.vehicles])
        self.get_destination()

    def _reset_payload(self):
        self.remain_payload = np.full((self.veh_amount, self.neighbor_amount), self.payload_size)

    def _reset_time(self):
        self.remain_time = self.time_budget

    def _reactive_ants(self):
        self.is_active = np.ones((self.veh_amount, self.neighbor_amount), dtype='bool')

    def _renew_positions(self):
        i = 0
        while i < len(self.vehicles):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                         self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if not change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and (
                                (self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                             self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if not change_direction:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and (
                            (self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (
                                    delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if not change_direction:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (
                                self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (
                                        delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                             self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if not change_direction:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if not change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if not change_direction:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and (
                            (self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if np.random.uniform(0, 1) < 0.4:
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (
                                    delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if not change_direction:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and (
                                (self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if np.random.uniform(0, 1) < 0.4:
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (
                                        delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if not change_direction:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (
                    self.vehicles[i].position[0] > self.map_x_length) or (
                    self.vehicles[i].position[1] > self.map_y_length):
                # delete
                #    print ('delete ', self.position[i])
                if self.vehicles[i].direction == 'u':
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if self.vehicles[i].direction == 'd':
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if self.vehicles[i].direction == 'l':
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if self.vehicles[i].direction == 'r':
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

        # 更新邻居
        self.get_destination()

    def get_destination(self):
        # 找到对每辆车找到距离它最近的self.n_des辆车
        # 每次更新位置之后都需要重新判断，因为数据包的有效期恰好也过了
        positions = np.array([c.position for c in self.vehicles])
        distance = np.zeros((self.veh_amount, self.veh_amount))
        for i in range(self.veh_amount):
            for j in range(self.veh_amount):
                # np.linalg.norm用于计算向量的模，此处可以用于计算两点间距离
                distance[i][j] = np.linalg.norm(positions[i] - positions[j])
        for i in range(self.veh_amount):
            sort_idx = np.argsort(distance[:, i])
            self.vehicles[i].neighbors = sort_idx[1:1 + self.veh_amount]

    def _get_agents_actions(self, action: np.ndarray):
        assert self.veh_amount * self.neighbor_amount == self.n_agents
        action = action.reshape((self.veh_amount, self.neighbor_amount))
        block_id = action % self.resource_blocks
        power = self.V2V_power_levels[action // self.resource_blocks]
        return block_id, power

    def _renew_for_episode(self):
        self._reset_payload()
        self._reset_time()
        self._renew_positions()
        self.channel.renew()

    def step(self, actions: np.ndarray):
        block_id, power = self._get_agents_actions(actions)  # 返回agents实际选择的rb编号列表数组以及功率数组
        # get_track_logger().info(block_id)
        V2V_rate = self.channel.get_V2V_rate(self.is_active, block_id, power, self.V2I_power)
        V2I_rate = self.channel.get_V2I_rate(self.is_active, block_id, power, self.V2I_power)
        self.remain_payload -= (V2V_rate * self.time_slow * self.bandwidth).astype('int32')
        self.remain_payload[self.remain_payload < 0] = 0
        self.remain_time -= self.time_fast

        reward_elements = V2V_rate / 10
        reward_elements[self.remain_payload <= 0] = 1
        self.is_active[np.multiply(self.is_active, self.remain_payload <= 0)] = 0
        l = 0.1
        reward = l * np.sum(V2I_rate) / (self.veh_amount * 10) + (1 - l) * np.sum(reward_elements) / (
                self.veh_amount * self.neighbor_amount)
        # 添加信道更新以及干扰计算
        self.channel.renew_fastfading()
        self.channel.get_V2V_rate(self.is_active, block_id, power, self.V2I_power)
        return reward, self.remain_time <= 0, {}

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        veh_id, nei_id = agent_id // self.neighbor_amount, agent_id % self.neighbor_amount
        demand = self.remain_payload[veh_id][nei_id] / self.payload_size
        remain_time = self.remain_time / self.time_slow
        V2V_interference_dB = self.channel.V2V_Interference_cache[veh_id][nei_id]
        V2I_fast = self.channel.V2I_channels_with_fastfading[veh_id, :] - self.channel.V2I_channels_abs[veh_id]
        V2V_fast = self.channel.V2V_channels_with_fastfading[:, self.vehicles[veh_id].neighbors[nei_id], :] \
                   - self.channel.V2V_channels_abs[:, self.vehicles[veh_id].neighbors[nei_id]]
        V2V_abs = self.channel.V2V_channels_abs[:, self.vehicles[veh_id].neighbors[nei_id]]
        V2I_abs = self.channel.V2I_channels_abs[veh_id]
        obs = [demand, remain_time, V2V_interference_dB, V2I_fast, V2V_fast, V2V_abs, V2I_abs]
        return np.concatenate([np.asarray([x]).reshape(-1) for x in obs])

    def get_obs_size(self):
        return self.get_obs_agent(0).shape[0]

    def get_state(self):
        return None

    def get_state_size(self):
        return 0

    def get_avail_actions(self):
        return [list(range(0, self.n_actions)) for i in range(self.n_agents)]
        # return np.repeat(np.array()[np.newaxis, :], self.n_agents, axis=0)

    def get_avail_agent_actions(self, agent_id):
        return np.array(list(range(0, self.n_actions)))

    def get_total_actions(self):
        return self.n_actions

    def reset(self):
        self._renew_for_episode()

    def render(self):
        pass

    def close(self):
        # 非单独线程，无需close
        pass

    def seed(self):
        return self.seed

    def save_replay(self):
        pass
