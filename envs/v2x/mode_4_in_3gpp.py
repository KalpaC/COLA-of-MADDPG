# mode_4_in_3gpp 2023/10/19 17:18
import math

import numpy as np

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
            map_shape="rectangle",
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
            seed=7777,
    ):
        # map_args:
        self.n_actions = resource_blocks * len(V2V_power_levels)
        self.n_agents = veh_amount * neighbor_amount
        self.map_shape = map_shape
        self.grid_x_length = grid_x_length
        self.grid_y_length = grid_y_length
        self.lane_width = lane_width
        self.lane_amount = lane_amount
        self.grid_x_amount = grid_x_amount
        self.grid_y_amount = grid_y_amount
        # bs_args:
        self.bs_position_x = bs_position_x
        self.bs_position_y = bs_position_y
        self.bs_ant_height = bs_ant_height
        self.bs_ant_gain = bs_ant_gain
        self.bs_noise_figure = bs_noise_figure
        # veh_args:
        self.veh_amount = veh_amount
        self.veh_ant_height = veh_ant_height
        self.veh_ant_gain = veh_ant_gain
        self.veh_noise_figure = veh_noise_figure
        self.veh_velocity = veh_velocity
        # V2X_args
        self.carry_frequency = carry_frequency
        self.resource_blocks = resource_blocks
        self.bandwidth = bandwidth * int(1e6)
        self.V2V_power_levels = np.array(V2V_power_levels)
        self.sig2_dB = sig2_dB
        #  V2V_args
        self.V2V_decorrelation_distance = V2V_decorrelation_distance
        self.V2V_shadow_std = V2V_shadow_std
        self.neighbor_amount = neighbor_amount
        #  V2I_args
        self.V2I_decorrelation_distance = V2I_decorrelation_distance
        self.V2I_shadow_std = V2I_shadow_std
        self.payload_size = payload_size * 8
        self.time_budget = time_budget
        self.V2I_power = V2I_power
        self.time_fast = time_fast
        self.seed = seed
        np.random.seed(self.seed)

        # compute useful args
        self.map_x_length = self.grid_x_length * self.grid_x_amount
        self.map_y_length = self.grid_y_length * self.grid_y_amount
        self.time_slow = self.time_budget  # s

        # utils
        self.channel = None
        self._init_map()

    def reset_game(self):
        self._init_vehicles()
        self._reset_payload()
        self._reset_time()
        self._reactive_ants()
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
        def get_destination():
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
                self.vehicles[i].destinations = sort_idx[1:1 + self.veh_amount]

        get_destination()

    def get_action_space(self):
        return list(range(0, self.resource_blocks * len(self.V2V_power_levels)))

    def _get_agents_actions(self, action: np.ndarray):
        block_id = action % self.resource_blocks
        power = self.V2V_power_levels[action / self.resource_blocks]
        return block_id, power

    def episode_renew(self):
        self._renew_positions()
        self.channel.renew()

    def step(self, actions):
        block_id, power = self._get_agents_actions(actions)  # 返回agents实际选择的rb编号列表数组以及功率数组
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
        return reward, self.remain_time == 0, {}

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
        V2V_abs = self.channel.V2V_channels_abs[:, self.vehicles[veh_id].neighbors[nei_id], :]
        V2I_abs = self.channel.V2I_channels_abs[veh_id]
        obs = [demand, remain_time, V2V_interference_dB, V2I_fast, V2V_fast, V2V_abs, V2I_abs]
        return np.concatenate([np.asarray([x]).reshape(-1) for x in obs])

    def get_obs_size(self):
        pass

    def get_state(self):
        pass

    def get_state_size(self):
        pass

    def get_avail_actions(self):
        pass

    def get_avail_agent_actions(self, agent_id):
        pass

    def get_total_actions(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def save_replay(self):
        pass
