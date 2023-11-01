# buffer 2023/10/31 14:21

# 这里是经验回放区，需要适合任意的算法，如何实现？
# 1. 回放区保存的是什么内容？有什么功能（方法）
# 2. 不同算法/不同环境的经验回放区有什么区别，能提取出什么共性？

# 1. 回放区保存的是transition，记录了环境每一次行动前后的状态、actor的行动、行动的奖励
#    需要支持的方法：记录数据、取出数据
# 2. 不同环境和算法的状态和动作是不同的，可以认为是不同维度的张量。
#    而奖励一定是一个浮点数，而非张量。
#    对于多agent问题来说，我们将所有agent的经验存到一起，所以agent的个数也会影响回放区的大小
#    由于我们使用episode run的方式，即一个episode之后保存一次经验、更新一次网络，我们还需要一次性存储一个episode的数据
#    故episode的时间步也有必要
#    最后是总的episode个数或者能能记录的episode上限。实际上如果所有经验都很重要，并且数据量过大我们可以考虑将数据转移至外存。
#    总之，需要的参数包括：state的维度、action的维度、t_max、episode_max
import pprint
from types import SimpleNamespace as SN

import numpy as np
import torch as th

example_scheme = {
    "state": {"shape": 10},  # 由于多agent的观测会默认被结合到一起，所以此处无需额外输入group信息
    "actions": {"shape": (1,), "group": "agents", "dtype": th.long},
    "reward": {"shape": (1,)}
}

example_groups = {
    "agents": 4
}


class EpisodeBatch:
    def __init__(self,
                 scheme: dict,
                 groups: dict,
                 episode_amount: int,
                 max_seq_length: int,
                 data: SN = None,
                 preprocess: dict = None,
                 device="cpu"):
        """
        存储transition数据。
        :param scheme: indicate what kinds of records need to be stored by dict and the attributes of record
        :param groups:
        :param episode_amount: how many episodes
        :param max_seq_length: how many transition in one episode
        :param data: prepared formatted data like data in this class
        :param preprocess: ?
        :param device: cuda option, "cpu" or "cuda"
        """
        self.data = self._new_data()
        self.scheme = scheme.copy()
        self.groups = groups
        self.episode_amount = episode_amount
        self.max_seq_length = max_seq_length
        self.device = device

        if data:
            self.data = data
        else:
            self._init_data(self.scheme, groups, episode_amount, max_seq_length)

    def update(self, other_eb, to_slice: slice):
        pass

    def _new_data(self):
        data = SN(episodes={})
        return data

    def _init_data(self, scheme, groups, batch_size, max_seq_length):
        for label, info in scheme.items():
            assert 'shape' in info, "Scheme must define shape for {}".format(label)
            shape = info['shape']
            group_name = info.get('group', None)
            dtype = info.get('dtype', th.float32)
            if isinstance(shape, int):
                shape = (shape,)
            if group_name:
                assert group_name in groups, \
                    "Group {} must have its number of members defined in _groups_".format(group_name)
                group_dim = groups[group_name]
                shape = (group_dim, *shape)
            self.data.episodes[label] = th.empty((batch_size, max_seq_length, *shape), dtype=dtype)

    def __getitem__(self, item):
        if isinstance(item, str):
            # key为字符串，直接从data中寻找该字段
            for sub_data in self.data.__dict__.values():
                if item in sub_data:
                    return sub_data[item]
            raise KeyError
        elif isinstance(item, int):
            new_sn = self._new_data()
            new_sn_dic = new_sn.__dict__
            for label, sub_data in self.data.__dict__.items():
                assert label in new_sn_dic, "unmatched _new_data function, {} not found in new data".format(label)
                new_sub_data = new_sn_dic[label]
                for field, tensor in sub_data.items():
                    new_sub_data[field] = tensor[item:item + 1]
            return EpisodeBatch(self.scheme, self.groups, 1, self.max_seq_length, data=new_sn, device=self.device)
        else:
            bs = 0
            new_sn = self._new_data()
            new_sn_dic = new_sn.__dict__
            for label, sub_data in self.data.__dict__.items():
                assert label in new_sn_dic, "unmatched _new_data function, {} not found in new data".format(label)
                new_sub_data = new_sn_dic[label]
                for field, tensor in sub_data.items():
                    new_sub_data[field] = tensor[item]
                    if bs == 0:
                        bs = tensor[item].shape[0]
            return EpisodeBatch(self.scheme, self.groups, bs, self.max_seq_length, data=new_sn, device=self.device)

    def __repr__(self):
        return "EpisodeBatch(\n" \
               "\tepisode_amount={}".format(self.episode_amount) + "\n" \
                                                                   "\tmax_seq_length={}".format(
            self.max_seq_length) + "\n" \
                                   "\tdevice={}".format(self.device) + "\n" \
                                                                       "\tscheme={}".format(self.scheme) + "\n" \
                                                                                                           "\tgroups={}".format(
            self.groups) + "\n)"
        # "\tdata={}".format(pprint.pformat(self.data)) + "\n)"


class ReplayBuffer:
    def __init__(self, scheme: dict, groups: dict, buffer_size: int, max_seq_length: int, data: SN = None,
                 preprocess: dict = None, device="cpu"):
        self.buffer = EpisodeBatch(scheme, groups, buffer_size, max_seq_length, data, preprocess, device)
        self.scheme = scheme
        self.groups = groups
        self.max_seq_length = max_seq_length
        self.device = device
        self.preprocess = preprocess
        self.buffer_size = buffer_size
        self.index = 0  # 循环列表的尾指针
        self.total = 0  # 总计有效episode数量

    def get_empty_episode_batch(self, episode_amount=1):
        # 生成一个指定episode_amount的EpisodeBatch
        return EpisodeBatch(self.scheme, self.groups, episode_amount, self.max_seq_length, None, self.preprocess,
                            self.device)

    def insert(self, eb: EpisodeBatch):
        if self.index + eb.episode_amount <= self.buffer_size:
            self.buffer.update(eb, slice(self.index, self.index + eb.episode_amount))
            self.index += eb.episode_amount
            self.total = max(self.total, self.index)
            self.index %= self.buffer_size
            assert self.index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.index
            self.insert(eb[0:buffer_left])
            self.insert(eb[buffer_left:])

    def can_sample(self, batch_size):
        return self.total >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.total == batch_size:
            return self.buffer[:batch_size]
        else:
            ep_ids = np.random.choice(self.total, batch_size, replace=False)
            return self.buffer[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.total,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())
#
#
# e = EpisodeBatch(example_scheme, example_groups, 5, 10)
# print(e[2:])
#
# rb = ReplayBuffer(example_scheme, example_groups, 8, 10)
# rb.insert(e)
# rb.insert(e)
# print(rb)
# sample = rb.sample(4)
# print(sample)

