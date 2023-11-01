# episode_runner 2023/10/31 10:54
from functools import partial

from components.buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self):
        # 一个episode
        assert self.new_batch
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)


        pass
