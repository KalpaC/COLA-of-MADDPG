# fingerprint_controller 2023/11/10 23:17

from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class FingerPrintMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # 注意ep_batch的第一维是batch_size，即并行在环境中执行动作的线程数。如果是单线程runner，则第一维长度为1
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
                                                            test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t_ep, t_env, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t_ep, t_env, test_mode)
        agent_outs = self.agent.calc_value(agent_inputs)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def parameters(self):
        return list(self.agent.parameters())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t_ep, t_env, test_mode):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t_ep]]
        # t时间下的obs自然包括n_agents的全部局部观察，注意batch["obs"][:, t]得到的
        if self.args.obs_last_action:
            # 意思是是否将上一次的行动作为环境传入
            if t_ep <= 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t_ep]))
            else:
                inputs.append(batch["actions_onehot"][:, t_ep - 1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        if self.args.finger_print:
            # 添加时间指纹和探索指纹
            fp = th.empty((bs, self.n_agents, 2), device=batch.device)
            fp[:, :, 0] = t_env if not test_mode else self.args.t_max
            fp[:, :, 1] = self.action_selector.epsilon if not test_mode else 1
            inputs.append(fp)
        inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
