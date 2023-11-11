# mlp_agent 2023/11/2 13:15

import torch.nn as nn
import torch.nn.functional as F
import torch


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args
        self.fcs = [nn.Linear(input_shape, args.mlp_hidden_dims[0])]
        self.fcs[-1].weight.data.normal_(0, 0.1)
        for i in range(len(args.mlp_hidden_dims) - 1):
            self.fcs.append(nn.Linear(args.mlp_hidden_dims[i], args.mlp_hidden_dims[i + 1]))
            self.fcs[-1].weight.data.normal_(0, 0.1)
        self.fcs.append(nn.Linear(args.mlp_hidden_dims[-1], args.n_actions))
        self.fcs[-1].weight.data.normal_(0, 0.1)

    def cal_value(self, inputs):
        inputs = self.fcs[0](inputs)
        for i in range(1, len(self.fcs)):
            inputs = F.relu(inputs)
            inputs = self.fcs[i](inputs)
        return inputs
