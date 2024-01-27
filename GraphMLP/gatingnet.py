import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, num_nodes, horizon):
        super(GatingNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        # 定义门控网络
        self.gate = nn.Sequential(
            nn.Linear(num_nodes * horizon * 2, num_nodes * horizon),
            nn.ReLU(),
            nn.Linear(num_nodes * horizon, num_nodes * horizon),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 将输入合并并展平
        batch_size = x1.size(0)
        combined = torch.cat([x1, x2], dim=-1).view(batch_size, -1)

        # 通过门控网络
        gate_weights = self.gate(combined).view(batch_size, self.num_nodes, self.horizon)

        # 使用门控权重对输入进行加权组合
        return gate_weights * x1 + (1 - gate_weights) * x2

