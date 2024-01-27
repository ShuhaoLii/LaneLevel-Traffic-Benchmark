import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer (nn.Module):
    def __init__(self, node_features, horizon):
        super (AttentionLayer, self).__init__ ()
        self.node_features = node_features
        self.horizon = horizon
        self.attention = nn.Sequential (
            nn.Linear (node_features * horizon, 128),
            nn.ReLU (),
            nn.Linear (128, 1)
        )

    def forward(self, x):
        batch_size, horizon, num_nodes = x.size ()
        x = x.reshape (batch_size, num_nodes, -1)  # reshape to (batch_size, num_nodes, node_features * horizon)

        # Compute attention scores
        scores = self.attention (x)  # (batch_size, num_nodes, 1)
        adjacency = torch.softmax (scores @ scores.transpose (1, 2), dim=-1)  # (batch_size, num_nodes, num_nodes)

        return adjacency


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.weight.size(1)**0.5
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        batch_size, num_nodes = adj.size(0), adj.size(1)

        output = torch.zeros(batch_size, num_nodes, self.out_features).to(support.device)

        # 对每个样本单独进行图卷积
        for i in range(batch_size):
            output[i] = torch.matmul(adj[i], support[i*num_nodes:(i+1)*num_nodes])

        return output


class DynamicGraphConvNet (nn.Module):
    def __init__(self, node_features, horizon, num_nodes):
        super (DynamicGraphConvNet, self).__init__ ()
        self.attention_layer = AttentionLayer (node_features, horizon)
        self.graph_conv = GraphConvolution (node_features * horizon, horizon)

    def forward(self, x):
        # x shape: (batch_size, horizon, num_nodes)
        batch_size = x.size (0)
        num_nodes = x.size (2)

        # Generate dynamic adjacency matrix
        adjacency = self.attention_layer (x)

        # Prepare data for graph convolution
        x = x.permute (0, 2, 1).reshape (batch_size * num_nodes,
                                         -1)  # (batch_size * num_nodes, node_features * horizon)

        # Apply graph convolution
        x = self.graph_conv (x, adjacency)
        x = x.view (batch_size, num_nodes, -1)  # reshape back to (batch_size, num_nodes, horizon)

        return x


