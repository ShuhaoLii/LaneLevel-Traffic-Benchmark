import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        # x shape: (batch_size, num_nodes, in_features)
        support = torch.matmul(adjacency_matrix, x)
        output = self.fc(support)
        return output

class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        output, _ = self.gru(x)
        return output

class GCN_GRU(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon, batch_size, seq_len,device, gcn_output_dim=16, gru_hidden_dim=64, gru_num_layers=2):
        super(GCN_GRU, self).__init__()
        self.horizon = horizon
        self.gcn = GCNLayer(input_dim, gcn_output_dim)
        self.gru = GRULayer(num_nodes * gcn_output_dim, gru_hidden_dim, gru_num_layers)
        self.fc = nn.Linear(gru_hidden_dim, num_nodes * horizon)

    def forward(self, x, adjacency_matrix):
        # x shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
        batch_size, seq_len, _, num_nodes, _ = x.shape

        # Reshape and apply GCN for each time step
        x = x.view(batch_size * seq_len, num_nodes, -1)
        x = self.gcn(x, adjacency_matrix)
        x = x.view(batch_size, seq_len, -1)

        # Apply GRU
        gru_outputs = self.gru(x)

        # Final output
        output = self.fc(gru_outputs[:, -1, :])
        return output.view(batch_size, num_nodes, self.horizon)


