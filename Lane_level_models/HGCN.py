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

class TemporalAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.V = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        scores = torch.matmul(torch.tanh(torch.matmul(x, self.W)), self.V)
        attention_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * attention_weights, dim=1)

class HGCN(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon, batch_size, seq_len,device, gcn_output_dim=16, gru_hidden_dim=64, gru_num_layers=2):
        super (HGCN, self).__init__ ()
        self.gcn_output_dim = gcn_output_dim
        self.horizon = horizon
        self.gcn = GCNLayer (input_dim, gcn_output_dim)
        self.gru = GRULayer (num_nodes * gcn_output_dim, gru_hidden_dim, gru_num_layers)
        self.temporal_attention = TemporalAttentionLayer (gru_hidden_dim)
        self.fc = nn.Linear (gru_hidden_dim, num_nodes * horizon)

    def forward(self, x, adjacency_matrix):
        # x shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
        batch_size, seq_len, input_dim, num_nodes, _ = x.shape

        # Reshape and apply GCN for each time step
        x = x.view (batch_size * seq_len, num_nodes, input_dim)
        x = self.gcn (x, adjacency_matrix)
        x = x.view (batch_size, seq_len, num_nodes * self.gcn_output_dim)

        # Apply GRU
        gru_outputs = self.gru (x)

        # Apply Temporal Attention
        attention_output = self.temporal_attention (gru_outputs)

        # Final output
        output = self.fc (attention_output)
        return output.view (batch_size, num_nodes, self.horizon)





