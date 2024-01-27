import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMSpeedProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_nodes):
        super(BiLSTMSpeedProcessor, self).__init__()
        self.num_nodes = num_nodes
        self.lstm = nn.LSTM(input_dim * num_nodes, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
        batch_size, seq_len, _, num_nodes, _ = x.shape
        x = x.view(batch_size, seq_len, -1)  # Reshape to (batch_size, seq_len, num_nodes * input_dim)
        x, _ = self.lstm(x)
        return x


class SpatialAttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim * 2, hidden_dim * 2))
        self.V = nn.Parameter(torch.randn(hidden_dim * 2))

    def forward(self, x):
        scores = torch.matmul(torch.tanh(torch.matmul(x, self.W)), self.V)
        attention_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(x * attention_weights, dim=1)

class TemporalAttentionDecoder(nn.Module):
    def __init__(self, hidden_dim, num_nodes, horizon):
        super(TemporalAttentionDecoder, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.fc = nn.Linear(hidden_dim * 2, num_nodes * horizon)

    def forward(self, x):
        return self.fc(x).view(-1, self.num_nodes, self.horizon)

class ST_AFN(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon, batch_size, seq_len,device, hidden_dim=64, num_layers=2):
        super(ST_AFN, self).__init__()
        self.speed_processor = BiLSTMSpeedProcessor(input_dim, hidden_dim, num_layers, num_nodes)
        self.spatial_attention = SpatialAttentionLayer(hidden_dim)
        self.temporal_decoder = TemporalAttentionDecoder(hidden_dim, num_nodes, horizon)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
        batch_size, seq_len, _, num_nodes, _ = x.shape


        # Speed processing
        x = self.speed_processor(x)

        # Spatial attention
        x = self.spatial_attention(x)

        # Temporal decoding
        output = self.temporal_decoder(x)
        return output

