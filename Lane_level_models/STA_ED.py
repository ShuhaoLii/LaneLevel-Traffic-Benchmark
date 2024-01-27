import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_nodes):
        super(SpatialAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, num_nodes))
        self.b = nn.Parameter(torch.randn(num_nodes))
        self.V = nn.Parameter(torch.randn(num_nodes))

    def forward(self, x):
        # x shape: (batch_size * seq_len, num_nodes, input_dim)
        x = torch.tanh(torch.matmul(x, self.W) + self.b)
        scores = torch.matmul(x, self.V)
        attention_weights = F.softmax(scores, dim=1)
        return attention_weights.unsqueeze(2)

class TemporalAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, seq_len):
        super(TemporalAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b = nn.Parameter(torch.randn(hidden_dim))
        self.V = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        x = torch.tanh(torch.matmul(x, self.W) + self.b)
        scores = torch.matmul(x, self.V.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=1)
        return attention_weights.unsqueeze(-1)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super(Encoder, self).__init__()
        self.spatial_attention = SpatialAttentionLayer(input_dim, num_nodes)
        self.lstm = nn.LSTM(input_size=num_nodes * input_dim, hidden_size=hidden_dim, batch_first=True)
        self.hidden_fc = nn.Linear (hidden_dim, num_nodes)  # 新增全连接层

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, _,num_nodes, _ = x.shape
        x = x.view(batch_size * seq_len, num_nodes, -1)  # Merge batch_size and seq_len

        # Apply spatial attention
        spatial_attention = self.spatial_attention(x)
        x = x * spatial_attention

        x = x.view(batch_size, seq_len, -1)  # Reshape back

        # LSTM
        outputs, (hidden, cell) = self.lstm(x)
        hidden = self.hidden_fc (hidden.permute (1, 0, 2)).permute (1, 0, 2)  # 调整隐藏状态尺寸
        cell = self.hidden_fc (cell.permute (1, 0, 2)).permute (1, 0, 2)  # 调整单元状态尺寸
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim, num_nodes, horizon, seq_len):
        super(Decoder, self).__init__()
        self.temporal_attention = TemporalAttentionLayer(hidden_dim, seq_len)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=num_nodes, batch_first=True)
        self.fc = nn.Linear(num_nodes, horizon)

    def forward(self, x, hidden, cell):
        # Apply temporal attention
        temporal_attention = self.temporal_attention(x)
        x = x * temporal_attention

        # LSTM
        outputs, _ = self.lstm(x, (hidden, cell))

        # Final output layer
        prediction = outputs.permute(0,2,1)
        return prediction


class STA_ED(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon, batch_size,seq_len,device,hidden_dim = 64 ):
        super(STA_ED, self).__init__()
        self.horizon = horizon
        self.encoder = Encoder(input_dim, hidden_dim, num_nodes)
        self.decoder = Decoder(hidden_dim, num_nodes, horizon, seq_len)

    def forward(self, x):
        encoder_outputs, hidden, cell = self.encoder(x)
        prediction = self.decoder(encoder_outputs, hidden, cell)
        prediction = prediction[:,:,:self.horizon]
        return prediction



