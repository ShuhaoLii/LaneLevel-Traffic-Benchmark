import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, in_channels, num_node, output_size, batch_size, seq_length,device, hidden_size = 64, num_layers = 2):
        super (LSTM, self).__init__ ()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_node = num_node
        self.num_directions = 1
        self.num_layers = num_layers
        self.relu = nn.ReLU (inplace=True)
        self.device = device

        self.lstm = nn.LSTM (input_size=seq_length, hidden_size=output_size, num_layers=num_layers, batch_first=True)

    #32 12 1 20 1
    def forward(self, x):
        x= x.squeeze().permute(0,2,1)
        batch_size, seq_len = x.size ()[0], x.size ()[1]
        h_0 = torch.randn (self.num_directions * self.num_layers, x.size (0), self.output_size).to(self.device)
        c_0 = torch.randn (self.num_directions * self.num_layers, x.size (0), self.output_size).to(self.device)
        x, _ = self.lstm(x, (h_0, c_0))  #32 20,3
        return x  #y 32 20 12