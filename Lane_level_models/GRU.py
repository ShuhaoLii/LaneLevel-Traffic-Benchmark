import torch.nn as nn
import torch
from torch.nn import Conv3d,Conv2d

class GRU(nn.Module):
    def __init__(self, in_channels, num_node, output_size, batch_size, seq_length,device, hidden_size = 64, num_layers = 2):
        super (GRU, self).__init__ ()
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


        self.gru = nn.GRU (input_size=seq_length, hidden_size=output_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(seq_length, num_node)

    #32 12 1 20 1
    def forward(self, x):
        x= x.squeeze().permute(0,2,1)
        x, _ = self.gru(x)
        return x  #y 32 20 12