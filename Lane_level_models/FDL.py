import torch.nn as nn
import torch
import numpy as np
from torch.nn import Conv3d,Conv2d

def entropy_weight(data):
    """
    Calculate the entropy weight.
    """
    # Normalize the data
    data_normalized = data / np.sum(data, axis=0)

    # Calculate the entropy
    epsilon = 1e-12  # To avoid log(0)
    entropy = -np.sum(data_normalized * np.log(data_normalized + epsilon), axis=0)

    # Calculate the weight
    weight = (1 - entropy) / (np.sum(1 - entropy))
    return weight

def grey_relational_analysis(target_sequence, comparative_sequences):
    """
    Perform grey relational analysis.
    """
    sequences = np.vstack((target_sequence, comparative_sequences))
    weight = entropy_weight(sequences)

    # Calculate grey relational coefficients
    min_diff = np.min(np.abs(sequences - target_sequence[:, None]), axis=0)
    max_diff = np.max(np.abs(sequences - target_sequence[:, None]), axis=0)

    rho = 0.5  # Distinguishing coefficient
    grc = (min_diff + rho * max_diff) / (np.abs(sequences - target_sequence[:, None]) + rho * max_diff)

    # Calculate grey relational grade
    grg = np.sum(grc * weight, axis=0)
    return grg


class FDL(nn.Module):
    def __init__(self, in_channels, num_node, output_size, batch_size, seq_length,device, hidden_size = 64, num_layers = 2):
        super (FDL, self).__init__ ()
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

        self.lstm = nn.LSTM (input_size=seq_length, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.gru = nn.GRU (input_size=hidden_size, hidden_size=output_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(seq_length, num_node)

    def entropy_grey_relational_analysis(self, x):
        grg = grey_relational_analysis(x,x)
        return grg


    #32 12 1 20 1
    def forward(self, x):
        x= x.squeeze().permute(0,2,1)
        batch_size, seq_len = x.size ()[0], x.size ()[1]
        h_0 = torch.randn (self.num_directions * self.num_layers, x.size (0), self.hidden_size).to(self.device)
        c_0 = torch.randn (self.num_directions * self.num_layers, x.size (0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h_0, c_0))  #32 20,3
        x, _ = self.gru(x)
        return x  #y 32 20 12