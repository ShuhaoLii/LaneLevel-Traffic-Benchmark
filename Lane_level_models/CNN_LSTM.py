import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
        batch_size, seq_len, _, num_nodes, _ = x.shape
        x = x.view(batch_size * seq_len, 1, num_nodes, -1)  # Reshape to merge batch_size and seq_len
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, seq_len, -1, num_nodes)  # Reshape back to separate batch_size and seq_len
        return x

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon, batch_size, seq_len,device,hidden_dim=16):
        super(CNNLSTM, self).__init__()
        self.horizon = horizon
        self.cnn = CNN(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim * num_nodes, hidden_size=num_nodes, num_layers=1, batch_first=True)
        self.fc = nn.Linear(num_nodes, horizon)

    def forward(self, x):
        batch_size = x.size(0)

        # Pass through CNN
        x = self.cnn(x)  # shape: (batch_size, seq_len, hidden_dim, num_nodes)
        x = x.view(batch_size, x.size(1), -1)  # Flatten spatial dimensions

        # Pass through LSTM
        x, _ = self.lstm(x)

        # Final output layer
        x = x.permute(0,2,1)
        x = x[:,:,:self.horizon]

        return x


