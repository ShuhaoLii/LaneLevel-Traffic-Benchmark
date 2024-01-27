import torch.nn as nn
import torch
from torch.nn import Conv3d,Conv2d

class TMCNN(nn.Module):
    def __init__(self, in_channels, num_node, output_size, batch_size, seq_length,device):
        super (TMCNN, self).__init__ ()
        self.in_channels = in_channels
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_node = num_node
        self.relu = nn.ReLU (inplace=True)

        self.end_conv_1 = Conv3d (seq_length,output_size, (1,num_node,1), bias=True)
        self.end_conv_2 = Conv3d (output_size, output_size*num_node, 1, bias=True)
        # self.end_conv_3 = Conv2d (output_size*num_node, num_node, (1, 1), bias=True)
        self.fc = nn.Linear (output_size*num_node, output_size)


    #32 12 1 20 1
    def forward(self, x):
        x = self.end_conv_1 (x)
        x = self.end_conv_2(x).squeeze(2).reshape(self.batch_size,self.num_node,self.output_size) # 32 240 1 1
        return x                      #y 32 20 12
