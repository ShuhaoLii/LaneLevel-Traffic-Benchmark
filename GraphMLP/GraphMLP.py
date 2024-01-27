import torch
import torch.nn as nn

from GraphMLP.mlp import RevIN
from GraphMLP.mlp import ResBlock
from GraphMLP.graph_attention import DynamicGraphConvNet
from GraphMLP.gatingnet import GatingNetwork


class GraphMLP(nn.Module):
    """Implementation of GraphMLP."""

    def __init__(self, input_dim, num_nodes, pred_len, batch_size, seq_len,device,n_block=2, dropout=0.05, ff_dim=2048, target_slice=slice(0,None,None)):
        super(GraphMLP, self).__init__()
        input_shape = (seq_len,num_nodes)
        self.target_slice = target_slice
        
        self.rev_norm = RevIN(input_shape[-1])
        self.DynamicGraphConvNet = DynamicGraphConvNet(input_dim,pred_len,num_nodes)
        self.gatingnet = GatingNetwork(num_nodes,pred_len)
        
        self.res_blocks = nn.ModuleList([ResBlock(input_shape, dropout, ff_dim) for _ in range(n_block)])
        
        self.linear = nn.Linear(input_shape[0], pred_len)
        
    def forward(self, x):
        x =  x.squeeze()
        x = self.rev_norm(x, 'norm')
        
        for res_block in self.res_blocks:
            x = res_block(x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        x = self.rev_norm(x, 'denorm', self.target_slice)
        x1 = x.transpose(1,2)
        x2 = self.DynamicGraphConvNet(x)
        output = self.gatingnet(x1,x2)

        return output