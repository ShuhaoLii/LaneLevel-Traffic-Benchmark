import torch
import torch.nn as nn
import torch.nn.functional as F
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def calculate_att_matrix(data):
    # data shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
    # 计算每个节点的平均速度
    avg_speed = data.mean (dim=1)  # (batch_size, input_dim, num_nodes, output_dim)
    num_nodes = avg_speed.size (2)

    # 初始化ATT矩阵
    att_matrix = torch.zeros ((num_nodes, num_nodes), dtype=torch.float32)

    # 计算ATT
    for i in range (num_nodes):
        for j in range (num_nodes):
            if i != j:
                # 这里假设节点间距离为1，可以根据实际情况调整
                att_matrix[i, j] = 1 / (avg_speed[0, 0, i, 0] + avg_speed[0, 0, j, 0]) / 2

    return att_matrix

def calculate_dtw_matrix(data):
    # data shape: (batch_size, seq_len, input_dim, num_nodes, output_dim)
    num_nodes = data.size(3)
    seq_len = data.size(1)

    # 初始化DTW矩阵
    dtw_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # 计算DTW
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance, _ = fastdtw(data[0, :, 0, i, 0], data[0, :, 0, j, 0], dist=euclidean)
                dtw_matrix[i, j] = distance / seq_len  # 归一化

    return dtw_matrix


class GraphConvolutionLayer(nn.Module):
    def __init__(self,  out_features,output_dim,):
        super(GraphConvolutionLayer, self).__init__()
        self.fc = nn.Linear(out_features,output_dim)

    def forward(self, x, adjacency_matrices):
        # x shape: (batch_size, seq_len, num_nodes, output_dim)
        # adjacency_matrices shape: (batch_size, seq_len, num_nodes, num_nodes)
        batch_size, seq_len, num_nodes, output_dim = x.shape

        # 初始化输出
        output = torch.zeros(batch_size, seq_len, num_nodes, self.fc.out_features, device=x.device)

        # 逐时间步进行图卷积
        for b in range(batch_size):
            for t in range(seq_len):
                x_bt = x[b, t]  # (num_nodes, output_dim)
                adj_bt = adjacency_matrices[b, t].transpose(1,0)  # (num_nodes, num_nodes)

                # 计算邻接矩阵和特征矩阵的乘积
                support = torch.matmul(x_bt,adj_bt)

                # 应用线性变换并更新输出
                for n in range(num_nodes):
                    # 逐节点应用全连接层
                    output[b, t, n, :] = self.fc(support[n,n ].unsqueeze(0))

        return output

class SelfAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SelfAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes, in_features)
        batch_size, seq_len, num_nodes, _ = x.shape

        # Reshape x to (batch_size * seq_len * num_nodes, in_features)
        x_reshaped = x.view(-1, self.in_features)

        # 计算Q, K, V
        Q = self.query(x_reshaped)
        K = self.key(x_reshaped)
        V = self.value(x_reshaped)

        # 重塑为原始维度并计算注意力
        Q = Q.view(batch_size, seq_len, num_nodes, -1)
        K = K.view(batch_size, seq_len, num_nodes, -1)
        V = V.view(batch_size, seq_len, num_nodes, -1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.out_features ** 0.5
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)

        return output


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class STMGG(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon, batch_size, seq_len,device, hidden_dim = 64):
        super(STMGG, self).__init__()
        self.horizon = horizon
        self.gcn_att = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gcn_dtw = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gcn_avg = GraphConvolutionLayer(input_dim, hidden_dim)
        self.self_attention = SelfAttentionLayer(hidden_dim, hidden_dim)
        self.ffn = FFN(hidden_dim * num_nodes * 3, hidden_dim, num_nodes * horizon)

    def forward(self, x):
        batch_size, seq_len, _, num_nodes, _ = x.shape
        x = x.view(batch_size, seq_len, num_nodes, -1)

        att_adj = self.self_attention(x)  # 自注意力图
        dtw_adj = calculate_dtw_matrix(x)  # 动态时间扭曲图
        avg_adj = calculate_att_matrix(x)  # 平均旅行时间图

        x_att = self.gcn_att(x, att_adj)
        x_dtw = self.gcn_dtw(x, dtw_adj)
        x_avg = self.gcn_avg(x, avg_adj)

        x_combined = torch.cat((x_att, x_dtw, x_avg), dim=-1)
        x_combined = x_combined.view(batch_size, -1)

        output = self.ffn(x_combined)
        return output.view(batch_size, num_nodes, self.horizon)
