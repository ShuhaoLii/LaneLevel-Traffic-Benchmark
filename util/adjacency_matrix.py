import torch
import numpy as np
import  pandas as pd
import scipy.sparse as sp

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_adj_PEMS():
    A = pd.read_csv ("../datasets/PEMS_adj.csv", index_col=0)
    A = np.array(A)
    adj_mx = A.astype(np.float32)
    # adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    return adj_mx

def load_adj_PEMSF():
    A = pd.read_csv ("../datasets/PEMSF_adj.csv", index_col=0)
    A = np.array(A)
    adj_mx = A.astype(np.float32)
    # adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    return adj_mx

def load_adj_HuanNan():
    A = pd.read_csv ("../datasets/Huanan_adj.csv", index_col=0)
    A = np.array(A)
    adj_mx = A.astype(np.float32)
    # adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

    return adj_mx


def make_graph_inputs(name_dataset,device):
    if name_dataset == 'HuaNan':
        adj_mx = load_adj_HuanNan()
        adj = torch.tensor(adj_mx).cuda(device)
    elif name_dataset == 'PEMS' :
        adj_mx = load_adj_PEMS()
        adj = torch.tensor(adj_mx).cuda(device)
    elif name_dataset == 'PEMSF' :
        adj_mx = load_adj_PEMSF()
        adj = torch.tensor(adj_mx).cuda(device)
    return adj