# import argparse
# import yaml
import numpy as np
import os
import torch
import time
from util.adjacency_matrix import make_graph_inputs
from Lane_level_models.FDL import FDL
from Lane_level_models.GRU import GRU
from Lane_level_models.LSTM import LSTM
from Lane_level_models.TMCNN import TMCNN
from Lane_level_models.MDL import MDL
from Lane_level_models.CNN_LSTM import CNNLSTM
from Lane_level_models.STA_ED import STA_ED
from Lane_level_models.Cat_RF_LSTM import Cat_RF_LSTM
from Lane_level_models.HGCN import HGCN
from Lane_level_models.GCN_GRU import GCN_GRU
from Lane_level_models.ST_AFN import ST_AFN
from Lane_level_models.STMGG import STMGG
from GraphMLP.GraphMLP import GraphMLP
from Lane_level_models.DLinear import Dlinear
from Graph_model.AGCRN import AGCRN
from Graph_model.MTGNN import MTGNN


device = torch.device ("cuda:1")

def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DataLoader (object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        print (len (xs), batch_size)
        if pad_with_last_sample:
            num_padding = (batch_size - (len (xs) % batch_size)) % batch_size
            x_padding = np.repeat (xs[-1:], num_padding, axis=0)
            y_padding = np.repeat (ys[-1:], num_padding, axis=0)
            xs = np.concatenate ([xs, x_padding], axis=0)
            ys = np.concatenate ([ys, y_padding], axis=0)
        self.size = len (xs)
        self.num_batch = int (self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation (self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min (self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper ()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(dataset_dir, batch_size,horizon):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['y_' + category] = data['y_' + category][:,0:int(horizon),:,:]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)
    data['scaler'] = scaler

    return data


def get_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
             y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy (x).float ()
    y = torch.from_numpy (y).float ()
    # print ("X: {}".format (x.size ()))
    # print ("y: {}".format (y.size ()))
    x = x.permute (1, 0, 2, 3)
    y = y.permute (1, 0, 2, 3)
    return x, y


def get_x_y_in_correct_dims(x, y):
    """
    :param x: shape (seq_len, batch_size, num_sensor, input_dim)
    :param y: shape (horizon, batch_size, num_sensor, input_dim)
    :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
            y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    batch_size = x.size (1)
    x = x.view (seq_len, batch_size, num_nodes, input_dim)
    y = y[..., :output_dim].view (horizon, batch_size,
                                  y.size(2), output_dim)
    return x, y


def prepare_data(x, y):
    x, y = get_x_y (x, y)
    x, y = get_x_y_in_correct_dims (x, y)
    return x.to (device), y.to (device)


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float ()
    mask /= mask.mean ()
    loss = torch.abs (y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean ()


def compute_loss(y_true, y_predicted):
    y_true = standard_scaler.inverse_transform (y_true)
    y_predicted = standard_scaler.inverse_transform (y_predicted)
    return masked_mae_loss (y_predicted, y_true)


def mask_and_fillna(loss, mask):
    loss = loss * mask
    loss = torch.where (torch.isnan (loss), torch.zeros_like (loss), loss)
    return torch.mean (loss)


def calc_metrics(preds, labels, null_val=0.):
    preds = torch.from_numpy (preds)
    labels = torch.from_numpy (labels)
    if np.isnan (null_val):
        mask = ~torch.isnan (labels)
    else:
        mask = (labels != null_val)
    mask = mask.float ()
    mask /= torch.mean (mask)
    mask = torch.where (torch.isnan (mask), torch.zeros_like (mask), mask)
    mse = (preds - labels) ** 2
    mae = torch.abs (preds - labels)
    mape = mae / labels
    mae, mape, mse = [mask_and_fillna (l, mask) for l in [mae, mape, mse]]

    rmse = torch.sqrt (mse)
    return mae, mape, rmse


def evaluate(model, dataset='val', batches_seen=0):
    """
    Computes mean L1Loss
    :return: mean L1Loss
    """
    with torch.no_grad ():
        model = model.eval ()

        val_iterator = data['{}_loader'.format (dataset)].get_iterator ()
        losses = []

        y_truths = []
        y_preds = []

        for _, (x, y) in enumerate (val_iterator):
            x, y = prepare_data (x, y)
            x = x.unsqueeze (4).permute (1, 0, 3, 2, 4)
            y = y.squeeze (3).permute (1, 2, 0)
            if is_graph_based:
                adj = make_graph_inputs (dataset_name,device)
                output = model (x, adj)
            else:
                output = model (x)
            loss = compute_loss (y, output)
            losses.append (loss.item ())
            y_truths.append (y.cpu ())
            y_preds.append (output.cpu ())

        mean_loss = np.mean (losses)
        y_preds = np.concatenate (y_preds, axis=1)
        y_truths = np.concatenate (y_truths, axis=1)  # concatenate on batch dimension

        y_truths_scaled = []
        y_preds_scaled = []
        for t in range (y_preds.shape[0]):
            y_truth = standard_scaler.inverse_transform (y_truths[t])
            y_pred = standard_scaler.inverse_transform (y_preds[t])
            y_truths_scaled.append (y_truth)
            y_preds_scaled.append (y_pred)
        y_preds_scaled = np.array (y_preds_scaled)
        y_truths_scaled = np.array (y_truths_scaled)
        mae, mape, rmse = calc_metrics (y_preds_scaled, y_truths_scaled, null_val=0.0)
        print ('mae:', mae.item (), 'mape:', mape.item (), 'rmse:', rmse.item ())
        return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}


if __name__ == '__main__':
    # data
    dataset_dir = './datasets/PEMSF'
    dataset_name = "PEMS" #'HuaNan' or 'PEMS'or 'PEMSF'
    batch_size = 64
    horizon = 6
    input_dim = 1
    seq_len = 12
    output_dim = 1
    num_nodes = 43
    # Lane_level_models
    hidden_dim = [64, 64, horizon]
    kernel_size = (3, 3)
    num_layers = 3
    batch_first = True
    bias = True
    return_all_layers = False
    is_graph_based = False
    # train
    base_lr = 0.01
    epsilon = 1.0e-3
    steps = [20, 30, 40, 50]
    lr_decay_ratio = 0.1
    epochs = 100
    max_grad_norm = 5
    patience = 10
    min_val_loss = float ('inf')

    data = load_dataset (dataset_dir, batch_size, horizon)
    standard_scaler = data['scaler']

    model = AGCRN(input_dim, num_nodes, horizon, batch_size, seq_len,device)
    model.to (device)

    """
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    optimizer = torch.optim.Adam (model.parameters (), lr=base_lr, eps=epsilon)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR (optimizer, milestones=steps,
                                                         gamma=lr_decay_ratio)

    num_batches = data['train_loader'].num_batch
    print ('num params:', count_parameters (model))

    for epoch_num in range (0, epochs):

        model = model.train ()

        train_iterator = data['train_loader'].get_iterator ()
        losses = []
        for _, (x, y) in enumerate (train_iterator):
            optimizer.zero_grad ()
            x, y = prepare_data (x, y)  # H,B,N,D
            x = x.unsqueeze (4).permute (1, 0, 3, 2, 4)
            y = y.squeeze (3).permute (1, 2, 0)  # x:32 12 1 20 1 y 32 20 12
            if is_graph_based:
                adj = make_graph_inputs(dataset_name,device)
                output = model(x,adj)
            else:
                output = model (x)
            loss = compute_loss (y, output)

            print (loss.item ())

            losses.append (loss.item ())

            loss.backward ()

            # gradient clipping - this does it in place
            torch.nn.utils.clip_grad_norm_ (model.parameters (), max_grad_norm)

            optimizer.step ()
        print ("epoch complete")
        lr_scheduler.step ()
        print ("evaluating now!")

        val_loss, _ = evaluate (model, dataset='val')

        end_time = time.time ()

    test_loss, _ = evaluate (model, dataset='test')
