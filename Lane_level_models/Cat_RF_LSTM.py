import numpy as np
import torch
import torch.nn as nn
from catboost import CatBoostRegressor # install
from sklearn.ensemble import RandomForestRegressor



# CatBoost 模型
def CatBoost(X_train,y_train):
    cat_model = CatBoostRegressor (iterations=1000, learning_rate=0.1, depth=6)
    cat_model.fit (X_train, y_train)
    return cat_model


# 随机森林模型
def RandomForest(X_train,y_train):
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_train, y_train)
    return rf_model

# 训练CatBoost 模型
def train_CatBoost(X_train, y_train):
    cat_model = CatBoostRegressor (iterations=1000, learning_rate=0.1, depth=6)
    cat_model.fit (X_train, y_train)
    return cat_model

# cat_model = train_CatBoost()

# 训练随机森林模型
def train_rf(X_train, y_train):
    rf_model = RandomForestRegressor (n_estimators=100)
    rf_model.fit (X_train, y_train)
    return rf_model

# rf_model = train_rf()

class Cat_RF_LSTM(nn.Module):
    def __init__(self, input_dim, num_nodes, horizon,batch_size, seq_len,device, hidden_dim = 64, num_layers = 2 ):
        super(Cat_RF_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.lstm = nn.LSTM (input_dim * num_nodes, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear (hidden_dim, num_nodes * horizon)

    #input shape (B,seq_len,in_dim, num_nodes, out_dim)
    def forward(self, x):
        batch_size, seq_len, _,num_nodes, _ = x.shape
        # cat_predictions = cat_model.predict(x)
        # rf_predictions = rf_model.predict(x)
        x = x.view(batch_size, seq_len, -1)  # Reshape x to (batch_size, seq_len, num_nodes * input_dim)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # We only use the output of the last time step
        output =  x.view (-1, self.num_nodes, self.horizon)
        # output = (cat_predictions + rf_predictions + output) / 3
        return output




