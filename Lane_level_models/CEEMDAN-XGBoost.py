import numpy as np
import xgboost as xgb #install
from PyEMD import CEEMDAN #install
import pandas as pd

# traffic_data 是一个形状为 (batch_size, seq_len, num_nodes, input_dim) 的NumPy数组
traffic_data = np.load('')

# 使用CEEMDAN分解交通流数据
def decompose_traffic_data(traffic_data, num_nodes, seq_len):
    ceemdan = CEEMDAN()
    imfs = np.zeros((num_nodes, seq_len, traffic_data.shape[2]))  # 存储每个节点的IMF分量

    for i in range(num_nodes):
        signal = traffic_data[:, :, i, 0].reshape(-1)
        imfs[i, :, :] = ceemdan(signal)

    return imfs

imfs = decompose_traffic_data(traffic_data, num_nodes=40, seq_len=12)

# 训练XGBoost模型
def train_xgboost_models(imfs, horizon, num_nodes):
    xgboost_models = {}

    for i in range(num_nodes):
        for j in range(imfs.shape[2]):
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            X = imfs[i, :-horizon, j]  # 使用除了最后horizon个数据点之外的所有数据作为特征
            y = imfs[i, horizon:, j]  # 预测未来horizon个数据点
            model.fit(X, y)
            xgboost_models[(i, j)] = model

    return xgboost_models

xgboost_models = train_xgboost_models(imfs, horizon=12, num_nodes=40)

# 预测
def predict_traffic_flow(xgboost_models, imfs, horizon, num_nodes):
    final_predictions = np.zeros((num_nodes, horizon))

    for i in range(num_nodes):
        predictions = np.zeros(horizon)
        for j in range(imfs.shape[2]):
            X_test = imfs[i, -horizon:, j]
            predictions += xgboost_models[(i, j)].predict(X_test)

        final_predictions[i, :] = predictions

    return final_predictions

final_predictions = predict_traffic_flow(xgboost_models, imfs, horizon=12, num_nodes=40)

# 最终预测结果为 (num_nodes, horizon) 形状的数组
