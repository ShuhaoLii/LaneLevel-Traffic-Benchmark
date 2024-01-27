# Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline

Welcome to the Git Hub repository for the Lane-Level Traffic Prediction Benchmark. This repository contains all the code and datasets related to our paper: "Unifying Lane-Level Traffic Prediction from a Graph Structural Perspective: Benchmark and Baseline." This project aims to provide a standardized platform for evaluating and comparing the performance of various lane-level traffic prediction models.

### Contents

 This repository includes the reproduced codes of the models and their corresponding abbreviated names mentioned in the paper. If you are the author of any of these models or if you find any issues with the reproduction, we welcome your contribution to help us build a better lane-level traffic prediction code library.

| Paper                                                        | Model           |
| ------------------------------------------------------------ | --------------- |
| Lane-Level Traffic Flow Prediction with Heterogeneous Data and Dynamic Graphs | HGCN            |
| Lane-Level Traffic Flow Prediction Based on Dynamic Graph Generation | DGCN            |
| Lane-Level Traffic Speed Forecasting: A Novel Mixed Deep Learning Model | MDL             |
| A Hybrid Ensemble Model for Urban Lane-Level Traffic Flow Prediction | Cat-RF-LSTM     |
| ST-AFN: a spatial-temporal attention based fusion network for lane-level traffic | ST-AFN          |
| Two-Stream Multi-Channel Convolutional Neural Network (TM-CNN) for MultiLane Traffic Speed Prediction Considering Traffic Volume Impact | TM-CNN          |
| A Hybrid Model for Lane-Level Traffic Flow Forecasting Based on Complete Ensemble Empirical Mode Decomposition and Extreme Gradient Boosting | CEEMDAN-XGBoost |
| Multi-Lane Short-Term Traffic Forecasting With Convolutional LSTM Network | CNN-LSTM        |
| Short-term prediction of lane-level traffic speeds: A fusion deep learning model | FDL             |
| Lane-Level Heterogeneous Traffic Flow Prediction: A Spatiotemporal Attention-Based Encoder– Decoder Model | STA-ED          |
| A Dynamic Spatio-Temporal Deep Learning Model for Lane-Level Traffic Prediction | GCN-GRU         |
| Modeling Dynamic Traffic Flow as Visibility Graphs: A Network-Scale Prediction Framework for Lane-Level Traffic Flow Based on LPR Data | STMGG           |

Additionally, the repository features implementations for models such as:
- LSTM
- GRU
- DLinear：Are transformers effective for time series forecasting?

And two graph-structured models:

- AGCRN : Spatio-temporal meta-graph learning for traffic forecasting
- MTGNN : Connecting the dots: Multivariate time series forecasting with graph neural networks

The source codes for some models mentioned in the paper, provided by their respective authors, can be downloaded from the following links:

- MDL :  https://github.com/lwqs93/MDL
- ST-AFN : https://github.com/lwqs93/MDL
- DCRNN : https://github.com/liyaguang/DCRNN
- STGCN : https://github.com/VeritasYin/STGCN_IJCAI-18
- MTGNN : https://github.com/nnzhan/MTGNN
- ASTGCN : https://github.com/wanhuaiyu/ASTGCN
- GraphWaveNet : https://github.com/nnzhan/Graph-WaveNet
- STSGCN : https://github.com/Davidham3/STSGCN
- AGCRN : https://github.com/LeiBAI/AGCRN
- STGODE : https://github.com/LeiBAI/AGCRN
- MegaCRN : https://github.com/LeiBAI/AGCRN



### Requirements

- Python 3.8

- torch

- numpy

- pandas

- catboost 

- sklearn

- PyEMD

- xgboost

- scipy

- fastdtw

  

### Data Preparation

In the "datasets" directory, you can find pre-processed files for three datasets, including the training, validation, and test sets, as well as the raw files for the datasets. You can choose to process these raw files according to your requirements. We also provide adjacency matrices based on neighboring relationships for each dataset, as some models require this information.



### Training

- Before using the GCN-GRU or HGCN models, you need to adjust the hyperparameter 'is_graph_based = Ture' in the main.py file.
- You can run any model except for the Cat-RF-LSTM and CEEMDAN-XGBoost models directly from the main file, simply by changing the model name, such as:

```python
model = AGCRN(input_dim, num_nodes, horizon, batch_size, seq_len,device)
```

- Training command:

```python
python main.py
```

- The Cat-RF-LSTM and CEEMDAN-XGBoost models do not use the typical deep learning training methods but are trained in stages in their respective folders under the "Lane_level_models" directory.

