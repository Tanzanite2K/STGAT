# 🌐 **STGAT — Spatio-Temporal Graph Attention Network for Traffic Flow Prediction**

This project implements **STGAT (Spatio-Temporal Graph Attention Network)** to predict traffic flow by learning **spatial dependencies** between road segments and **temporal patterns** across time.
The model combines **graph attention**, **temporal modeling**, and **sequence learning** to improve forecasting accuracy on road-traffic datasets.

---

## 🚀 **Key Features**

* **Graph Attention Network (GAT)** for capturing spatial relationships between nodes (road segments).
* **Temporal Convolution / GRU-LSTM** layers for modeling time-based dependencies.
* **ST-Attention Fusion** to jointly learn spatial + temporal dependencies.
* **Dynamic adjacency matrix** based on real traffic connectivity.
* **Multi-step traffic flow prediction**.
* **Fully modular PyTorch implementation**.
* **Training, validation, and evaluation scripts included**.

---

## 📁 **Project Structure**

```
STGAT/
│── data/                      # Traffic flow datasets
│── models/
│     ├── gat_layer.py         # Graph Attention Layer
│     ├── stgat.py             # Full STGAT model
│── utils/
│     ├── graph_utils.py       # Adjacency matrix + preprocessing
│     ├── data_loader.py       # Dataset preparation pipeline
│── train.py                   # Training script
│── test.py                    # Evaluation script
│── requirements.txt           # Dependencies
│── README.md                  # Documentation
```

---

## 🧠 **Model Architecture Overview**

**STGAT** is built using 3 major components:

### 1️⃣ **Spatial Module (Graph Attention Network)**

* Learns weighted relationships between nodes.
* Uses attention coefficients to focus on relevant neighbors.
* Handles dynamic edge weights.

### 2️⃣ **Temporal Module**

Can be implemented using:

* **Temporal Convolutional Networks (TCN)**
  or
* **Recurrent Networks (GRU / LSTM)**

This captures trends like:

* Peak hours
* Seasonal patterns
* Traffic fluctuations

### 3️⃣ **Fusion Layer**

Combines spatial graph features + temporal context → final output.

---

## 📊 **Dataset**

You can use any traffic dataset like:

* METR-LA
* PEMS-BAY
* PEMS-D / PEMS-04 / PEMS-08
* Custom city traffic dataset

Expected input shape:

```
(batch_size, time_steps, num_nodes, features)
```

---

## 🏋️ **Training the Model**

Run training:

```
python train.py
```

Adjust hyperparameters in `train.py`:

* Learning rate
* Batch size
* Hidden size
* Number of graph attention heads
* Number of past time steps
* Forecast horizon

---

## 🧪 **Testing / Evaluation**

```
python test.py
```

Metrics:

* MAE
* MAPE
* RMSE

---

## 🛠️ **Installation**

```
git clone https://github.com/Tanzanite2k/STGAT.git
cd STGAT
pip install -r requirements.txt
```

---

## 🤝 **Contributions**

Feel free to open:

* Issues
* Pull requests
* Feature suggestions

---

## 📄 **License**

This project uses the **MIT License**.

---

## ✨ **Author**

**Karri Pavan Prabhas**
B.Tech CSE — SRM University AP
AI/ML & Graph Neural Networks Enthusiast

## **Outputs**

<img width="871" height="679" alt="image" src="https://github.com/user-attachments/assets/d1c4a631-1d72-477f-a13b-4d9581d54023" />

