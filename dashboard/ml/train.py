"""
การบ้านสัปดาห์ที่ 11 — Timeseries LSTM Dashboard

นักศึกษาต้องเขียนโค้ดในไฟล์นี้ให้สมบูรณ์ โดยต้องมีฟังก์ชัน:

    def event_stream():
        # generator ที่ yield dict ข้อมูลทีละ epoch แล้ว stream ผ่าน SSE

yield dict ที่มี key ต่อไปนี้ (แต่ละ epoch ระหว่างฝึกสอน):
    {
        "type": "progress",
        "epoch": <int>,
        "loss": <float>,
        "val_loss": <float>,
    }

และเมื่อฝึกสอนเสร็จ yield dict สุดท้าย:
    {
        "type": "done",
        "results": <ผลสรุป เช่น ลิงก์ไปยัง prediction_plot.png>,
    }

ข้อกำหนด:
    1. โหลดข้อมูลอนุกรมเวลา (จาก data/ เช่น temperature.csv)
    2. ใช้ sliding window แปลงข้อมูลเป็น supervised learning
    3. สร้างโมเดล LSTM (nn.LSTM + nn.Linear)
    4. Train/Val split (80/20)
    5. บันทึกกราฟ predicted vs actual เป็น PNG
       ที่ dashboard/static/dashboard/prediction_plot.png

ดูตัวอย่างโค้ดได้จาก slides wk11.html
"""

# TODO: เขียนโค้ดการบ้านตรงนี้
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error


# --- 1. Dataset สำหรับทำ Sliding Window ---
class TimeSeriesDataset(Dataset):
    """แปลงข้อมูล Time-series เป็น Supervised Learning ด้วย Sliding Window"""
    def __init__(self, data, window_size=24):
        self.x = []
        self.y = []
        for i in range(len(data) - window_size):
            self.x.append(data[i:i + window_size])
            self.y.append(data[i + window_size])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# --- 2. โมเดล LSTM และ Vanilla RNN ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# --- 3. Generator ฟังก์ชัน ---
def event_stream(model_type="lstm"):
    csv_path = 'data/temperature.csv'
    static_dir = 'dashboard/static/dashboard'
    os.makedirs(static_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    col_name = df.select_dtypes(include=[np.number]).columns[0]
    raw_data = df[col_name].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    window_size = 24
    dataset = TimeSeriesDataset(scaled_data, window_size=window_size)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_x, train_y = dataset.x[:train_size], dataset.y[:train_size]
    val_x, val_y = dataset.x[train_size:], dataset.y[train_size:]

    train_loader = DataLoader(list(zip(train_x, train_y)), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(val_x, val_y)), batch_size=32, shuffle=False)

    m_type = str(model_type).strip().lower()
    if m_type == 'rnn':
        model = RNNModel(input_size=1, hidden_size=64, num_layers=2)
        model_title = 'Vanilla RNN'
    else:
        model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
        model_title = 'LSTM'

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    num_epochs = 20

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_x)

        epoch_loss = float(total_train_loss / len(train_x))

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                v_pred = model(batch_x)
                v_loss = criterion(v_pred, batch_y)
                total_val_loss += v_loss.item() * len(batch_x)

        epoch_val_loss = float(total_val_loss / len(val_x))

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)

        yield {
            "type": "progress",
            "epoch": epoch,
            "loss": epoch_loss,
            "val_loss": epoch_val_loss
        }

    # บันทึก loss_curve.png
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title(f'{model_title} Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'loss_curve.png'))
    plt.close()

    # บันทึก prediction_plot.png
    model.eval()
    with torch.no_grad():
        preds_scaled = model(val_x).numpy()

    preds_actual = scaler.inverse_transform(preds_scaled)
    true_actual = scaler.inverse_transform(val_y.numpy())

    mse = mean_squared_error(true_actual, preds_actual)
    rmse = root_mean_squared_error(true_actual, preds_actual)

    plt.figure(figsize=(10, 5))
    plt.plot(true_actual[:150], label='Actual Values', color='blue', alpha=0.7)
    plt.plot(preds_actual[:150], label=f'{model_title} Predicted', color='orange', linestyle='--')
    plt.title(f'{model_title} - Predicted vs Actual (Validation Set) | RMSE: {rmse:.4f}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'prediction_plot.png'))
    plt.close()

# 6. บันทึกผลลัพธ์ (TODO: ปรับให้สร้างกราฟจริง)
    yield {
        "type": "done",
        "results": "/static/dashboard/prediction_plot.png"
    }


train = event_stream