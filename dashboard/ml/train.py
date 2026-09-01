import os
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def train(on_progress=None):
    torch.manual_seed(42)
    # 1. สร้าง Synthetic Dataset (200 samples, 2 features)
    X = torch.randn(200, 2)
    y = ((X[:, 0] * 1.5 + X[:, 1] - 0.5) > 0).float().unsqueeze(1)

    learning_rates = [0.001, 0.1, 1.0]
    num_epochs = 100
    loss_fn = nn.BCELoss()
    history = {}
    summary = []

    for lr in learning_rates:
        # Perceptron: Linear(2, 1) + Sigmoid
        model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        losses = []

        for epoch in range(num_epochs):
            y_hat = model(X)
            loss = loss_fn(y_hat, y)

            loss.backward()

            # Manual Weight & Bias Update (ไม่ใช้ optimizer)
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
                model.zero_grad()

            preds = (y_hat >= 0.5).float()
            acc = (preds == y).float().mean().item()
            losses.append(loss.item())

            # ส่งความคืบหน้าแบบ Realtime เฉพาะรอบ LR = 0.1
            if on_progress and lr == 0.1:
                on_progress(epoch + 1, loss.item(), acc)
                time.sleep(0.03)

        history[lr] = losses
        summary.append({
            "lr": lr,
            "final_loss": round(losses[-1], 4),
            "final_acc": round(acc * 100, 2)
        })

    # บันทึกรูปกราฟ Loss Curve
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'dashboard')
    os.makedirs(static_dir, exist_ok=True)
    save_path = os.path.join(static_dir, 'loss_curve.png')

    plt.figure(figsize=(8, 4.5))
    for lr, losses in history.items():
        plt.plot(losses, label=f'LR = {lr}')
    plt.title('Perceptron Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return summary