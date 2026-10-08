# Trains a small neural network to predict wine quality from different values, printing timing and train/validation/test metrics each epoch.
#
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)

# data: 70% train / 20% validation / 10% test
df = pd.read_csv(Path(__file__).parent / "Data" / "winequality-red.csv")
X = df.drop(columns="quality").values
y = df["quality"].values

X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.20 / 0.90, random_state=42)

scaler = StandardScaler().fit(X_train) # statistics from training data only
X_train, X_val, X_test = [torch.tensor(scaler.transform(a), dtype=torch.float32)
                          for a in (X_train, X_val, X_test)]
y_train, y_val, y_test = [torch.tensor(a, dtype=torch.float32).view(-1, 1)
                          for a in (y_train, y_val, y_test)]

# model, loss, optimizer
model = nn.Sequential(
    nn.Linear(11, 20), nn.ReLU(),
    nn.Linear(20, 20), nn.ReLU(),
    nn.Linear(20, 10), nn.ReLU(),
    nn.Linear(10, 1),
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X)
    err = pred - y
    mse = loss_fn(pred, y).item()
    mae = err.abs().mean().item()
    r2 = 1 - (err ** 2).sum().item() / ((y - y.mean()) ** 2).sum().item()
    return mse, mae, r2


# train
EPOCHS, BATCH_SIZE = 100, 100

for epoch in range(1, EPOCHS + 1):
    t0 = time.perf_counter() # time the training pass only
    model.train()
    order = torch.randperm(len(X_train))
    for i in range(0, len(X_train), BATCH_SIZE):
        idx = order[i:i + BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        loss.backward()
        optimizer.step()
    epoch_time = time.perf_counter() - t0

    train_mse, _, _ = evaluate(X_train, y_train)
    val_mse, val_mae, val_r2 = evaluate(X_val, y_val)
    test_mse, test_mae, test_r2 = evaluate(X_test, y_test) # monitoring only, not used for any decision
    print(f"Epoch {epoch:3d}/{EPOCHS} | train time {epoch_time * 1000:6.1f} ms | train mse {train_mse:.4f} | "
          f"val mse {val_mse:.4f} mae {val_mae:.4f} r2 {val_r2:.3f} | "
          f"test mse {test_mse:.4f} mae {test_mae:.4f} r2 {test_r2:.3f}")

# evaluate on the test set
mse, mae, r2 = evaluate(X_test, y_test)
print(f"\nTest | mse {mse:.4f} | mae {mae:.4f} | r2 {r2:.3f}")


# to be done:
# save model and size
# track memory usage
# find inference cost
# all validation metrics per epoch
# all training metrics per epoch