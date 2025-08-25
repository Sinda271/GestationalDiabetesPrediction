import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import json
from glob import glob
import tracemalloc
import time
from sklearn.preprocessing import StandardScaler


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.activation(out)
        return out

# ResNet for Tabular Data
class TabularResNet(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks=3, hidden_dim=128):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.blocks(x)
        x = self.output_layer(x)
        return x

# Training & Evaluation
def train_resnet_tabular(data_path, batch_size=1024, epochs=10000, lr=1e-3):

    df = pd.read_csv(data_path)
    X = df.drop(columns=["Diabetes_012"])
    y = df["Diabetes_012"]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = TabularResNet(input_dim=X.shape[1], num_classes=len(np.unique(y)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    epoch_times = []
    epoch_memory = []
    patience_counter = 0
    patience = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Memory and time tracking
        start_time = time.time()
        tracemalloc.start()
        # Training
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Memory and time tracking
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        epoch_times.append(end_time - start_time)
        epoch_memory.append({
            "current_MB": current / 1024 / 1024,
            "peak_MB": peak / 1024 / 1024
        })
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

        

        # Save the best model with patience  
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), f"./results/tabular_resnet_best_{dataset.split("/")[-1].strip(".csv")}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor.to(device))
        y_pred = torch.argmax(y_pred, dim=1).cpu()

    print(f"Accuracy: {accuracy_score(y_test_tensor, y_pred) * 100:.2f}%")
    print(f"F1 Score: {f1_score(y_test_tensor, y_pred, average='macro') * 100:.2f}%")
    print(f"Precision: {precision_score(y_test_tensor, y_pred, average='macro') * 100:.2f}%")
    print(f"Recall: {recall_score(y_test_tensor, y_pred, average='macro') * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_tensor, y_pred))

    # Convert confusion matrix to list of lists
    cm = confusion_matrix(y_test_tensor, y_pred)
    cm = [[int(x) for x in row] for row in cm]

    # Save results
    results = {
        "epochs": epochs,
        "confusion_matrix": cm,
        "epoch_times": epoch_times,
        "memory_usages": epoch_memory,
        "accuracy": accuracy_score(y_test_tensor, y_pred),
        "f1_score": f1_score(y_test_tensor, y_pred, average="macro"),
        "precision": precision_score(y_test_tensor, y_pred, average="macro"),
        "recall": recall_score(y_test_tensor, y_pred, average="macro"),
    }
    with open(f"./results/resnet_metrics_{data_path.split("/")[-1].strip(".csv")}.json", "w") as f:
        json.dump(results, f)



if __name__ == "__main__":
    dataset = "./processed_datasets/CL/dataset_dataset_without_da_pca_rfe_CL.csv"
    train_resnet_tabular(dataset)
