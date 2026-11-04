import os
import sys
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

import torchinfo
import thop


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== Environment & Library Versions ===")
print(f"Python: {sys.version.split()[0]}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"torchinfo: {torchinfo.__version__}")
print(f"thop: {thop.__version__}")
print(f"Device: {device}")
print("======================================\n")

os.makedirs("results", exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)




class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



class PatchAttention(nn.Module):
    def __init__(self, patch_size=7, in_channels=1, d_model=64, nhead=4, dim_feedforward=128, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Linear(self.patch_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, C * p * p)

        x_proj = self.projection(patches)
        out = self.transformer(x_proj)
        out_pooled = out.mean(dim=1)
        logits = self.classifier(out_pooled)
        return logits


models = {
    'MLP': MLP().to(device),
    'CNN': CNN().to(device),
    'Attention': PatchAttention().to(device)
}


print("- Summary of parameters and FLOPs ---")
dummy_input_b1 = torch.randn(1, 1, 28, 28).to(device)
dummy_input_b128 = torch.randn(128, 1, 28, 28).to(device)

model_stats = {}

for name, model in models.items():
    model.eval()
    params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs, _ = thop.profile(model, inputs=(dummy_input_b1,), verbose=False)
    flops_thop = macs * 2

    for _ in range(10):
        _ = model(dummy_input_b1)

    times_b1 = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = model(dummy_input_b1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_b1.append((t1 - t0) * 1000)
    lat_b1 = np.median(times_b1)

    for _ in range(5):
        _ = model(dummy_input_b128)

    times_b128 = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = model(dummy_input_b128)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_b128.append((t1 - t0) * 1000)
    lat_b128 = np.median(times_b128)

    model_stats[name] = {
        'Params': params_count,
        'THOP_FLOPS_B1': flops_thop,
        'Lat_B1_ms': lat_b1,
        'Lat_B128_ms': lat_b128
    }

    print(f"[{name}] Parameters: {params_count:,} | THOP FLOPS (B=1): {flops_thop:,} | Latency B1: {lat_b1:.3f}ms | Latency B128: {lat_b128:.3f}ms")


def train_and_evaluate(model, train_loader, test_loader, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

    test_acc = (correct / total) * 100.0
    return test_acc

print("\n- Starting training (5 epochs per model) ---")
for name, model in models.items():
    acc = train_and_evaluate(model, train_loader, test_loader, epochs=5, lr=1e-3)
    model_stats[name]['Test_Accuracy'] = acc
    print(f"[{name}] Accuracy on the test set: {acc:.2f}%")

# Save summary CSV
df_stats = pd.DataFrame.from_dict(model_stats, orient='index')
df_stats.to_csv("results/model_summary.csv")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name in model_stats:
    plt.scatter(model_stats[name]['Params'], model_stats[name]['Test_Accuracy'], s=120, label=name)
    plt.annotate(name, (model_stats[name]['Params'], model_stats[name]['Test_Accuracy']),
                 textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlabel("Number of parameters")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs. Parameters")
plt.grid(True)

plt.subplot(1, 2, 2)
for name in model_stats:
    plt.scatter(model_stats[name]['Lat_B128_ms'], model_stats[name]['Test_Accuracy'], s=120, label=name)
    plt.annotate(name, (model_stats[name]['Lat_B128_ms'], model_stats[name]['Test_Accuracy']),
                 textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlabel("Inference latency B128 (ms)")
plt.ylabel("Test accuracy (%)")
plt.title("Accuracy vs. Latency (Batch 128)")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/tradeoff_plots.png")
plt.close()

print("\nStartup completed successfully. The charts and summary have been saved in the results/directory.")