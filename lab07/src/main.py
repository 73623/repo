import os
import time
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 768)
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 3x3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class StudentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def measure_cpu_latency(model, runs=100):
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_input = torch.randn(1, 1, 28, 28)
    # Warmup
    for _ in range(10):
        _ = model_cpu(dummy_input)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model_cpu(dummy_input)
    end = time.perf_counter()
    return ((end - start) / runs) * 1000.0  # ms per sample

def get_disk_size_mb(model, filename, sparse=False):
    path = os.path.join("results", filename)
    state = model.state_dict()
    if sparse:
        state = {k: v.to_sparse() if v.ndim >= 2 else v for k, v in state.items()}
    torch.save(state, path)
    return os.path.getsize(path) / (1024 * 1024)

def train_standard(model, epochs=5, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


if __name__ == "__main__":
    results_log = {}

    # Step 1: Oversized Baseline
    print("--- Step 1: Training Oversized Baseline ---")
    baseline = train_standard(BaselineCNN(), epochs=6)
    base_acc = evaluate_accuracy(baseline, test_loader)
    base_params = count_parameters(baseline)
    base_size = get_disk_size_mb(baseline, "baseline_dense.pt")
    base_lat = measure_cpu_latency(baseline)
    results_log["baseline"] = {"params": base_params, "acc": base_acc, "size_mb": base_size, "latency_ms": base_lat}
    print(f"Baseline | Params: {base_params:,} | Acc: {base_acc:.2f}% | Size: {base_size:.2f} MB | Latency: {base_lat:.3f} ms")

    # Step 2: Unstructured & Structured Pruning
    print("\n--- Step 2: Magnitude Pruning ---")
    sparsities = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
    unstruct_accs = []

    for s in sparsities:
        m = copy.deepcopy(baseline)
        if s > 0:
            params_to_prune = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
            prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=s)
            # Fine-tune 3 epochs with mask fixed
            train_standard(m, epochs=3, lr=5e-4)
            for mod, name in params_to_prune:
                prune.remove(mod, name)
        acc = evaluate_accuracy(m, test_loader)
        unstruct_accs.append(acc)
        print(f"Unstructured Sparsity {s*100:.0f}% -> Acc: {acc:.2f}%")
        if s == 0.9:
            dense_90_size = get_disk_size_mb(m, "pruned90_dense.pt", sparse=False)
            sparse_90_size = get_disk_size_mb(m, "pruned90_sparse.pt", sparse=True)
            lat_90 = measure_cpu_latency(m)
            results_log["unstructured_90"] = {
                "acc": acc, "dense_mb": dense_90_size, "sparse_mb": sparse_90_size, "latency_ms": lat_90
            }

    struct_ratios = [0.10, 0.25, 0.50]
    struct_accs = []
    for r in struct_ratios:
        m = copy.deepcopy(baseline)
        conv_layers = [mod for mod in m.modules() if isinstance(mod, nn.Conv2d)]
        for mod in conv_layers:
            prune.ln_structured(mod, name="weight", amount=r, n=2, dim=0)
        train_standard(m, epochs=3, lr=5e-4)
        for mod in conv_layers:
            prune.remove(mod, "weight")
        acc = evaluate_accuracy(m, test_loader)
        struct_accs.append(acc)
        print(f"Structured L2 Filter Pruning {r*100:.0f}% -> Acc: {acc:.2f}%")
        if r == 0.50:
            results_log["structured_50"] = {
                "acc": acc, "size_mb": get_disk_size_mb(m, "struct50.pt"), "latency_ms": measure_cpu_latency(m)
            }

    # Plot Sparsity Curves
    plt.figure(figsize=(8, 5))
    plt.plot([s * 100 for s in sparsities], unstruct_accs, marker="o", label="Unstructured Global L1")
    plt.plot([r * 100 for r in struct_ratios], struct_accs, marker="s", label="Structured L2 Filters")
    plt.xlabel("Sparsity / Filter Removal (%)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Sparsity on Fashion-MNIST")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/sparsity_curves.png")
    plt.close()

    # Step 4: Knowledge Distillation
    print("\n--- Step 4: Knowledge Distillation ---")
    plain_student = train_standard(StudentCNN(), epochs=6)
    plain_acc = evaluate_accuracy(plain_student, test_loader)
    student_params = count_parameters(plain_student)
    student_size = get_disk_size_mb(plain_student, "student.pt")
    student_lat = measure_cpu_latency(plain_student)

    # Distilled Student (T=4, alpha=0.5)
    distilled_student = StudentCNN().to(device)
    optimizer = torch.optim.Adam(distilled_student.parameters(), lr=1e-3)
    T, alpha = 4.0, 0.5
    baseline.eval()

    for epoch in range(6):
        distilled_student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            student_logits = distilled_student(x)
            with torch.no_grad():
                teacher_logits = baseline(x)
            ce_loss = F.cross_entropy(student_logits, y)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean"
            ) * (T ** 2)
            loss = alpha * ce_loss + (1.0 - alpha) * kl_loss
            loss.backward()
            optimizer.step()

    distill_acc = evaluate_accuracy(distilled_student, test_loader)
    results_log["plain_student"] = {"params": student_params, "acc": plain_acc, "size_mb": student_size, "latency_ms": student_lat}
    results_log["distilled_student"] = {"params": student_params, "acc": distill_acc, "size_mb": student_size, "latency_ms": student_lat}

    print(f"Plain Student     | Params: {student_params:,} | Acc: {plain_acc:.2f}% | Size: {student_size:.2f} MB | Latency: {student_lat:.3f} ms")
    print(f"Distilled Student | Params: {student_params:,} | Acc: {distill_acc:.2f}% | Size: {student_size:.2f} MB | Latency: {student_lat:.3f} ms")

    with open("results/metrics.json", "w") as f:
        json.dump(results_log, f, indent=4)
    print("\nAll results saved to results/ directory.")