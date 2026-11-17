import os
import sys
import time
import random
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.profiler import profile, record_function, ProfilerActivity
import mlflow

# ---------------------------------------------------------
# 0. Load YAML Config, Set Seeds & Print Versions
# ---------------------------------------------------------
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

RANDOM_SEED = cfg["seed"]
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
print(f"matplotlib: {matplotlib.__version__}")
print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"mlflow: {mlflow.__version__}")
print(f"yaml: {yaml.__version__}")
print(f"Device: {device}")
print("======================================\n")

os.makedirs("results", exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((cfg["dataset"]["mean"],), (cfg["dataset"]["std"],))
])

full_train_dataset = FashionMNIST(
    root=cfg["dataset"]["root"], train=True, download=True, transform=transform
)
test_dataset = FashionMNIST(
    root=cfg["dataset"]["root"], train=False, download=True, transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [cfg["dataset"]["train_size"], cfg["dataset"]["val_size"]],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

def get_loaders(batch_size, num_workers=0):
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return train_loader, val_loader, test_loader

# ---------------------------------------------------------
# Best Model from Lab 3: CNN
# ---------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(32 * 7 * 7, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            out = model(X_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * y_b.size(0)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
    return (correct / total) * 100.0, total_loss / total


def run_baseline_mlflow():
    print("[Step 1] Running YAML-driven baseline training with MLflow logging...")
    mlflow.set_tracking_uri("file:./results/mlruns")
    mlflow.set_experiment("Lab06_FashionMNIST_CNN")

    torch.manual_seed(RANDOM_SEED)
    model = CNN(cfg["model"]["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    train_loader, val_loader, test_loader = get_loaders(
        cfg["training"]["batch_size"], num_workers=0
    )

    with mlflow.start_run(run_name="YAML_Baseline_CNN"):
        mlflow.log_params({
            "seed": RANDOM_SEED,
            "model": cfg["model"]["name"],
            "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"],
            "lr": cfg["training"]["lr"],
            "optimizer": cfg["training"]["optimizer"]
        })

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            model.train()
            running_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * y_b.size(0)

            train_loss = running_loss / len(train_dataset)
            val_acc, val_loss = evaluate(model, val_loader)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

        test_acc, _ = evaluate(model, test_loader)
        mlflow.log_metric("test_acc", test_acc)

        ckpt_path = cfg["training"]["checkpoint_path"]
        torch.save({
            "epoch": cfg["training"]["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, ckpt_path)
        mlflow.log_artifact(ckpt_path)
        mlflow.log_artifact(config_path)

    print(f"Baseline completed. Test Accuracy: {test_acc:.2f}% | MLflow logged to results/mlruns\n")


def run_profiler_epoch():
    print("[Step 2] Profiling 1 epoch with torch.profiler...")
    torch.manual_seed(RANDOM_SEED)
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    train_loader, _, _ = get_loaders(cfg["training"]["batch_size"], num_workers=0)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    t_data, t_fwd, t_bwd, t_opt = 0.0, 0.0, 0.0, 0.0
    model.train()

    with profile(activities=activities, record_shapes=True) as prof:
        data_iter = iter(train_loader)
        while True:
            t0 = time.perf_counter()
            with record_function("data_loading"):
                try:
                    X_b, y_b = next(data_iter)
                except StopIteration:
                    break
                X_b, y_b = X_b.to(device), y_b.to(device)
            t1 = time.perf_counter()
            t_data += (t1 - t0)

            optimizer.zero_grad()
            with record_function("forward_pass"):
                out = model(X_b)
                loss = criterion(out, y_b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            t_fwd += (t2 - t1)

            with record_function("backward_pass"):
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            t_bwd += (t3 - t2)

            with record_function("optimizer_step"):
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            t_opt += (t4 - t3)

    total_t = t_data + t_fwd + t_bwd + t_opt
    shares = {
        "Stage": ["Data Loading", "Forward Pass", "Backward Pass", "Optimizer Step"],
        "Time_s": [t_data, t_fwd, t_bwd, t_opt],
        "Share_pct": [
            (t_data / total_t) * 100,
            (t_fwd / total_t) * 100,
            (t_bwd / total_t) * 100,
            (t_opt / total_t) * 100
        ]
    }
    df_prof = pd.DataFrame(shares)
    df_prof.to_csv("results/profiler_breakdown.csv", index=False)
    print(df_prof.to_string(index=False))
    print("\nTop PyTorch Profiler Ops:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=6))
    print()


def run_workers_study():
    print("[Step 3] Studying num_workers [0, 1, 2, 4, 8]...")
    workers_list = cfg["studies"]["workers_list"]
    results = []

    for nw in workers_list:
        train_loader, _, _ = get_loaders(cfg["training"]["batch_size"], num_workers=nw)
        # Warm-up pass
        for _ in train_loader:
            break

        run_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for X_b, y_b in train_loader:
                X_b = X_b.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            run_times.append(time.perf_counter() - t0)

        med_time = np.median(run_times)
        throughput = len(train_dataset) / med_time
        results.append({"num_workers": nw, "Epoch_Time_s": med_time, "Throughput_samples_s": throughput})
        print(f"num_workers={nw} -> Median Epoch Load Time: {med_time:.3f}s ({throughput:.0f} samples/s)")

    df_w = pd.DataFrame(results)
    df_w.to_csv("results/workers_study.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(df_w["num_workers"], df_w["Epoch_Time_s"], marker="o", linewidth=2)
    plt.xlabel("num_workers")
    plt.ylabel("Median Data Loading Epoch Time (s)")
    plt.title("DataLoader Worker Scaling on Fashion-MNIST")
    plt.xticks(workers_list)
    plt.grid(True)
    plt.savefig("results/num_workers_plot.png")
    plt.close()
    print("Saved worker scaling plot to results/num_workers_plot.png\n")


def run_batch_size_study():
    print("[Step 4] Running Batch-Size Study [32, 64, 128, 256, 512] (5 epochs each)...")
    batch_sizes = cfg["studies"]["batch_sizes"]
    records = []

    for bs in batch_sizes:
        torch.manual_seed(RANDOM_SEED)
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
        train_loader, _, test_loader = get_loaders(bs, num_workers=0)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(5):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        throughput = (len(train_dataset) * 5) / elapsed
        if device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            # Estimate activation + parameter memory footprint on CPU (MB)
            param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            act_bytes = bs * (16 * 28 * 28 + 32 * 14 * 14) * 4
            peak_mem_mb = (param_bytes * 3 + act_bytes) / (1024 ** 2)

        test_acc, _ = evaluate(model, test_loader)
        records.append({
            "Batch_Size": bs,
            "Throughput_samples_s": round(throughput, 1),
            "Peak_Mem_MB": round(peak_mem_mb, 2),
            "Test_Accuracy_pct": round(test_acc, 2)
        })
        print(f"Batch {bs:3d} -> Throughput: {throughput:7.1f} samples/s | Peak Mem: {peak_mem_mb:6.2f} MB | Acc: {test_acc:.2f}%")

    df_bs = pd.DataFrame(records)
    df_bs.to_csv("results/batch_size_study.csv", index=False)
    print()


def run_scheduler_optimizer_study():
    print("[Step 5] Comparing Schedulers (StepLR, CosineAnnealingLR, ReduceLROnPlateau) & Optimizers (Adam vs SGD+Momentum)...")
    epochs = cfg["studies"]["scheduler_epochs"]
    train_loader, val_loader, test_loader = get_loaders(cfg["training"]["batch_size"], num_workers=0)

    configs = [
        ("Adam_StepLR", "Adam", "StepLR"),
        ("Adam_CosineAnnealingLR", "Adam", "Cosine"),
        ("Adam_ReduceLROnPlateau", "Adam", "Plateau"),
        ("SGD_Momentum_Cosine", "SGD", "Cosine")
    ]

    summary = []
    plt.figure(figsize=(9, 5))

    for label, opt_name, sched_name in configs:
        torch.manual_seed(RANDOM_SEED)
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()

        if opt_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

        if sched_name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        elif sched_name == "Cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        val_accs = []
        for ep in range(epochs):
            model.train()
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                optimizer.step()

            val_acc, val_loss = evaluate(model, val_loader)
            val_accs.append(val_acc)

            if sched_name == "Plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        test_acc, _ = evaluate(model, test_loader)
        summary.append({"Config": label, "Optimizer": opt_name, "Scheduler": sched_name, "Test_Accuracy_pct": round(test_acc, 2)})
        plt.plot(range(1, epochs + 1), val_accs, marker="o", label=f"{label} ({test_acc:.2f}%)")
        print(f"{label:26s} -> Final Test Accuracy: {test_acc:.2f}%")

    pd.DataFrame(summary).to_csv("results/scheduler_optimizer_study.csv", index=False)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("LR Schedulers & Optimizers Comparison (10 Epochs)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/scheduler_comparison.png")
    plt.close()
    print()


def run_gradient_accumulation():
    print("[Step 6] Verifying Gradient Accumulation (64 x 4 steps vs true Batch 256)...")
    criterion = nn.CrossEntropyLoss()

    # Run A: True Batch 256 (3 epochs)
    torch.manual_seed(RANDOM_SEED)
    model_true = CNN().to(device)
    opt_true = optim.Adam(model_true.parameters(), lr=1e-3)
    loader_256, _, test_loader = get_loaders(256, num_workers=0)

    losses_256 = []
    for _ in range(3):
        model_true.train()
        ep_loss = 0.0
        for X_b, y_b in loader_256:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt_true.zero_grad()
            loss = criterion(model_true(X_b), y_b)
            loss.backward()
            opt_true.step()
            ep_loss += loss.item() * y_b.size(0)
        losses_256.append(ep_loss / len(train_dataset))
    acc_256, _ = evaluate(model_true, test_loader)

    
    torch.manual_seed(RANDOM_SEED)
    model_accum = CNN().to(device)
    opt_accum = optim.Adam(model_accum.parameters(), lr=1e-3)
    loader_64, _, _ = get_loaders(cfg["studies"]["accum_micro_batch"], num_workers=0)
    accum_steps = cfg["studies"]["accum_steps"]

    losses_accum = []
    for _ in range(3):
        model_accum.train()
        ep_loss = 0.0
        opt_accum.zero_grad()
        for step_idx, (X_b, y_b) in enumerate(loader_64):
            X_b, y_b = X_b.to(device), y_b.to(device)
            raw_loss = criterion(model_accum(X_b), y_b)
            ep_loss += raw_loss.item() * y_b.size(0)
            scaled_loss = raw_loss / accum_steps
            scaled_loss.backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader_64):
                opt_accum.step()
                opt_accum.zero_grad()
        losses_accum.append(ep_loss / len(train_dataset))
    acc_accum, _ = evaluate(model_accum, test_loader)

    print(f"True Batch 256      -> Losses: {[round(l, 4) for l in losses_256]} | Test Acc: {acc_256:.2f}%")
    print(f"Accumulated (64x4)  -> Losses: {[round(l, 4) for l in losses_accum]} | Test Acc: {acc_accum:.2f}%\n")


def run_checkpoint_resume():
    print("[Step 7] Testing Checkpoint Save at Epoch 3 & Smooth Resume to Epoch 6...")
    train_loader, val_loader, _ = get_loaders(cfg["training"]["batch_size"], num_workers=0)
    criterion = nn.CrossEntropyLoss()
    resume_ckpt = "results/resume_epoch3.pt"

    
    torch.manual_seed(RANDOM_SEED)
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_curve = []

    for epoch in range(1, 4):
        model.train()
        ep_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * y_b.size(0)
        loss_curve.append(ep_loss / len(train_dataset))

    torch.save({
        "epoch": 3,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "loss_curve": loss_curve
    }, resume_ckpt)
    print(f"Interrupted at Epoch 3. Saved checkpoint -> {resume_ckpt}")

    
    model_resumed = CNN().to(device)
    opt_resumed = optim.Adam(model_resumed.parameters(), lr=1e-3)
    ckpt = torch.load(resume_ckpt, weights_only=False)
    model_resumed.load_state_dict(ckpt["model_state_dict"])
    opt_resumed.load_state_dict(ckpt["optimizer_state_dict"])
    torch.set_rng_state(ckpt["rng_state"])
    resumed_curve = list(ckpt["loss_curve"])

    for epoch in range(ckpt["epoch"] + 1, 7):
        model_resumed.train()
        ep_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt_resumed.zero_grad()
            loss = criterion(model_resumed(X_b), y_b)
            loss.backward()
            opt_resumed.step()
            ep_loss += loss.item() * y_b.size(0)
        resumed_curve.append(ep_loss / len(train_dataset))

    print(f"Resumed Epochs 1..6 Training Loss Curve: {[round(l, 4) for l in resumed_curve]}")

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 4), resumed_curve[:3], marker="o", label="Before Interrupt (Epochs 1-3)")
    plt.plot(range(3, 7), resumed_curve[2:], marker="s", linestyle="--", label="Resumed from Checkpoint (Epochs 3-6)")
    plt.axvline(3, color="red", linestyle=":", label="Interrupt & Save Point")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Checkpoint Interrupt & Smooth Resume Verification")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/checkpoint_resume.png")
    plt.close()
    print("Saved checkpoint resume curve to results/checkpoint_resume.png\n")

if __name__ == "__main__":
    run_baseline_mlflow()
    run_profiler_epoch()
    run_workers_study()
    run_batch_size_study()
    run_scheduler_optimizer_study()
    run_gradient_accumulation()
    run_checkpoint_resume()
    print("All Lab 6 experiments completed successfully!")