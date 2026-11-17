import os
import sys
import math
import time
import random

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

print("=== Environment & Library Versions ===")
print(f"Python: {sys.version.split()[0]}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"torch: {torch.__version__}")
print("======================================\n")

os.makedirs("results", exist_ok=True)



class Value:
    """Stores a single scalar value and its gradient, building a computation graph."""

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1.0))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = min(max(self.data, -50.0), 50.0)
        e = math.exp(x)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        val = max(self.data, 1e-12)
        out = Value(math.log(val), (self,), 'log')

        def _backward():
            self.grad += (1.0 / val) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        x = min(max(self.data, -50.0), 50.0)
        s = 1.0 / (1.0 + math.exp(-x))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += (s * (1.0 - s)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()



print("[Step 4] Verifying scalar autograd against PyTorch & Numerical Gradients...")


a_vals = [2.0, -3.0, 1.5]
b_vals = [-1.5, 2.0, 4.0]
c_vals = [1.0, 4.0, -7.0]


a_custom = [Value(v) for v in a_vals]
b_custom = [Value(v) for v in b_vals]
c_custom = [Value(v) for v in c_vals]

out_custom = sum(((a * b + c) ** 2).relu() for a, b, c in zip(a_custom, b_custom, c_custom))
out_custom.backward()

grads_a_custom = np.array([v.grad for v in a_custom])
grads_b_custom = np.array([v.grad for v in b_custom])
grads_c_custom = np.array([v.grad for v in c_custom])


a_pt = torch.tensor(a_vals, dtype=torch.float64, requires_grad=True)
b_pt = torch.tensor(b_vals, dtype=torch.float64, requires_grad=True)
c_pt = torch.tensor(c_vals, dtype=torch.float64, requires_grad=True)

out_pt = torch.relu((a_pt * b_pt + c_pt) ** 2).sum()
out_pt.backward()


eps = 1e-6


def f_eval(av, bv, cv):
    return sum(max(0.0, (x * y + z) ** 2) for x, y, z in zip(av, bv, cv))


num_grads_a = []
for i in range(len(a_vals)):
    a_plus = list(a_vals)
    a_minus = list(a_vals)
    a_plus[i] += eps
    a_minus[i] -= eps
    num_grads_a.append((f_eval(a_plus, b_vals, c_vals) - f_eval(a_minus, b_vals, c_vals)) / (2 * eps))
num_grads_a = np.array(num_grads_a)

max_diff_pt_a = np.max(np.abs(grads_a_custom - a_pt.grad.numpy()))
max_diff_pt_b = np.max(np.abs(grads_b_custom - b_pt.grad.numpy()))
max_diff_pt_c = np.max(np.abs(grads_c_custom - c_pt.grad.numpy()))
max_diff_num_a = np.max(np.abs(grads_a_custom - num_grads_a))

overall_max_diff = max(max_diff_pt_a, max_diff_pt_b, max_diff_pt_c, max_diff_num_a)
print(f"Max absolute difference vs PyTorch & Numerical: {overall_max_diff:.2e}")
assert overall_max_diff < 1e-6, f"Gradient verification failed! Diff: {overall_max_diff}"
print("Gradient check passed (< 1e-6).\n")



print("[Step 5] Training 2->8->1 MLP on make_moons (Custom Autograd vs PyTorch)...")

X_data, y_data = make_moons(n_samples=200, noise=0.1, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.3, stratify=y_data, random_state=RANDOM_SEED
)


class ScalarMLP:
    def __init__(self, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = [[Value(rng.randn() * 0.5) for _ in range(8)] for _ in range(2)]
        self.b1 = [Value(0.0) for _ in range(8)]
        self.W2 = [Value(rng.randn() * 0.5) for _ in range(8)]
        self.b2 = Value(0.0)

    def parameters(self):
        params = []
        for row in self.W1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.W2)
        params.append(self.b2)
        return params

    def forward(self, x):
        h = []
        for j in range(8):
            act = x[0] * self.W1[0][j] + x[1] * self.W1[1][j] + self.b1[j]
            h.append(act.relu())
        out = self.b2
        for j in range(8):
            out = out + h[j] * self.W2[j]
        return out.sigmoid()


def run_custom_training(epochs=15, lr=0.1):
    model = ScalarMLP(seed=RANDOM_SEED)
    params = model.parameters()
    for _ in range(epochs):
        total_loss = Value(0.0)
        for xi, yi in zip(X_train, y_train):
            pred = model.forward(xi)
            if yi == 1:
                loss_i = -pred.log()
            else:
                loss_i = -(Value(1.0) - pred).log()
            total_loss = total_loss + loss_i
        loss = total_loss * (1.0 / len(X_train))
        for p in params:
            p.grad = 0.0
        loss.backward()
        for p in params:
            p.data -= lr * p.grad

    correct = 0
    for xi, yi in zip(X_test, y_test):
        pred = model.forward(xi).data
        if (1 if pred >= 0.5 else 0) == yi:
            correct += 1
    return correct / len(X_test)


class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


def run_pytorch_training(epochs=15, lr=0.1):
    torch.manual_seed(RANDOM_SEED)
    model = TorchMLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        X_te = torch.tensor(X_test, dtype=torch.float32)
        preds = (model(X_te).squeeze() >= 0.5).numpy().astype(int)
        acc = np.mean(preds == y_test)
    return acc



_ = run_custom_training(epochs=2)
_ = run_pytorch_training(epochs=2)


custom_times, custom_accs = [], []
pt_times, pt_accs = [], []

for _ in range(5):
    t0 = time.perf_counter()
    acc_c = run_custom_training(epochs=15, lr=0.1)
    custom_times.append((time.perf_counter() - t0) * 1000.0)
    custom_accs.append(acc_c)

    t0 = time.perf_counter()
    acc_p = run_pytorch_training(epochs=15, lr=0.1)
    pt_times.append((time.perf_counter() - t0) * 1000.0)
    pt_accs.append(acc_p)

med_custom_ms = np.median(custom_times)
med_pt_ms = np.median(pt_times)
speedup = med_custom_ms / med_pt_ms

print(f"Custom Scalar Autograd -> Median Time (5 runs): {med_custom_ms:.2f} ms | Acc Test: {np.median(custom_accs) * 100:.1f}%")
print(f"PyTorch Vectorized -> Median Time (5 runs): {med_pt_ms:.2f} ms | Acc Test: {np.median(pt_accs) * 100:.1f}%")
print(f"PyTorch Speedup Factor: {speedup:.1f}x faster\n")



print("[Step 6] PyTorch Graph Inspection, Eager vs Traced/Compiled, and Contiguity...")


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


demo_model = DemoNet().eval()
sample_in = torch.randn(32, 64)

tracked_model = torch.jit.trace(demo_model, sample_in)
print("- Traced Computation Graph (IR) ---")
print(tracked_model.graph)

compiled_label = "torch.compile"
try:
    compiled_model = torch.compile(demo_model, backend="eager")
    with torch.no_grad():
        _ = compiled_model(sample_in)
except Exception as exc:
    print(f"Note: torch.compile fallback triggered ({type(exc).__name__}); using torch.jit.trace.")
    compiled_model = tracked_model
    compiled_label = "torch.jit.trace"

with torch.no_grad():
    for _ in range(10):
        _ = demo_model(sample_in)
        _ = compiled_model(sample_in)

# Benchmark Eager vs Compiled/Traced (100 runs)
eager_times, comp_times = [], []
with torch.no_grad():
    for _ in range(100):
        t0 = time.perf_counter()
        _ = demo_model(sample_in)
        eager_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = compiled_model(sample_in)
        comp_times.append((time.perf_counter() - t0) * 1000.0)

med_eager_ms = np.median(eager_times)
med_comp_ms = np.median(comp_times)
print(f"Eager Execution Median (100 runs):            {med_eager_ms:.4f} ms")
print(f"{compiled_label} Execution Median (100 runs): {med_comp_ms:.4f} ms\n")

# Tensor Metadata & Non-Contiguous View Speed Test
print("- Tensor Metadata & Contiguity Overhead ---")
t_contig = torch.randn(1024, 1024)
t_non_contig = t_contig.t()

print(f"Original Tensor -> shape: {tuple(t_contig.shape)}, dtype: {t_contig.dtype}, is_contiguous: {t_contig.is_contiguous()}, stride: {t_contig.stride()}")
print(f"Transposed View -> shape: {tuple(t_non_contig.shape)}, dtype: {t_non_contig.dtype}, is_contiguous: {t_non_contig.is_contiguous()}, stride: {t_non_contig.stride()}")

for _ in range(10):
    _ = t_contig.view(-1)
    _ = t_non_contig.reshape(-1)

contig_op_times, non_contig_op_times = [], []
for _ in range(100):
    t0 = time.perf_counter()
    _ = t_contig.reshape(-1).sum()
    contig_op_times.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    _ = t_non_contig.reshape(-1).sum()
    non_contig_op_times.append((time.perf_counter() - t0) * 1000.0)

med_contig_op = np.median(contig_op_times)
med_non_contig_op = np.median(non_contig_op_times)
print(f"Contiguous Tensor Op Median (100 runs):     {med_contig_op:.4f} ms")
print(f"Non-Contiguous View Op Median (100 runs):   {med_non_contig_op:.4f} ms (Slowdown: {med_non_contig_op / med_contig_op:.2f}x)")

summary_df = pd.DataFrame([
    {"Experiment": "Gradient Max Abs Diff", "Metric": "Max Error", "Value": f"{overall_max_diff:.2e}"},
    {"Experiment": "Moons Training (Scalar Autograd)", "Metric": "Median Time (ms)", "Value": f"{med_custom_ms:.2f}"},
    {"Experiment": "Moons Training (PyTorch)", "Metric": "Median Time (ms)", "Value": f"{med_pt_ms:.2f}"},
    {"Experiment": "Inference (Eager)", "Metric": "Median Time (ms)", "Value": f"{med_eager_ms:.4f}"},
    {"Experiment": f"Inference ({compiled_label})", "Metric": "Median Time (ms)", "Value": f"{med_comp_ms:.4f}"},
    {"Experiment": "Contiguous Tensor Op", "Metric": "Median Time (ms)", "Value": f"{med_contig_op:.4f}"},
    {"Experiment": "Non-Contiguous View Op", "Metric": "Median Time (ms)", "Value": f"{med_non_contig_op:.4f}"},
])
summary_df.to_csv("results/lab05_timings.csv", index=False)
print("\nSaved timing summary to results/lab05_timings.csv.")
