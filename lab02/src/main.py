import os
import sys
import random
import time

import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------

# ---------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
except ImportError:
    torch = None

print("=== Environment & Library Versions ===")
print(f"Python: {sys.version.split()[0]}")
print(f"numpy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scipy: {scipy.__version__}")
print(f"scikit-learn: {__import__('sklearn').__version__}")
print(f"matplotlib: {matplotlib.__version__}")
if torch:
    print(f"torch: {torch.__version__}")
print("======================================\n")


os.makedirs("results", exist_ok=True)



X, y = load_digits(return_X_y=True)
X = X / 16.0  # Scale inputs by dividing by 16


num_classes = 10
Y_onehot = np.zeros((y.size, num_classes))
Y_onehot[np.arange(y.size), y] = 1


X_train, X_test, y_train_oh, y_test_oh, y_train_raw, y_test_raw = train_test_split(
    X, Y_onehot, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED
)

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")



def initialize_parameters(input_dim, hidden_dim, output_dim, seed=42):
    np.random.seed(seed)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros((hidden_dim,))
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    b2 = np.zeros((hidden_dim,))
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    b3 = np.zeros((output_dim,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def forward_propagation(X, params):
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    A1 = relu(Z1)

    Z2 = np.dot(A1, params["W2"]) + params["b2"]
    A2 = relu(Z2)

    Z3 = np.dot(A2, params["W3"]) + params["b3"]
    A3 = softmax(Z3)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache



dummy_params = initialize_parameters(64, 16, 10)
dummy_out, _ = forward_propagation(X_train[:32], dummy_params)
assert dummy_out.shape == (32, 10), f"Expected shape (32, 10), got {dummy_out.shape}"
assert np.allclose(np.sum(dummy_out, axis=1), 1.0, atol=1e-6), "Row sums do not equal 1!"
print("Forward pass shape and row-sum checks passed successfully.")


def compute_loss(Y_true, Y_pred):
    m = Y_true.shape[0]
    Y_pred_clipped = np.clip(Y_pred, 1e-15, 1.0 - 1e-15)
    loss = -np.sum(Y_true * np.log(Y_pred_clipped)) / m
    return loss



untrained_out, _ = forward_propagation(X_train, dummy_params)
untrained_loss = compute_loss(y_train_oh, untrained_out)
print(f"Untrained network loss: {untrained_loss:.4f} (Expected around ln(10) ~ 2.3026)")


def backward_propagation(Y_true, params, cache):
    m = Y_true.shape[0]
    A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2 = cache["Z1"], cache["Z2"]
    X = cache["X"]

    dZ3 = A3 - Y_true
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0) / m

    dA2 = np.dot(dZ3, params["W3"].T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m

    dA1 = np.dot(dZ2, params["W2"].T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check():
    params = initialize_parameters(64, 16, 10)
    X_sample = X_train[:5]
    y_sample = y_train_oh[:5]

    _, cache = forward_propagation(X_sample, params)
    grads = backward_propagation(y_sample, params, cache)

    eps = 1e-5
    checked = 0
    passed = 0

    keys_to_check = ["W1", "W2", "W3"]
    for key in keys_to_check:
        for _ in range(3):
            idx = tuple(np.random.randint(0, dim) for dim in params[key].shape)

            grad_analytic = grads[key][idx]

            params[key][idx] += eps
            loss_plus, _ = forward_propagation(X_sample, params)
            loss_p = compute_loss(y_sample, loss_plus)

            params[key][idx] -= 2 * eps
            loss_minus, _ = forward_propagation(X_sample, params)
            loss_m = compute_loss(y_sample, loss_minus)

            params[key][idx] += eps

            grad_num = (loss_p - loss_m) / (2 * eps)

            rel_error = np.abs(grad_analytic - grad_num) / (np.abs(grad_analytic) + np.abs(grad_num) + 1e-12)
            checked += 1
            if rel_error < 1e-6:
                passed += 1

    print(f"Gradient Check: {passed}/{checked} weights passed with relative error < 1e-6.")
    assert passed >= 9, "Gradient check failed: fewer than 9 of 10 weights met the threshold."


gradient_check()

# ---------------------------------------------------------
# 6. Training Loop & Hyperparameter Grid Search
# ---------------------------------------------------------
learning_rates = [0.001, 0.01, 0.1]
hidden_widths = [16, 64]
epochs = 50
batch_size = 32

results_summary = []
plt.figure(figsize=(10, 6))

for hidden in hidden_widths:
    for lr in learning_rates:
        params = initialize_parameters(64, hidden, 10, seed=42)
        loss_history = []

        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                out, cache = forward_propagation(X_batch, params)
                grads = backward_propagation(y_batch, params, cache)

                for k in params:
                    params[k] -= lr * grads[k]

            train_out, _ = forward_propagation(X_train, params)
            epoch_loss = compute_loss(y_train_oh, train_out)
            loss_history.append(epoch_loss)

        test_out, _ = forward_propagation(X_test, params)
        test_preds = np.argmax(test_out, axis=1)
        test_acc = np.mean(test_preds == y_test_raw) * 100

        results_summary.append({"Width": hidden, "LR": lr, "Accuracy Test (%)": test_acc})
        plt.plot(loss_history, label=f"W={hidden}, lr={lr}")

plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Curves across Learning Rates and Hidden Widths")
plt.legend()
plt.grid(True)
plt.savefig("results/loss_curves.png")
plt.close()

print("\n- Training Results Summary ---")
df_results = pd.DataFrame(results_summary)
print(df_results.to_string(index=False))

# ---------------------------------------------------------
# 7. Timing Diagnostics (Median of 100 runs)
# ---------------------------------------------------------
print("\nRunning timing diagnostics...")
timing_params = initialize_parameters(64, 64, 10)
sample_batch = X_train[:32]
sample_y = y_train_oh[:32]

# Warm up once
_, cache_warm = forward_propagation(sample_batch, timing_params)
_ = backward_propagation(sample_y, timing_params, cache_warm)

forward_times = []
backward_times = []

for _ in range(100):
    t0 = time.perf_counter()
    _, cache = forward_propagation(sample_batch, timing_params)
    t1 = time.perf_counter()
    forward_times.append(t1 - t0)

    t0 = time.perf_counter()
    _ = backward_propagation(sample_y, timing_params, cache)
    t1 = time.perf_counter()
    backward_times.append(t1 - t0)

median_fwd = np.median(forward_times) * 1000  # ms
median_bwd = np.median(backward_times) * 1000  # ms

print(f"Median Forward Pass Time (32 samples): {median_fwd:.4f} ms")
print(f"Median Backward Pass Time (32 samples): {median_bwd:.4f} ms")
print("Timing diagnostics complete. All results saved successfully under results/.")