"""数值分析与人工智能综合实验报告——受控实验脚本。

图表输出到 ../实验结果/图表/报告实验/：
  - 二次型收敛性与稳定性.png
  - 梯度有限差分误差.png
  - Adam与SGD基线对比.png、学习率敏感性.png
  - 矩估计系数敏感性.png、稳定项敏感性.png
数据表输出到 ../实验结果/数据表/报告实验/：
  - 二次型实验汇总.csv、梯度检验数据.csv
  - CNN实验汇总.csv、CNN训练历史.csv

在 `jup` 环境中运行：
  MPLBACKEND=Agg python 代码/报告实验脚本.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
SPLIT_SEED = 2026
SEEDS = [2026, 2027, 2028]
EPOCHS = 18
BATCH_SIZE = 64
DEVICE = torch.device("cpu")  # the jup environment has the CPU build of PyTorch
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "实验结果"
FIG_DIR = OUT / "图表" / "报告实验"
DATA_DIR = OUT / "数据表" / "报告实验"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 1. Quadratic problem: step-size stability, momentum, Adam, scale invariance
# ---------------------------------------------------------------------------
def quadratic_experiment() -> None:
    d = 20
    lam = np.logspace(0.0, 2.0, d)  # eigenvalues 1 ... 100, condition number 100
    theta0 = np.ones(d)
    f0 = 0.5 * np.sum(lam * theta0**2)
    T = 1000

    def ratio(theta: np.ndarray) -> float:
        return float(0.5 * np.sum(lam * theta**2) / f0)

    def run_gd(alpha: float, scale: float = 1.0) -> np.ndarray:
        theta = theta0.copy()
        hist = np.empty(T)
        for t in range(T):
            hist[t] = scale * ratio(theta)
            g = scale * lam * theta
            theta = theta - alpha * g
            if not np.all(np.isfinite(theta)):
                hist[t + 1 :] = np.inf
                break
        return hist

    def run_momentum(alpha: float, beta: float = 0.9, scale: float = 1.0) -> np.ndarray:
        theta = theta0.copy()
        velocity = np.zeros(d)
        hist = np.empty(T)
        for t in range(T):
            hist[t] = scale * ratio(theta)
            g = scale * lam * theta
            velocity = beta * velocity + g
            theta = theta - alpha * velocity
            if not np.all(np.isfinite(theta)):
                hist[t + 1 :] = np.inf
                break
        return hist

    def run_adam(
        alpha: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        scale: float = 1.0,
    ) -> np.ndarray:
        theta = theta0.copy()
        m = np.zeros(d)
        v = np.zeros(d)
        hist = np.empty(T)
        for t in range(1, T + 1):
            hist[t - 1] = scale * ratio(theta)
            g = scale * lam * theta
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g * g
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            theta = theta - alpha * m_hat / (np.sqrt(v_hat) + eps)
            if not np.all(np.isfinite(theta)):
                hist[t:] = np.inf
                break
        return hist

    alphas = [1e-3, 1e-2, 1.9e-2, 2.1e-2, 5e-2]
    gd_hist = {alpha: run_gd(alpha) for alpha in alphas}
    momentum_hist = run_momentum(1e-2)
    adam_hist = run_adam(1e-2)
    adam_scaled_hist = run_adam(1e-2, scale=1e3)
    gd_scaled_hist = run_gd(1e-2, scale=1e3)

    # ---- figure: three panels -------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(alphas)))
    for alpha, color in zip(alphas, colors):
        h = np.clip(gd_hist[alpha], 1e-16, 1e12)
        label = f"$\\alpha$={alpha:g}" + (" (diverge)" if alpha >= 2e-2 else "")
        ax.plot(h, color=color, lw=1.6, label=label)
    ax.axhline(1e-6, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("$f(\\theta_t)/f(\\theta_0)$")
    ax.set_title("(a) GD: step-size stability")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.plot(np.clip(gd_hist[1e-2], 1e-16, 1e12), label="GD", lw=1.6)
    ax.plot(np.clip(momentum_hist, 1e-16, 1e12), label="Momentum", lw=1.6)
    ax.plot(np.clip(adam_hist, 1e-16, 1e12), label="Adam", lw=1.6)
    ax.axhline(1e-6, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("$f(\\theta_t)/f(\\theta_0)$")
    ax.set_title("(b) GD vs Momentum vs Adam")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[2]
    ax.plot(np.clip(adam_hist, 1e-16, 1e12), label="Adam, $f$", lw=1.6)
    ax.plot(np.clip(adam_scaled_hist, 1e-16, 1e12), label="Adam, $10^3 f$", lw=1.6, ls="--")
    ax.plot(np.clip(gd_scaled_hist, 1e-16, 1e12), label="GD, $10^3 f$ (diverge)", lw=1.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration $t$")
    ax.set_ylabel("$f(\\theta_t)/f(\\theta_0)$")
    ax.set_title("(c) Scale invariance")
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "二次型收敛性与稳定性.png", bbox_inches="tight")
    plt.close(fig)

    # ---- table ----------------------------------------------------------------
    def iterations_to_tol(hist: np.ndarray, tol: float = 1e-6) -> int | None:
        idx = np.where(hist <= tol)[0]
        return int(idx[0] + 1) if idx.size else None

    rows = []
    for alpha in alphas:
        h = gd_hist[alpha]
        rows.append(
            {
                "method": "GD",
                "setting": f"alpha={alpha:g}",
                "diverged": bool(np.any(~np.isfinite(h)) or np.nanmax(h) > 1e6),
                "iter_to_1e-6": iterations_to_tol(h),
                "final_ratio": float(h[-1]) if np.isfinite(h[-1]) else np.inf,
            }
        )
    for name, h in [("Momentum", momentum_hist), ("Adam", adam_hist)]:
        rows.append(
            {
                "method": name,
                "setting": "alpha=0.01",
                "diverged": bool(np.any(~np.isfinite(h)) or np.nanmax(h) > 1e6),
                "iter_to_1e-6": iterations_to_tol(h),
                "final_ratio": float(h[-1]) if np.isfinite(h[-1]) else np.inf,
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "二次型实验汇总.csv", index=False, encoding="utf-8-sig")

    print("Quadratic experiment done.")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3e}"))


# ---------------------------------------------------------------------------
# 2. Data and model for the CNN experiment
# ---------------------------------------------------------------------------
digits = load_digits()
X = (digits.images / 16.0).astype(np.float32)
y = digits.target.astype(np.int64)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SPLIT_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SPLIT_SEED, stratify=y_temp
)


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    shuffle: bool,
    generator_seed: int | None = None,
) -> DataLoader:
    images_t = torch.from_numpy(images).unsqueeze(1)  # N, 1, 8, 8
    labels_t = torch.from_numpy(labels)
    dataset = TensorDataset(images_t, labels_t)
    generator = None
    if generator_seed is not None:
        generator = torch.Generator().manual_seed(generator_seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


val_loader = make_loader(X_val, y_val, shuffle=False)
test_loader = make_loader(X_test, y_test, shuffle=False)


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            logits = model(x_batch)
            total_loss += criterion(logits, y_batch).item() * y_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)
    return total_loss / total, total_correct / total


# ---------------------------------------------------------------------------
# 3. Finite-difference gradient check
# ---------------------------------------------------------------------------
def gradient_check() -> None:
    set_seed(SPLIT_SEED)
    model = SmallCNN().double()
    criterion = nn.CrossEntropyLoss()
    check_loader = make_loader(X_train, y_train, shuffle=False)
    x_batch, y_batch = next(iter(check_loader))
    x_batch = x_batch.double()
    y_batch = y_batch.long()

    model.zero_grad()
    loss = criterion(model(x_batch), y_batch)
    loss.backward()

    param = model.features[0].weight
    grad_auto = param.grad.detach().view(-1).clone()
    n_check = min(8, grad_auto.numel())
    # Use the coordinates with the largest gradient magnitudes: some filters are
    # inactive for this batch, and their exact gradient is zero.
    check_indices = torch.argsort(grad_auto.abs(), descending=True)[:n_check].tolist()
    hs = np.logspace(-8.0, 0.0, 33)
    med_err = []

    for h in hs:
        errs = []
        for idx in check_indices:
            flat = param.data.view(-1)
            original = flat[idx].item()
            flat[idx] = original + h
            with torch.no_grad():
                loss_plus = criterion(model(x_batch), y_batch).item()
            flat[idx] = original - h
            with torch.no_grad():
                loss_minus = criterion(model(x_batch), y_batch).item()
            flat[idx] = original
            fd = (loss_plus - loss_minus) / (2.0 * h)
            denom = max(abs(grad_auto[idx].item()), 1e-12)
            errs.append(abs(fd - grad_auto[idx].item()) / denom)
        med_err.append(float(np.median(errs)))

    med_err = np.asarray(med_err)
    best = int(np.argmin(med_err))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.loglog(hs, med_err, "o-", ms=3, lw=1.4)
    ax.axvline(hs[best], color="crimson", ls="--", lw=1)
    ax.annotate(
        f"min at h={hs[best]:.1e}",
        xy=(hs[best], med_err[best]),
        xytext=(hs[best] * 8, med_err[best] * 6),
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=9,
    )
    ax.set_xlabel("Finite-difference step $h$")
    ax.set_ylabel("Median relative error")
    ax.set_title("Central difference vs autograd")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "梯度有限差分误差.png", bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"h": hs, "median_relative_error": med_err}).to_csv(
        DATA_DIR / "梯度检验数据.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Gradient check done: best h={hs[best]:.3e}, error={med_err[best]:.3e}")


# ---------------------------------------------------------------------------
# 4. CNN training experiments
# ---------------------------------------------------------------------------
def adam_factory(lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    return lambda params: torch.optim.Adam(
        params, lr=lr, betas=(beta1, beta2), eps=eps
    )


CONFIGS = [
    {
        "key": "sgd",
        "label": "SGD (lr=0.02, m=0.9)",
        "group": "baseline",
        "factory": lambda p: torch.optim.SGD(p, lr=0.02, momentum=0.9),
    },
    {
        "key": "adam",
        "label": "Adam (default)",
        "group": "baseline",
        "factory": adam_factory(),
    },
    {
        "key": "adam_lr1e-4",
        "label": "Adam (lr=1e-4)",
        "group": "lr",
        "factory": adam_factory(lr=1e-4),
    },
    {
        "key": "adam_lr1e-2",
        "label": "Adam (lr=1e-2)",
        "group": "lr",
        "factory": adam_factory(lr=1e-2),
    },
    {
        "key": "adam_lr5e-2",
        "label": "Adam (lr=5e-2)",
        "group": "lr",
        "factory": adam_factory(lr=5e-2),
    },
    {
        "key": "adam_lr1e-1",
        "label": "Adam (lr=1e-1)",
        "group": "lr",
        "factory": adam_factory(lr=1e-1),
    },
    {
        "key": "adam_b1_0.5",
        "label": "Adam (beta1=0.5)",
        "group": "beta1",
        "factory": adam_factory(beta1=0.5),
    },
    {
        "key": "adam_b1_0.99",
        "label": "Adam (beta1=0.99)",
        "group": "beta1",
        "factory": adam_factory(beta1=0.99),
    },
    {
        "key": "adam_b2_0.9",
        "label": "Adam (beta2=0.9)",
        "group": "beta2",
        "factory": adam_factory(beta2=0.9),
    },
    {
        "key": "adam_b2_0.9999",
        "label": "Adam (beta2=0.9999)",
        "group": "beta2",
        "factory": adam_factory(beta2=0.9999),
    },
    {
        "key": "adam_eps1e-5",
        "label": "Adam (eps=1e-5)",
        "group": "eps",
        "factory": adam_factory(eps=1e-5),
    },
    {
        "key": "adam_eps1e-3",
        "label": "Adam (eps=1e-3)",
        "group": "eps",
        "factory": adam_factory(eps=1e-3),
    },
]


def train_one(config: dict, seed: int) -> dict:
    set_seed(seed)
    model = SmallCNN().to(DEVICE)
    train_loader = make_loader(X_train, y_train, shuffle=True, generator_seed=seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = config["factory"](model.parameters())

    history = {
        key: np.full(EPOCHS, np.nan)
        for key in ["train_loss", "val_loss", "val_acc", "grad_norm"]
    }
    diverged = False

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = 0.0
        samples = 0
        grad_norm_sum = 0.0
        batches = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()

            squared_norm = 0.0
            grad_finite = True
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    grad_finite = False
                    break
                squared_norm += parameter.grad.detach().pow(2).sum().item()
            if not grad_finite:
                diverged = True
                break

            grad_norm_sum += math.sqrt(squared_norm)
            batches += 1
            optimizer.step()
            loss_sum += loss.item() * y_batch.size(0)
            samples += y_batch.size(0)

            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                diverged = True
                break

        if diverged:
            break

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history["train_loss"][epoch] = loss_sum / samples
        history["val_loss"][epoch] = val_loss
        history["val_acc"][epoch] = val_acc
        history["grad_norm"][epoch] = grad_norm_sum / batches

    if diverged:
        test_loss, test_acc = np.nan, np.nan
    else:
        test_loss, test_acc = evaluate(model, test_loader, criterion)

    return {
        "config": config["key"],
        "label": config["label"],
        "group": config["group"],
        "seed": seed,
        "diverged": diverged,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_grad_norm": history["grad_norm"][-1],
        "history": history,
    }


def plot_group(keys: list[str], metric: str, ylabel: str, filename: str, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for key in keys:
        runs = [r for r in all_runs if r["config"] == key]
        if not runs:
            continue
        stacked = np.vstack([r["history"][metric] for r in runs])
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        epochs = np.arange(1, EPOCHS + 1)
        ax.plot(epochs, mean, marker="o", ms=3, lw=1.5, label=runs[0]["label"])
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.18)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def cnn_experiments() -> None:
    global all_runs
    all_runs = []
    for config in CONFIGS:
        for seed in SEEDS:
            result = train_one(config, seed)
            all_runs.append(result)
            status = "DIVERGED" if result["diverged"] else f"test_acc={result['test_acc']:.4f}"
            print(f"{config['label']:<24} seed={seed} {status}")

    # Long-format history table
    rows = []
    for result in all_runs:
        for epoch in range(EPOCHS):
            rows.append(
                {
                    "config": result["config"],
                    "label": result["label"],
                    "seed": result["seed"],
                    "epoch": epoch + 1,
                    "train_loss": result["history"]["train_loss"][epoch],
                    "val_loss": result["history"]["val_loss"][epoch],
                    "val_acc": result["history"]["val_acc"][epoch],
                    "grad_norm": result["history"]["grad_norm"][epoch],
                }
            )
    histories = pd.DataFrame(rows)
    histories.to_csv(DATA_DIR / "CNN训练历史.csv", index=False, encoding="utf-8-sig")

    # Summary with mean and std across seeds
    summary_rows = []
    for config in CONFIGS:
        runs = [r for r in all_runs if r["config"] == config["key"]]
        finite = [r for r in runs if not r["diverged"]]
        def mean_std(values):
            values = np.asarray(values, dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return np.nan, np.nan
            return float(np.mean(finite_values)), float(np.std(finite_values))
        test_loss_mean, test_loss_std = mean_std([r["test_loss"] for r in finite])
        test_acc_mean, test_acc_std = mean_std([r["test_acc"] for r in finite])
        val_loss_mean, val_loss_std = mean_std([r["final_val_loss"] for r in finite])
        val_acc_mean, val_acc_std = mean_std([r["final_val_acc"] for r in finite])
        grad_mean, grad_std = mean_std([r["final_grad_norm"] for r in finite])
        summary_rows.append(
            {
                "config": config["key"],
                "label": config["label"],
                "group": config["group"],
                "n_finite": len(finite),
                "n_diverged": len(runs) - len(finite),
                "test_loss_mean": test_loss_mean,
                "test_loss_std": test_loss_std,
                "test_acc_mean": test_acc_mean,
                "test_acc_std": test_acc_std,
                "final_val_loss_mean": val_loss_mean,
                "final_val_loss_std": val_loss_std,
                "final_val_acc_mean": val_acc_mean,
                "final_val_acc_std": val_acc_std,
                "final_grad_norm_mean": grad_mean,
                "final_grad_norm_std": grad_std,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "CNN实验汇总.csv", index=False, encoding="utf-8-sig")
    print("\nCNN summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Baseline: loss, accuracy, gradient norm
    baseline_keys = ["sgd", "adam"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for ax, metric, ylabel, title in zip(
        axes,
        ["val_loss", "val_acc", "grad_norm"],
        ["Validation loss", "Validation accuracy", "Mean gradient norm"],
        ["(a) Validation loss", "(b) Validation accuracy", "(c) Gradient norm"],
    ):
        for key in baseline_keys:
            runs = [r for r in all_runs if r["config"] == key]
            stacked = np.vstack([r["history"][metric] for r in runs])
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            epochs = np.arange(1, EPOCHS + 1)
            ax.plot(epochs, mean, marker="o", ms=3, lw=1.5, label=runs[0]["label"])
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.18)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Adam与SGD基线对比.png", bbox_inches="tight")
    plt.close(fig)

    # Learning-rate sweep
    lr_keys = ["adam_lr1e-4", "adam", "adam_lr1e-2", "adam_lr5e-2", "adam_lr1e-1"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for ax, metric, ylabel, title in zip(
        axes,
        ["val_loss", "val_acc"],
        ["Validation loss", "Validation accuracy"],
        ["(a) Validation loss", "(b) Validation accuracy"],
    ):
        for key in lr_keys:
            runs = [r for r in all_runs if r["config"] == key]
            stacked = np.vstack([r["history"][metric] for r in runs])
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            epochs = np.arange(1, EPOCHS + 1)
            ax.plot(epochs, mean, marker="o", ms=3, lw=1.4, label=runs[0]["label"])
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "学习率敏感性.png", bbox_inches="tight")
    plt.close(fig)

    # beta1 and beta2 sweeps
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    for ax, keys, title in zip(
        axes,
        [
            ["adam_b1_0.5", "adam", "adam_b1_0.99"],
            ["adam_b2_0.9", "adam", "adam_b2_0.9999"],
        ],
        ["(a) beta1 sweep", "(b) beta2 sweep"],
    ):
        for key in keys:
            runs = [r for r in all_runs if r["config"] == key]
            stacked = np.vstack([r["history"]["val_loss"] for r in runs])
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            epochs = np.arange(1, EPOCHS + 1)
            ax.plot(epochs, mean, marker="o", ms=3, lw=1.4, label=runs[0]["label"])
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "矩估计系数敏感性.png", bbox_inches="tight")
    plt.close(fig)

    # epsilon sweep
    plot_group(
        ["adam", "adam_eps1e-5", "adam_eps1e-3"],
        "val_loss",
        "Validation loss",
        "稳定项敏感性.png",
    )


if __name__ == "__main__":
    quadratic_experiment()
    gradient_check()
    cnn_experiments()
    print(f"\n图表已保存到 {FIG_DIR.resolve()}")
    print(f"数据表已保存到 {DATA_DIR.resolve()}")
