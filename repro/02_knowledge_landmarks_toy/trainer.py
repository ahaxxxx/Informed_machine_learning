import copy
import json
import math
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from matplotlib import patches
from torch.utils.data import DataLoader

from model import TinyRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ManualAdamW:
    def __init__(self, params, lr: float, weight_decay: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = [param for param in params if param.requires_grad]
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.state = {
            id(param): {
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }
            for param in self.params
        }

    def zero_grad(self) -> None:
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self) -> None:
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1 ** self.step_count
        bias_correction2 = 1.0 - self.beta2 ** self.step_count

        with torch.no_grad():
            for param in self.params:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[id(param)]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(self.eps)
                step_size = self.lr / bias_correction1

                if self.weight_decay > 0.0:
                    param.add_(param, alpha=-self.lr * self.weight_decay)

                param.addcdiv_(exp_avg, denom, value=-step_size)


def _make_optimizer(model, config):
    return ManualAdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def interval_violation(pred: torch.Tensor, y_low: torch.Tensor, y_high: torch.Tensor) -> torch.Tensor:
    return F.relu(y_low - pred) + F.relu(pred - y_high)


def knowledge_distance(
    pred: torch.Tensor,
    y_low: torch.Tensor,
    y_high: torch.Tensor,
    center_pull_weight: float,
) -> torch.Tensor:
    violation = interval_violation(pred, y_low, y_high)
    center = 0.5 * (y_low + y_high)
    half_width = 0.5 * (y_high - y_low).clamp_min(1e-6)
    center_pull = ((pred - center) / half_width).pow(2)
    return violation + center_pull_weight * center_pull


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(F.mse_loss(pred, target)).item()


def evaluate(model, x: torch.Tensor, y: torch.Tensor, config, support: tuple[torch.Tensor, ...]) -> dict:
    model.eval()
    x = x.to(config.device)
    y = y.to(config.device)
    support_x, support_low, support_high, _ = support
    support_x = support_x.to(config.device)
    support_low = support_low.to(config.device)
    support_high = support_high.to(config.device)

    with torch.no_grad():
        pred = model(x)
        support_pred = model(support_x)
        violation = interval_violation(support_pred, support_low, support_high)
        knowledge_penalty = knowledge_distance(
            support_pred,
            support_low,
            support_high,
            center_pull_weight=config.center_pull_weight,
        )

    local_mask = (x[:, 0] >= config.local_region_low) & (x[:, 0] <= config.local_region_high)
    extrapolation_mask = ~local_mask

    metrics = {
        "rmse_full": compute_rmse(pred, y),
        "rmse_local_window": compute_rmse(pred[local_mask], y[local_mask]),
        "landmark_violation_mean": violation.mean().item(),
        "landmark_penalty_mean": knowledge_penalty.mean().item(),
        "landmark_inside_rate": (violation <= 1e-6).float().mean().item(),
    }
    if extrapolation_mask.any():
        metrics["rmse_outside_window"] = compute_rmse(pred[extrapolation_mask], y[extrapolation_mask])
    else:
        metrics["rmse_outside_window"] = None
    return metrics


def train_baseline(datasets, support, config):
    model = TinyRegressor(hidden_dim=config.hidden_dim).to(config.device)
    optimizer = _make_optimizer(model, config)
    loader = DataLoader(datasets.train_local, batch_size=config.batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    history = {"train_loss": [], "val_rmse": []}

    for _ in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        total_seen = 0

        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            optimizer.step()
            batch_size = xb.size(0)
            epoch_loss += loss.item() * batch_size
            total_seen += batch_size

        val_metrics = evaluate(model, datasets.val_global[0], datasets.val_global[1], config, support)
        history["train_loss"].append(epoch_loss / total_seen)
        history["val_rmse"].append(val_metrics["rmse_full"])

        if val_metrics["rmse_full"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse_full"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def train_knowledge_guided(datasets, support, config):
    model = TinyRegressor(hidden_dim=config.hidden_dim).to(config.device)
    optimizer = _make_optimizer(model, config)
    loader = DataLoader(datasets.train_local, batch_size=config.batch_size, shuffle=True)

    support_x, support_low, support_high, _ = support
    support_x = support_x.to(config.device)
    support_low = support_low.to(config.device)
    support_high = support_high.to(config.device)

    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    history = {"total_loss": [], "data_loss": [], "knowledge_loss": [], "val_rmse": []}

    for _ in range(config.epochs):
        model.train()
        total_loss_sum = 0.0
        data_loss_sum = 0.0
        knowledge_loss_sum = 0.0
        total_seen = 0

        for xb, yb in loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad()

            pred_local = model(xb)
            data_loss = F.mse_loss(pred_local, yb)

            pred_support = model(support_x)
            knowledge_loss = knowledge_distance(
                pred_support,
                support_low,
                support_high,
                center_pull_weight=config.center_pull_weight,
            ).mean()

            loss = config.lambda_data * data_loss + (1.0 - config.lambda_data) * knowledge_loss
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_seen += batch_size
            total_loss_sum += loss.item() * batch_size
            data_loss_sum += data_loss.item() * batch_size
            knowledge_loss_sum += knowledge_loss.item() * batch_size

        val_metrics = evaluate(model, datasets.val_global[0], datasets.val_global[1], config, support)
        history["total_loss"].append(total_loss_sum / total_seen)
        history["data_loss"].append(data_loss_sum / total_seen)
        history["knowledge_loss"].append(knowledge_loss_sum / total_seen)
        history["val_rmse"].append(val_metrics["rmse_full"])

        if val_metrics["rmse_full"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse_full"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def save_metrics(metrics: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def plot_training_curves(baseline_history: dict, kd_history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(baseline_history["train_loss"], label="baseline")
    axes[0].plot(kd_history["total_loss"], label="kd total")
    axes[0].plot(kd_history["data_loss"], label="kd data", linestyle="--")
    axes[0].plot(kd_history["knowledge_loss"], label="kd knowledge", linestyle=":")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(baseline_history["val_rmse"], label="baseline")
    axes[1].plot(kd_history["val_rmse"], label="knowledge-guided")
    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_landmarks(ax, landmarks):
    for landmark in landmarks:
        width = landmark.x_high - landmark.x_low
        height = landmark.y_high - landmark.y_low
        rect = patches.Rectangle(
            (landmark.x_low, landmark.y_low),
            width,
            height,
            facecolor="none",
            edgecolor="tab:orange" if "good" in landmark.quality or "coarse" in landmark.quality else "tab:red",
            linewidth=1.4,
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(rect)


def plot_prediction_curves(baseline_model, kd_model, datasets, landmarks, config, path: Path) -> None:
    x_test, y_test = datasets.test_global
    x_train, y_train = datasets.train_local.tensors

    x_test_device = x_test.to(config.device)
    with torch.no_grad():
        baseline_pred = baseline_model(x_test_device).cpu()
        kd_pred = kd_model(x_test_device).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True, constrained_layout=True)
    models = [("Baseline", baseline_pred), ("Knowledge-guided", kd_pred)]

    for ax, (title, pred) in zip(axes, models):
        ax.plot(x_test[:, 0], y_test[:, 0], label="true function", color="black", linewidth=2.0)
        ax.plot(x_test[:, 0], pred[:, 0], label="prediction", color="tab:blue", linewidth=2.0)
        ax.scatter(x_train[:, 0], y_train[:, 0], s=30, color="tab:green", edgecolors="black", label="local data")
        ax.axvspan(config.local_region_low, config.local_region_high, color="lightgray", alpha=0.2, label="local window")
        _draw_landmarks(ax, landmarks)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("y")
    fig.savefig(path, dpi=180)
    plt.close(fig)
