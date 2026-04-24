import copy
import itertools
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader

from constraints import hard_constraint_satisfaction, satisfaction_probability, semantic_loss
from model import TinySemanticMLP

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


def _to_device(batch, device: str):
    return [item.to(device) for item in batch]


def _one_hot_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes=num_classes).float()


def semantic_weight_at(epoch: int, config) -> float:
    ramp = min(1.0, epoch / max(1, config.ramp_up_epochs))
    return config.lambda_semantic * ramp


def evaluate(model, dataset, config, constraint_spec) -> dict:
    model.eval()
    x, y = dataset.tensors
    x = x.to(config.device)
    y = y.to(config.device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = probs.argmax(dim=1)
        targets = _one_hot_targets(y, config.num_classes)

        supervised_loss = F.binary_cross_entropy_with_logits(logits, targets).item()
        constraint_probability = satisfaction_probability(logits, constraint_spec)
        hard_satisfaction = hard_constraint_satisfaction(
            logits,
            constraint_spec,
            threshold=config.constraint_threshold,
        )

    return {
        "loss": supervised_loss,
        "accuracy": (preds == y).float().mean().item(),
        "mean_confidence": probs.max(dim=1).values.mean().item(),
        "mean_satisfaction_probability": constraint_probability.mean().item(),
        "hard_constraint_satisfaction_rate": hard_satisfaction.mean().item(),
        "hard_constraint_violation_rate": 1.0 - hard_satisfaction.mean().item(),
    }


def _make_optimizer(model, config):
    return ManualAdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_baseline(datasets, config, constraint_spec):
    model = TinySemanticMLP(hidden_dim=config.hidden_dim, num_classes=config.num_classes).to(config.device)
    optimizer = _make_optimizer(model, config)
    labeled_loader = DataLoader(datasets.labeled, batch_size=config.batch_size_labeled, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    history = {"train_loss": [], "val_accuracy": [], "val_satisfaction_probability": []}

    for _epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in labeled_loader:
            xb, yb = _to_device(batch, config.device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, _one_hot_targets(yb, config.num_classes))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        val_metrics = evaluate(model, datasets.val, config, constraint_spec)
        history["train_loss"].append(epoch_loss / len(datasets.labeled))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_satisfaction_probability"].append(val_metrics["mean_satisfaction_probability"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def train_semantic_guided(datasets, config, constraint_spec):
    model = TinySemanticMLP(hidden_dim=config.hidden_dim, num_classes=config.num_classes).to(config.device)
    optimizer = _make_optimizer(model, config)
    labeled_loader = DataLoader(datasets.labeled, batch_size=config.batch_size_labeled, shuffle=True)
    unlabeled_loader = DataLoader(datasets.unlabeled, batch_size=config.batch_size_unlabeled, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    history = {
        "total_loss": [],
        "supervised_loss": [],
        "semantic_loss": [],
        "val_accuracy": [],
        "val_satisfaction_probability": [],
    }

    for epoch in range(1, config.epochs + 1):
        model.train()
        lambda_t = semantic_weight_at(epoch, config)
        labeled_iter = itertools.cycle(labeled_loader)
        unlabeled_iter = itertools.cycle(unlabeled_loader)
        steps = max(len(labeled_loader), len(unlabeled_loader))

        total_loss_sum = 0.0
        supervised_loss_sum = 0.0
        semantic_loss_sum = 0.0
        total_seen = 0

        for _ in range(steps):
            xb_l, yb_l = _to_device(next(labeled_iter), config.device)
            xb_u, _ = _to_device(next(unlabeled_iter), config.device)

            optimizer.zero_grad()
            labeled_logits = model(xb_l)
            unlabeled_logits = model(xb_u)

            supervised_term = F.binary_cross_entropy_with_logits(
                labeled_logits,
                _one_hot_targets(yb_l, config.num_classes),
            )
            semantic_term = semantic_loss(unlabeled_logits, constraint_spec)
            total_loss = supervised_term + lambda_t * semantic_term
            total_loss.backward()
            optimizer.step()

            batch_size = xb_l.size(0)
            total_seen += batch_size
            total_loss_sum += total_loss.item() * batch_size
            supervised_loss_sum += supervised_term.item() * batch_size
            semantic_loss_sum += semantic_term.item() * batch_size

        val_metrics = evaluate(model, datasets.val, config, constraint_spec)
        history["total_loss"].append(total_loss_sum / total_seen)
        history["supervised_loss"].append(supervised_loss_sum / total_seen)
        history["semantic_loss"].append(semantic_loss_sum / total_seen)
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_satisfaction_probability"].append(val_metrics["mean_satisfaction_probability"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def save_metrics(metrics: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def plot_training_curves(baseline_history: dict, semantic_history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(baseline_history["train_loss"], label="baseline")
    axes[0].plot(semantic_history["total_loss"], label="semantic total")
    axes[0].plot(semantic_history["supervised_loss"], label="semantic supervised", linestyle="--")
    axes[0].plot(semantic_history["semantic_loss"], label="semantic constraint", linestyle=":")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(baseline_history["val_accuracy"], label="baseline acc")
    axes[1].plot(semantic_history["val_accuracy"], label="semantic acc")
    axes[1].plot(baseline_history["val_satisfaction_probability"], label="baseline sat", linestyle="--")
    axes[1].plot(semantic_history["val_satisfaction_probability"], label="semantic sat", linestyle=":")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_decision_and_constraint_maps(model_a, model_b, datasets, config, constraint_spec, path: Path) -> None:
    x_l, y_l = datasets.labeled.tensors
    x_min, x_max = -2.2, 2.2
    y_min, y_max = -2.2, 2.2

    xs = torch.arange(x_min, x_max, config.mesh_step)
    ys = torch.arange(y_min, y_max, config.mesh_step)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1).to(config.device)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True)
    models = [("Baseline", model_a), ("Semantic-guided", model_b)]
    class_cmap = ListedColormap(["#d95f02", "#1b9e77", "#7570b3", "#e7298a"])

    for col, (title, model) in enumerate(models):
        model.eval()
        with torch.no_grad():
            logits = model(grid)
            probs = torch.sigmoid(logits)
            pred_labels = probs.argmax(dim=1).reshape(grid_x.shape).cpu().numpy()
            sat_prob = satisfaction_probability(logits, constraint_spec).reshape(grid_x.shape).cpu().numpy()

        top_ax = axes[0, col]
        bottom_ax = axes[1, col]

        top_ax.contourf(
            grid_x.numpy(),
            grid_y.numpy(),
            pred_labels,
            levels=np.arange(config.num_classes + 1) - 0.5,
            cmap=class_cmap,
            alpha=0.72,
        )
        top_ax.scatter(x_l[:, 0], x_l[:, 1], c=y_l, cmap=class_cmap, edgecolors="black", s=34)
        top_ax.set_title(f"{title} Decision Regions")
        top_ax.set_xlabel("x1")

        sat_map = bottom_ax.contourf(grid_x.numpy(), grid_y.numpy(), sat_prob, levels=20, cmap="viridis")
        bottom_ax.scatter(x_l[:, 0], x_l[:, 1], c=y_l, cmap=class_cmap, edgecolors="black", s=28)
        bottom_ax.set_title(f"{title} Constraint Probability")
        bottom_ax.set_xlabel("x1")

        fig.colorbar(sat_map, ax=bottom_ax, shrink=0.88, label="P(constraint satisfied)")

    axes[0, 0].set_ylabel("x2")
    axes[1, 0].set_ylabel("x2")
    fig.savefig(path, dpi=180)
    plt.close(fig)
