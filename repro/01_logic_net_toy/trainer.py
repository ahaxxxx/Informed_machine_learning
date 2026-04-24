import copy
import itertools
import json
import math
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import TinyMLP
from rules import (
    aggregated_hard_rule_prediction,
    build_teacher_probs,
    hard_rule_prediction_for_rule,
    short_rule_label,
)

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


def distill_weight_at(epoch: int, config) -> float:
    ramp = min(1.0, epoch / max(1, config.ramp_up_epochs))
    return config.max_distill_weight * ramp


def evaluate(model, dataset, config, rule_specs) -> dict:
    model.eval()
    x, y = dataset.tensors
    x = x.to(config.device)
    y = y.to(config.device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

    loss = F.cross_entropy(logits, y).item()
    acc = (preds == y).float().mean().item()
    if not rule_specs:
        return {
            "loss": loss,
            "accuracy": acc,
            "rule_accuracy_on_dataset": None,
            "prediction_rule_agreement": None,
            "per_rule": [],
        }

    aggregated_rule_preds = aggregated_hard_rule_prediction(x, rule_specs, config.rule_temperature)
    rule_acc = (aggregated_rule_preds == y).float().mean().item()
    rule_agreement = (preds == aggregated_rule_preds).float().mean().item()

    per_rule = []
    for rule in rule_specs:
        single_rule_preds = hard_rule_prediction_for_rule(x, rule, config.rule_temperature)
        per_rule.append(
            {
                "name": rule.name,
                "description": rule.description,
                "weight": rule.weight,
                "accuracy_on_dataset": (single_rule_preds == y).float().mean().item(),
                "prediction_agreement": (preds == single_rule_preds).float().mean().item(),
            }
        )

    return {
        "loss": loss,
        "accuracy": acc,
        "rule_accuracy_on_dataset": rule_acc,
        "prediction_rule_agreement": rule_agreement,
        "per_rule": per_rule,
    }


def _make_optimizer(model, config):
    return ManualAdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def train_baseline(datasets, config):
    model = TinyMLP(hidden_dim=config.hidden_dim).to(config.device)
    optimizer = _make_optimizer(model, config)
    labeled_loader = DataLoader(datasets.labeled, batch_size=config.batch_size_labeled, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in labeled_loader:
            xb, yb = _to_device(batch, config.device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(datasets.labeled)
        val_metrics = evaluate(model, datasets.val, config, rule_specs=[])
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def train_logic_guided(datasets, config, rule_specs):
    model = TinyMLP(hidden_dim=config.hidden_dim).to(config.device)
    optimizer = _make_optimizer(model, config)
    labeled_loader = DataLoader(datasets.labeled, batch_size=config.batch_size_labeled, shuffle=True)
    unlabeled_loader = DataLoader(datasets.unlabeled, batch_size=config.batch_size_unlabeled, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    history = {"total_loss": [], "supervised_loss": [], "distill_loss": [], "val_accuracy": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        pi_t = distill_weight_at(epoch, config)
        labeled_iter = itertools.cycle(labeled_loader)
        unlabeled_iter = itertools.cycle(unlabeled_loader)
        steps = max(len(labeled_loader), len(unlabeled_loader))

        total_loss_sum = 0.0
        sup_loss_sum = 0.0
        distill_loss_sum = 0.0
        total_seen = 0

        for _ in range(steps):
            xb_l, yb_l = _to_device(next(labeled_iter), config.device)
            xb_u, _ = _to_device(next(unlabeled_iter), config.device)

            optimizer.zero_grad()

            labeled_logits = model(xb_l)
            supervised_loss = F.cross_entropy(labeled_logits, yb_l)

            x_rule = torch.cat((xb_l, xb_u), dim=0)
            student_logits = model(x_rule)
            with torch.no_grad():
                teacher_probs = build_teacher_probs(
                    student_logits.detach(),
                    x_rule,
                    rule_specs=rule_specs,
                    rule_strength=config.rule_strength,
                    temperature=config.rule_temperature,
                )

            distill_loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                teacher_probs,
                reduction="batchmean",
            )

            loss = (1.0 - pi_t) * supervised_loss + pi_t * distill_loss
            loss.backward()
            optimizer.step()

            batch_size = xb_l.size(0)
            total_seen += batch_size
            total_loss_sum += loss.item() * batch_size
            sup_loss_sum += supervised_loss.item() * batch_size
            distill_loss_sum += distill_loss.item() * batch_size

        val_metrics = evaluate(model, datasets.val, config, rule_specs)
        history["total_loss"].append(total_loss_sum / total_seen)
        history["supervised_loss"].append(sup_loss_sum / total_seen)
        history["distill_loss"].append(distill_loss_sum / total_seen)
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, history


def save_metrics(metrics: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def plot_training_curves(baseline_history: dict, logic_history: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(baseline_history["train_loss"], label="baseline")
    axes[0].plot(logic_history["total_loss"], label="logic total")
    axes[0].plot(logic_history["supervised_loss"], label="logic supervised", linestyle="--")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(baseline_history["val_accuracy"], label="baseline")
    axes[1].plot(logic_history["val_accuracy"], label="logic-guided")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_rule_boundaries(ax, rule_specs, x_min: float, x_max: float) -> None:
    colors = ["gold", "limegreen", "black", "cyan", "orange"]
    for idx, rule in enumerate(rule_specs):
        color = colors[idx % len(colors)]
        a, b = rule.coefficients
        c = rule.bias
        label = short_rule_label(rule)

        if abs(b) > 1e-6:
            xs = torch.tensor([x_min, x_max])
            ys = -(a * xs + c) / b
            ax.plot(xs.numpy(), ys.numpy(), linestyle="--", color=color, linewidth=1.5, label=label)
        elif abs(a) > 1e-6:
            x_line = -c / a
            ax.axvline(x_line, linestyle="--", color=color, linewidth=1.5, label=label)


def plot_decision_boundaries(baseline_model, logic_model, datasets, config, rule_specs, path: Path) -> None:
    x_l, y_l = datasets.labeled.tensors
    x_t, y_t = datasets.test.tensors

    x_min, x_max = -2.2, 2.2
    y_min, y_max = -2.2, 2.2
    xs = torch.arange(x_min, x_max, config.mesh_step)
    ys = torch.arange(y_min, y_max, config.mesh_step)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1).to(config.device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    models = [("Baseline", baseline_model), ("Logic-guided", logic_model)]

    for ax, (title, model) in zip(axes, models):
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(grid), dim=1)[:, 1].reshape(grid_x.shape).cpu().numpy()

        contour = ax.contourf(grid_x.numpy(), grid_y.numpy(), probs, levels=30, cmap="coolwarm", alpha=0.75)
        ax.contour(grid_x.numpy(), grid_y.numpy(), probs, levels=[0.5], colors="black", linewidths=1.2)
        ax.scatter(x_t[:, 0], x_t[:, 1], c=y_t, cmap="coolwarm", s=12, alpha=0.18)
        ax.scatter(x_l[:, 0], x_l[:, 1], c=y_l, cmap="coolwarm", edgecolors="black", s=36)
        _plot_rule_boundaries(ax, rule_specs, x_min, x_max)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("x2")
    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.9, label="P(class=1)")
    fig.savefig(path, dpi=180)
    plt.close(fig)
