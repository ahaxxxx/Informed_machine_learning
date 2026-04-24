from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset


@dataclass
class DatasetBundle:
    labeled: TensorDataset
    unlabeled: TensorDataset
    val: TensorDataset
    test: TensorDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def class_scores(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    s0 = 1.10 * x1 + 0.25 * x2 - 0.18 * x1.square() + 0.20 * torch.sin(1.8 * x2)
    s1 = -0.90 * x1 + 0.95 * x2 + 0.28 * torch.sin(2.0 * x1)
    s2 = -0.75 * x1 - 1.10 * x2 + 0.22 * torch.cos(2.2 * x2)
    s3 = 0.85 * x1 - 0.80 * x2 + 0.32 * torch.sin(1.6 * (x1 - x2))
    return torch.stack((s0, s1, s2, s3), dim=1)


def make_labels(x: torch.Tensor) -> torch.Tensor:
    return class_scores(x).argmax(dim=1)


def _sample_candidate_pool(num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty(num_points, 2).uniform_(-2.0, 2.0, generator=generator)
    y = make_labels(x)
    return x, y


def make_balanced_dataset(
    num_points: int,
    num_classes: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    desired_counts = [num_points // num_classes for _ in range(num_classes)]
    for idx in range(num_points % num_classes):
        desired_counts[idx] += 1

    xs = []
    ys = []
    current_counts = [0 for _ in range(num_classes)]

    while any(current < target for current, target in zip(current_counts, desired_counts)):
        x_pool, y_pool = _sample_candidate_pool(max(6 * num_points, 512), generator)

        for class_idx in range(num_classes):
            remaining = desired_counts[class_idx] - current_counts[class_idx]
            if remaining <= 0:
                continue

            class_mask = y_pool == class_idx
            available = int(class_mask.sum().item())
            take = min(remaining, available)
            if take <= 0:
                continue

            xs.append(x_pool[class_mask][:take])
            ys.append(y_pool[class_mask][:take])
            current_counts[class_idx] += take

    x = torch.cat(xs, dim=0)[:num_points]
    y = torch.cat(ys, dim=0)[:num_points]
    order = torch.randperm(num_points, generator=generator)
    return x[order], y[order]


def create_datasets(config) -> DatasetBundle:
    generator = torch.Generator().manual_seed(config.seed)

    x_l, y_l = make_balanced_dataset(config.num_labeled, config.num_classes, generator)
    x_u, y_u = make_balanced_dataset(config.num_unlabeled, config.num_classes, generator)
    x_v, y_v = make_balanced_dataset(config.num_val, config.num_classes, generator)
    x_t, y_t = make_balanced_dataset(config.num_test, config.num_classes, generator)

    return DatasetBundle(
        labeled=TensorDataset(x_l, y_l),
        unlabeled=TensorDataset(x_u, y_u),
        val=TensorDataset(x_v, y_v),
        test=TensorDataset(x_t, y_t),
    )
