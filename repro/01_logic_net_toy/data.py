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


def decision_score(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    return 1.1 * x1 - 0.55 * x2 + 0.35 * torch.sin(2.6 * x2) - 0.12 * x1.square()


def make_labels(x: torch.Tensor) -> torch.Tensor:
    return (decision_score(x) > 0.0).long()


def _sample_candidate_pool(num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.empty(num_points, 2).uniform_(-2.0, 2.0, generator=generator)
    y = make_labels(x)
    return x, y


def make_balanced_dataset(num_points: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    half = num_points // 2
    needed_pos = half
    needed_neg = num_points - half

    xs = []
    ys = []
    current_pos = 0
    current_neg = 0

    while current_pos < needed_pos or current_neg < needed_neg:
        x_pool, y_pool = _sample_candidate_pool(max(4 * num_points, 512), generator)

        pos_mask = y_pool == 1
        neg_mask = ~pos_mask

        if current_pos < needed_pos:
            take_pos = min(needed_pos - current_pos, int(pos_mask.sum().item()))
            if take_pos > 0:
                xs.append(x_pool[pos_mask][:take_pos])
                ys.append(y_pool[pos_mask][:take_pos])
                current_pos += take_pos

        if current_neg < needed_neg:
            take_neg = min(needed_neg - current_neg, int(neg_mask.sum().item()))
            if take_neg > 0:
                xs.append(x_pool[neg_mask][:take_neg])
                ys.append(y_pool[neg_mask][:take_neg])
                current_neg += take_neg

    x = torch.cat(xs, dim=0)[:num_points]
    y = torch.cat(ys, dim=0)[:num_points]
    shuffle = torch.randperm(num_points, generator=generator)
    return x[shuffle], y[shuffle]


def create_datasets(config) -> DatasetBundle:
    generator = torch.Generator().manual_seed(config.seed)

    x_l, y_l = make_balanced_dataset(config.num_labeled, generator)
    x_u, y_u = make_balanced_dataset(config.num_unlabeled, generator)
    x_v, y_v = make_balanced_dataset(config.num_val, generator)
    x_t, y_t = make_balanced_dataset(config.num_test, generator)

    return DatasetBundle(
        labeled=TensorDataset(x_l, y_l),
        unlabeled=TensorDataset(x_u, y_u),
        val=TensorDataset(x_v, y_v),
        test=TensorDataset(x_t, y_t),
    )
