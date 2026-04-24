from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset


@dataclass
class DatasetBundle:
    train_local: TensorDataset
    val_global: tuple[torch.Tensor, torch.Tensor]
    test_global: tuple[torch.Tensor, torch.Tensor]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def true_function(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, 1)
    return (
        0.18 * x.pow(3)
        - 0.82 * x
        + 0.42 * torch.sin(2.4 * x)
        + 0.08 * torch.cos(4.2 * x)
    )


def sample_local_data(num_points: int, low: float, high: float, noise_std: float, generator: torch.Generator):
    x = torch.empty(num_points, 1).uniform_(low, high, generator=generator)
    y_clean = true_function(x)
    noise = noise_std * torch.randn(num_points, 1, generator=generator)
    y_noisy = y_clean + noise
    return x, y_noisy


def make_global_grid(num_points: int, low: float, high: float):
    x = torch.linspace(low, high, num_points).reshape(-1, 1)
    y = true_function(x)
    return x, y


def create_datasets(config) -> DatasetBundle:
    generator = torch.Generator().manual_seed(config.seed)

    x_train, y_train = sample_local_data(
        num_points=config.num_train_local,
        low=config.local_region_low,
        high=config.local_region_high,
        noise_std=config.label_noise_std,
        generator=generator,
    )
    x_val, y_val = make_global_grid(
        num_points=config.num_val_global,
        low=config.domain_low,
        high=config.domain_high,
    )
    x_test, y_test = make_global_grid(
        num_points=config.num_test_global,
        low=config.domain_low,
        high=config.domain_high,
    )

    return DatasetBundle(
        train_local=TensorDataset(x_train, y_train),
        val_global=(x_val, y_val),
        test_global=(x_test, y_test),
    )
