from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    seed: int = 11
    num_train_local: int = 32
    num_val_global: int = 301
    num_test_global: int = 601
    hidden_dim: int = 48
    batch_size: int = 24
    epochs: int = 220
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    lambda_data: float = 0.7
    center_pull_weight: float = 0.10
    label_noise_std: float = 0.03
    local_region_low: float = -0.7
    local_region_high: float = 0.7
    domain_low: float = -3.0
    domain_high: float = 3.0
    support_points_per_landmark: int = 40
    landmark_set_name: str = "good"
    device: str = "cpu"
    experiment_name: str = "default"
    results_root: Path = Path(__file__).resolve().parent / "results"
    results_dir: Optional[Path] = None

    def ensure_results_dir(self) -> Path:
        if self.results_dir is None:
            self.results_dir = self.results_root / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "num_train_local": self.num_train_local,
            "num_val_global": self.num_val_global,
            "num_test_global": self.num_test_global,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lambda_data": self.lambda_data,
            "center_pull_weight": self.center_pull_weight,
            "label_noise_std": self.label_noise_std,
            "local_region_low": self.local_region_low,
            "local_region_high": self.local_region_high,
            "domain_low": self.domain_low,
            "domain_high": self.domain_high,
            "support_points_per_landmark": self.support_points_per_landmark,
            "landmark_set_name": self.landmark_set_name,
            "device": self.device,
            "experiment_name": self.experiment_name,
            "results_dir": str(self.results_dir) if self.results_dir is not None else None,
        }
