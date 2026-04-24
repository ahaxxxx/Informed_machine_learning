from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    seed: int = 17
    num_classes: int = 4
    num_labeled: int = 48
    num_unlabeled: int = 768
    num_val: int = 256
    num_test: int = 512
    hidden_dim: int = 48
    batch_size_labeled: int = 24
    batch_size_unlabeled: int = 96
    epochs: int = 160
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    lambda_semantic: float = 0.8
    ramp_up_epochs: int = 40
    constraint_set_name: str = "exactly_one"
    constraint_threshold: float = 0.5
    mesh_step: float = 0.05
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
            "num_classes": self.num_classes,
            "num_labeled": self.num_labeled,
            "num_unlabeled": self.num_unlabeled,
            "num_val": self.num_val,
            "num_test": self.num_test,
            "hidden_dim": self.hidden_dim,
            "batch_size_labeled": self.batch_size_labeled,
            "batch_size_unlabeled": self.batch_size_unlabeled,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "lambda_semantic": self.lambda_semantic,
            "ramp_up_epochs": self.ramp_up_epochs,
            "constraint_set_name": self.constraint_set_name,
            "constraint_threshold": self.constraint_threshold,
            "mesh_step": self.mesh_step,
            "device": self.device,
            "experiment_name": self.experiment_name,
            "results_dir": str(self.results_dir) if self.results_dir is not None else None,
        }
