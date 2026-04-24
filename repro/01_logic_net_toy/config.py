from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    seed: int = 7
    num_labeled: int = 64
    num_unlabeled: int = 512
    num_val: int = 256
    num_test: int = 512
    hidden_dim: int = 32
    batch_size_labeled: int = 32
    batch_size_unlabeled: int = 64
    epochs: int = 140
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    rule_temperature: float = 2.0
    rule_strength: float = 1.25
    rule_set_name: str = "single_good"
    max_distill_weight: float = 0.65
    ramp_up_epochs: int = 40
    mesh_step: float = 0.04
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
            "rule_temperature": self.rule_temperature,
            "rule_strength": self.rule_strength,
            "rule_set_name": self.rule_set_name,
            "max_distill_weight": self.max_distill_weight,
            "ramp_up_epochs": self.ramp_up_epochs,
            "mesh_step": self.mesh_step,
            "device": self.device,
            "experiment_name": self.experiment_name,
            "results_dir": str(self.results_dir) if self.results_dir is not None else None,
        }
