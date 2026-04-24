from dataclasses import dataclass
import itertools

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    description: str
    num_classes: int
    valid_assignments: torch.Tensor


def _all_binary_assignments(num_classes: int) -> torch.Tensor:
    assignments = list(itertools.product([0.0, 1.0], repeat=num_classes))
    return torch.tensor(assignments, dtype=torch.float32)


def available_constraint_sets() -> list[str]:
    return ["exactly_one", "at_least_one", "exactly_two_bad"]


def get_constraint_spec(name: str, num_classes: int = 4) -> ConstraintSpec:
    assignments = _all_binary_assignments(num_classes)
    counts = assignments.sum(dim=1)

    if name == "exactly_one":
        mask = counts == 1
        description = "Exactly one class-indicator should be active."
    elif name == "at_least_one":
        mask = counts >= 1
        description = "At least one class-indicator should be active."
    elif name == "exactly_two_bad":
        mask = counts == 2
        description = "Exactly two class-indicators should be active. This is intentionally wrong."
    else:
        raise ValueError(f"Unknown constraint set: {name}")

    return ConstraintSpec(
        name=name,
        description=description,
        num_classes=num_classes,
        valid_assignments=assignments[mask],
    )


def log_satisfaction_mass(logits: torch.Tensor, spec: ConstraintSpec) -> torch.Tensor:
    assignments = spec.valid_assignments.to(device=logits.device, dtype=logits.dtype)
    log_prob_one = F.logsigmoid(logits).unsqueeze(1)
    log_prob_zero = F.logsigmoid(-logits).unsqueeze(1)
    log_terms = assignments.unsqueeze(0) * log_prob_one + (1.0 - assignments).unsqueeze(0) * log_prob_zero
    return torch.logsumexp(log_terms.sum(dim=2), dim=1)


def satisfaction_probability(logits: torch.Tensor, spec: ConstraintSpec) -> torch.Tensor:
    return log_satisfaction_mass(logits, spec).exp()


def semantic_loss(logits: torch.Tensor, spec: ConstraintSpec) -> torch.Tensor:
    return -log_satisfaction_mass(logits, spec).mean()


def hard_constraint_satisfaction(
    logits: torch.Tensor,
    spec: ConstraintSpec,
    threshold: float = 0.5,
) -> torch.Tensor:
    assignments = spec.valid_assignments.to(device=logits.device, dtype=torch.int64)
    predicted_bits = (torch.sigmoid(logits) >= threshold).to(torch.int64)
    matches = (predicted_bits.unsqueeze(1) == assignments.unsqueeze(0)).all(dim=2)
    return matches.any(dim=1).float()


def serialize_constraint_spec(spec: ConstraintSpec) -> dict:
    return {
        "name": spec.name,
        "description": spec.description,
        "num_classes": spec.num_classes,
        "num_valid_assignments": int(spec.valid_assignments.shape[0]),
        "valid_assignments": spec.valid_assignments.int().tolist(),
    }
