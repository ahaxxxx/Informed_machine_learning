from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str
    coefficients: tuple[float, float]
    bias: float = 0.0
    positive_class: int = 1
    weight: float = 1.0
    temperature_scale: float = 1.0


RULE_SETS = {
    "single_good": [
        RuleSpec(
            name="diag_positive",
            description="if x1 > x2, favor class 1",
            coefficients=(1.0, -1.0),
            positive_class=1,
            weight=1.0,
        )
    ],
    "single_bad": [
        RuleSpec(
            name="diag_wrong",
            description="if x1 > x2, favor class 0",
            coefficients=(1.0, -1.0),
            positive_class=0,
            weight=1.0,
        )
    ],
    "multi_good": [
        RuleSpec(
            name="diag_positive",
            description="if x1 > x2, favor class 1",
            coefficients=(1.0, -1.0),
            positive_class=1,
            weight=1.0,
        ),
        RuleSpec(
            name="x1_positive",
            description="if x1 > 0.15, favor class 1",
            coefficients=(1.0, 0.0),
            bias=-0.15,
            positive_class=1,
            weight=0.85,
        ),
        RuleSpec(
            name="x2_small",
            description="if x2 < 0.10, favor class 1",
            coefficients=(0.0, -1.0),
            bias=0.10,
            positive_class=1,
            weight=0.75,
        ),
    ],
    "multi_mixed": [
        RuleSpec(
            name="diag_positive",
            description="if x1 > x2, favor class 1",
            coefficients=(1.0, -1.0),
            positive_class=1,
            weight=1.0,
        ),
        RuleSpec(
            name="x1_positive",
            description="if x1 > 0.15, favor class 1",
            coefficients=(1.0, 0.0),
            bias=-0.15,
            positive_class=1,
            weight=0.85,
        ),
        RuleSpec(
            name="wrong_x2_large",
            description="if x2 > 0.35, favor class 1",
            coefficients=(0.0, 1.0),
            bias=-0.35,
            positive_class=1,
            weight=0.9,
        ),
    ],
    "multi_bad": [
        RuleSpec(
            name="diag_wrong",
            description="if x1 > x2, favor class 0",
            coefficients=(1.0, -1.0),
            positive_class=0,
            weight=1.15,
        ),
        RuleSpec(
            name="x1_wrong",
            description="if x1 > 0.15, favor class 0",
            coefficients=(1.0, 0.0),
            bias=-0.15,
            positive_class=0,
            weight=1.0,
        ),
        RuleSpec(
            name="x2_wrong",
            description="if x2 < 0.10, favor class 0",
            coefficients=(0.0, -1.0),
            bias=0.10,
            positive_class=0,
            weight=0.95,
        ),
    ],
}


def available_rule_sets() -> list[str]:
    return sorted(RULE_SETS.keys())


def get_rule_specs(rule_set_name: str) -> list[RuleSpec]:
    if rule_set_name not in RULE_SETS:
        raise ValueError(f"Unknown rule set '{rule_set_name}'. Available: {', '.join(available_rule_sets())}")
    return list(RULE_SETS[rule_set_name])


def serialize_rule_specs(rule_specs: list[RuleSpec]) -> list[dict]:
    return [asdict(rule) for rule in rule_specs]


def rule_margin(x: torch.Tensor, rule: RuleSpec) -> torch.Tensor:
    return rule.coefficients[0] * x[:, 0] + rule.coefficients[1] * x[:, 1] + rule.bias


def soft_rule_probability_for_rule(x: torch.Tensor, rule: RuleSpec, base_temperature: float) -> torch.Tensor:
    temperature = base_temperature * rule.temperature_scale
    positive_prob = torch.sigmoid(temperature * rule_margin(x, rule))
    if rule.positive_class == 1:
        return positive_prob
    return 1.0 - positive_prob


def rule_distribution(x: torch.Tensor, rule: RuleSpec, base_temperature: float) -> torch.Tensor:
    class1_prob = soft_rule_probability_for_rule(x, rule, base_temperature)
    return torch.stack((1.0 - class1_prob, class1_prob), dim=1)


def hard_rule_prediction_for_rule(x: torch.Tensor, rule: RuleSpec, base_temperature: float) -> torch.Tensor:
    return (soft_rule_probability_for_rule(x, rule, base_temperature) >= 0.5).long()


def aggregate_rule_distribution(x: torch.Tensor, rule_specs: list[RuleSpec], base_temperature: float) -> torch.Tensor:
    if not rule_specs:
        raise ValueError("At least one rule is required to aggregate a rule distribution.")

    log_rule = torch.zeros(x.size(0), 2, device=x.device)
    for rule in rule_specs:
        distribution = rule_distribution(x, rule, base_temperature)
        log_rule = log_rule + rule.weight * torch.log(distribution.clamp_min(1e-6))
    return F.softmax(log_rule, dim=1)


def aggregated_hard_rule_prediction(x: torch.Tensor, rule_specs: list[RuleSpec], base_temperature: float) -> torch.Tensor:
    return aggregate_rule_distribution(x, rule_specs, base_temperature).argmax(dim=1)


def build_teacher_probs(
    student_logits: torch.Tensor,
    x: torch.Tensor,
    rule_specs: list[RuleSpec],
    rule_strength: float,
    temperature: float,
) -> torch.Tensor:
    student_probs = F.softmax(student_logits, dim=1)
    rule_probs = aggregate_rule_distribution(x, rule_specs, temperature)

    log_student = torch.log(student_probs.clamp_min(1e-6))
    log_rule = torch.log(rule_probs.clamp_min(1e-6))
    teacher_logits = log_student + rule_strength * log_rule
    return F.softmax(teacher_logits, dim=1)


def short_rule_label(rule: RuleSpec) -> str:
    class_label = f"c{rule.positive_class}"
    return f"{rule.name}->{class_label}"
