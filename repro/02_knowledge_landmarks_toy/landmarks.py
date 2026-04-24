from dataclasses import asdict, dataclass

import torch

from data import true_function


@dataclass(frozen=True)
class Landmark:
    name: str
    x_low: float
    x_high: float
    y_low: float
    y_high: float
    quality: str


def _interval_range(x_low: float, x_high: float, padding: float = 0.0, shift: float = 0.0):
    grid = torch.linspace(x_low, x_high, 160).reshape(-1, 1)
    values = true_function(grid).reshape(-1)
    return float(values.min() - padding + shift), float(values.max() + padding + shift)


def _base_intervals():
    return [
        (-3.0, -2.0),
        (-2.0, -1.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
    ]


def _build_landmarks(intervals, paddings, shifts, quality_labels):
    landmarks = []
    for idx, ((x_low, x_high), padding, shift, quality) in enumerate(
        zip(intervals, paddings, shifts, quality_labels), start=1
    ):
        y_low, y_high = _interval_range(x_low, x_high, padding=padding, shift=shift)
        landmarks.append(
            Landmark(
                name=f"lm_{idx}",
                x_low=x_low,
                x_high=x_high,
                y_low=y_low,
                y_high=y_high,
                quality=quality,
            )
        )
    return landmarks


def available_landmark_sets() -> list[str]:
    return ["coarse_good", "good", "mixed", "shifted_bad"]


def get_landmarks(set_name: str) -> list[Landmark]:
    intervals = _base_intervals()
    if set_name == "good":
        return _build_landmarks(
            intervals=intervals,
            paddings=[0.10] * len(intervals),
            shifts=[0.0] * len(intervals),
            quality_labels=["good"] * len(intervals),
        )
    if set_name == "coarse_good":
        return _build_landmarks(
            intervals=intervals,
            paddings=[0.32] * len(intervals),
            shifts=[0.0] * len(intervals),
            quality_labels=["coarse"] * len(intervals),
        )
    if set_name == "mixed":
        return _build_landmarks(
            intervals=intervals,
            paddings=[0.12, 0.12, 0.12, 0.18, 0.18, 0.18],
            shifts=[0.0, 0.0, 0.0, 0.35, 0.35, 0.35],
            quality_labels=["good", "good", "good", "shifted", "shifted", "shifted"],
        )
    if set_name == "shifted_bad":
        return _build_landmarks(
            intervals=intervals,
            paddings=[0.12] * len(intervals),
            shifts=[0.55] * len(intervals),
            quality_labels=["shifted_bad"] * len(intervals),
        )
    raise ValueError(f"Unknown landmark set '{set_name}'. Available: {', '.join(available_landmark_sets())}")


def serialize_landmarks(landmarks: list[Landmark]) -> list[dict]:
    return [asdict(landmark) for landmark in landmarks]


def sample_landmark_support(landmarks: list[Landmark], points_per_landmark: int, seed: int):
    generator = torch.Generator().manual_seed(seed + 1000)
    xs = []
    y_lows = []
    y_highs = []
    landmark_ids = []

    for idx, landmark in enumerate(landmarks):
        x = torch.empty(points_per_landmark, 1).uniform_(landmark.x_low, landmark.x_high, generator=generator)
        xs.append(x)
        y_lows.append(torch.full((points_per_landmark, 1), landmark.y_low))
        y_highs.append(torch.full((points_per_landmark, 1), landmark.y_high))
        landmark_ids.append(torch.full((points_per_landmark, 1), idx, dtype=torch.long))

    return (
        torch.cat(xs, dim=0),
        torch.cat(y_lows, dim=0),
        torch.cat(y_highs, dim=0),
        torch.cat(landmark_ids, dim=0),
    )
