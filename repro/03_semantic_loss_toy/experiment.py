from constraints import get_constraint_spec, serialize_constraint_spec
from data import create_datasets, set_seed
from trainer import (
    evaluate,
    plot_decision_and_constraint_maps,
    plot_training_curves,
    save_metrics,
    train_baseline,
    train_semantic_guided,
)


def run_experiment(config, save_artifacts: bool = True, save_plots: bool = True) -> dict:
    config.ensure_results_dir()
    set_seed(config.seed)
    datasets = create_datasets(config)
    constraint_spec = get_constraint_spec(config.constraint_set_name, num_classes=config.num_classes)

    baseline_model, baseline_history = train_baseline(datasets, config, constraint_spec)
    semantic_model, semantic_history = train_semantic_guided(datasets, config, constraint_spec)

    baseline_metrics = evaluate(baseline_model, datasets.test, config, constraint_spec)
    semantic_metrics = evaluate(semantic_model, datasets.test, config, constraint_spec)

    metrics = {
        "config": config.to_dict(),
        "constraint": serialize_constraint_spec(constraint_spec),
        "baseline": baseline_metrics,
        "semantic_guided": semantic_metrics,
        "delta_accuracy": semantic_metrics["accuracy"] - baseline_metrics["accuracy"],
        "delta_satisfaction_probability": (
            semantic_metrics["mean_satisfaction_probability"] - baseline_metrics["mean_satisfaction_probability"]
        ),
        "delta_violation_rate": (
            semantic_metrics["hard_constraint_violation_rate"] - baseline_metrics["hard_constraint_violation_rate"]
        ),
    }

    if save_artifacts:
        save_metrics(metrics, config.results_dir / "metrics.json")
        if save_plots:
            plot_decision_and_constraint_maps(
                model_a=baseline_model,
                model_b=semantic_model,
                datasets=datasets,
                config=config,
                constraint_spec=constraint_spec,
                path=config.results_dir / "decision_and_constraint_maps.png",
            )
            plot_training_curves(
                baseline_history=baseline_history,
                semantic_history=semantic_history,
                path=config.results_dir / "training_curves.png",
            )

    return metrics
