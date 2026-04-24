from trainer import (
    evaluate,
    plot_decision_boundaries,
    plot_training_curves,
    save_metrics,
    train_baseline,
    train_logic_guided,
)
from data import create_datasets, set_seed
from rules import get_rule_specs, serialize_rule_specs


def run_experiment(config, save_artifacts: bool = True, save_plots: bool = True) -> dict:
    config.ensure_results_dir()
    set_seed(config.seed)
    datasets = create_datasets(config)
    rule_specs = get_rule_specs(config.rule_set_name)

    baseline_model, baseline_history = train_baseline(datasets, config)
    logic_model, logic_history = train_logic_guided(datasets, config, rule_specs)

    baseline_metrics = evaluate(baseline_model, datasets.test, config, rule_specs)
    logic_metrics = evaluate(logic_model, datasets.test, config, rule_specs)
    metrics = {
        "config": config.to_dict(),
        "rules": serialize_rule_specs(rule_specs),
        "baseline": baseline_metrics,
        "logic_guided": logic_metrics,
        "delta_accuracy": logic_metrics["accuracy"] - baseline_metrics["accuracy"],
        "delta_rule_agreement": (
            logic_metrics["prediction_rule_agreement"] - baseline_metrics["prediction_rule_agreement"]
        ),
    }

    if save_artifacts:
        save_metrics(metrics, config.results_dir / "metrics.json")
        if save_plots:
            plot_decision_boundaries(
                baseline_model=baseline_model,
                logic_model=logic_model,
                datasets=datasets,
                config=config,
                rule_specs=rule_specs,
                path=config.results_dir / "decision_boundaries.png",
            )
            plot_training_curves(
                baseline_history=baseline_history,
                logic_history=logic_history,
                path=config.results_dir / "training_curves.png",
            )

    return metrics
