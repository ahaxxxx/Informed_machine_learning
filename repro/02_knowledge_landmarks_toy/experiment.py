from data import create_datasets, set_seed
from landmarks import get_landmarks, sample_landmark_support, serialize_landmarks
from trainer import (
    evaluate,
    plot_prediction_curves,
    plot_training_curves,
    save_metrics,
    train_baseline,
    train_knowledge_guided,
)


def run_experiment(config, save_artifacts: bool = True, save_plots: bool = True) -> dict:
    config.ensure_results_dir()
    set_seed(config.seed)
    datasets = create_datasets(config)
    landmarks = get_landmarks(config.landmark_set_name)
    support = sample_landmark_support(
        landmarks=landmarks,
        points_per_landmark=config.support_points_per_landmark,
        seed=config.seed,
    )

    baseline_model, baseline_history = train_baseline(datasets, support, config)
    kd_model, kd_history = train_knowledge_guided(datasets, support, config)

    baseline_metrics = evaluate(baseline_model, datasets.test_global[0], datasets.test_global[1], config, support)
    kd_metrics = evaluate(kd_model, datasets.test_global[0], datasets.test_global[1], config, support)

    metrics = {
        "config": config.to_dict(),
        "landmarks": serialize_landmarks(landmarks),
        "baseline": baseline_metrics,
        "knowledge_guided": kd_metrics,
        "delta_rmse_full": baseline_metrics["rmse_full"] - kd_metrics["rmse_full"],
        "delta_rmse_outside_window": (
            baseline_metrics["rmse_outside_window"] - kd_metrics["rmse_outside_window"]
            if baseline_metrics["rmse_outside_window"] is not None and kd_metrics["rmse_outside_window"] is not None
            else None
        ),
        "delta_landmark_violation": baseline_metrics["landmark_violation_mean"] - kd_metrics["landmark_violation_mean"],
        "delta_landmark_penalty": baseline_metrics["landmark_penalty_mean"] - kd_metrics["landmark_penalty_mean"],
    }

    if save_artifacts:
        save_metrics(metrics, config.results_dir / "metrics.json")
        if save_plots:
            plot_prediction_curves(
                baseline_model=baseline_model,
                kd_model=kd_model,
                datasets=datasets,
                landmarks=landmarks,
                config=config,
                path=config.results_dir / "prediction_curves.png",
            )
            plot_training_curves(
                baseline_history=baseline_history,
                kd_history=kd_history,
                path=config.results_dir / "training_curves.png",
            )

    return metrics
