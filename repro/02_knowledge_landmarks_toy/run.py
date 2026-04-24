import argparse
from pathlib import Path

from config import ExperimentConfig
from experiment import run_experiment
from landmarks import available_landmark_sets


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge landmarks toy experiment")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--num-train-local", type=int, default=32)
    parser.add_argument("--lambda-data", type=float, default=0.7)
    parser.add_argument("--center-pull-weight", type=float, default=0.10)
    parser.add_argument("--landmark-set", type=str, default="good", choices=available_landmark_sets())
    parser.add_argument("--label-noise-std", type=float, default=0.03)
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        num_train_local=args.num_train_local,
        lambda_data=args.lambda_data,
        center_pull_weight=args.center_pull_weight,
        landmark_set_name=args.landmark_set,
        label_noise_std=args.label_noise_std,
        experiment_name=args.experiment_name,
        results_dir=Path(__file__).resolve().parent / "results" / args.experiment_name,
    )
    config.ensure_results_dir()
    return config


def main():
    args = parse_args()
    config = build_config(args)
    metrics = run_experiment(config, save_artifacts=True, save_plots=not args.skip_plots)

    print("Saved outputs to:", config.results_dir)
    print("Landmark set:", config.landmark_set_name)
    print("Baseline full RMSE:", round(metrics["baseline"]["rmse_full"], 4))
    print("Knowledge-guided full RMSE:", round(metrics["knowledge_guided"]["rmse_full"], 4))
    print("Full RMSE improvement:", round(metrics["delta_rmse_full"], 4))


if __name__ == "__main__":
    main()
