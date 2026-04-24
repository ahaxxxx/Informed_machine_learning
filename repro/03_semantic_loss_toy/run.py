import argparse
from pathlib import Path

from config import ExperimentConfig
from constraints import available_constraint_sets
from experiment import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic loss toy experiment")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--num-labeled", type=int, default=48)
    parser.add_argument("--num-unlabeled", type=int, default=768)
    parser.add_argument("--lambda-semantic", type=float, default=0.8)
    parser.add_argument(
        "--constraint-set",
        type=str,
        default="exactly_one",
        choices=available_constraint_sets(),
    )
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        num_labeled=args.num_labeled,
        num_unlabeled=args.num_unlabeled,
        lambda_semantic=args.lambda_semantic,
        constraint_set_name=args.constraint_set,
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
    print("Constraint set:", config.constraint_set_name)
    print("Baseline test accuracy:", round(metrics["baseline"]["accuracy"], 4))
    print("Semantic-guided test accuracy:", round(metrics["semantic_guided"]["accuracy"], 4))
    print("Accuracy delta:", round(metrics["delta_accuracy"], 4))
    print(
        "Satisfaction probability delta:",
        round(metrics["delta_satisfaction_probability"], 4),
    )


if __name__ == "__main__":
    main()
