import argparse
from pathlib import Path

from config import ExperimentConfig
from experiment import run_experiment
from rules import available_rule_sets


def parse_args():
    parser = argparse.ArgumentParser(description="Extended logic-net toy experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--num-labeled", type=int, default=64)
    parser.add_argument("--num-unlabeled", type=int, default=512)
    parser.add_argument("--rule-strength", type=float, default=1.25)
    parser.add_argument("--max-distill-weight", type=float, default=0.65)
    parser.add_argument("--rule-set", type=str, default="single_good", choices=available_rule_sets())
    parser.add_argument("--experiment-name", type=str, default="default")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def build_config(args) -> ExperimentConfig:
    config = ExperimentConfig(
        seed=args.seed,
        epochs=args.epochs,
        num_labeled=args.num_labeled,
        num_unlabeled=args.num_unlabeled,
        rule_strength=args.rule_strength,
        rule_set_name=args.rule_set,
        max_distill_weight=args.max_distill_weight,
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
    print("Rule set:", config.rule_set_name)
    print("Baseline test accuracy:", round(metrics["baseline"]["accuracy"], 4))
    print("Logic-guided test accuracy:", round(metrics["logic_guided"]["accuracy"], 4))
    print("Accuracy delta:", round(metrics["delta_accuracy"], 4))


if __name__ == "__main__":
    main()
