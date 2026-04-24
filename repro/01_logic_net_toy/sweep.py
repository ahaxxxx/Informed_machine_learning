import argparse
import csv
import itertools
import json
from pathlib import Path

from config import ExperimentConfig
from experiment import run_experiment
from rules import available_rule_sets


def parse_csv_list(raw_value: str, cast_fn):
    return [cast_fn(item.strip()) for item in raw_value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Parameter sweep for logic_net_toy")
    parser.add_argument("--name", type=str, default="default_sweep")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="7,13")
    parser.add_argument("--num-labeled-values", type=str, default="32,64,128")
    parser.add_argument("--rule-strengths", type=str, default="0.5,1.0,1.5")
    parser.add_argument("--distill-weights", type=str, default="0.35,0.65")
    parser.add_argument(
        "--rule-sets",
        type=str,
        default="single_good,single_bad,multi_good,multi_mixed,multi_bad",
        help=f"Comma separated subset of: {', '.join(available_rule_sets())}",
    )
    parser.add_argument("--num-unlabeled", type=int, default=512)
    parser.add_argument("--save-plots", action="store_true")
    return parser.parse_args()


def validate_rule_sets(rule_sets: list[str]) -> list[str]:
    allowed = set(available_rule_sets())
    invalid = [rule_set for rule_set in rule_sets if rule_set not in allowed]
    if invalid:
        raise ValueError(f"Unknown rule sets: {', '.join(invalid)}")
    return rule_sets


def save_summary(rows: list[dict], output_dir: Path) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    seeds = parse_csv_list(args.seeds, int)
    num_labeled_values = parse_csv_list(args.num_labeled_values, int)
    rule_strengths = parse_csv_list(args.rule_strengths, float)
    distill_weights = parse_csv_list(args.distill_weights, float)
    rule_sets = validate_rule_sets(parse_csv_list(args.rule_sets, str))

    sweep_root = Path(__file__).resolve().parent / "results" / "sweeps" / args.name
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows = []
    combinations = list(
        itertools.product(seeds, num_labeled_values, rule_strengths, distill_weights, rule_sets)
    )

    for idx, (seed, num_labeled, rule_strength, distill_weight, rule_set) in enumerate(combinations, start=1):
        experiment_name = (
            f"sweeps/{args.name}/run_{idx:03d}_{rule_set}_n{num_labeled}"
            f"_rs{rule_strength:.2f}_dw{distill_weight:.2f}_s{seed}"
        )
        config = ExperimentConfig(
            seed=seed,
            epochs=args.epochs,
            num_labeled=num_labeled,
            num_unlabeled=args.num_unlabeled,
            rule_strength=rule_strength,
            rule_set_name=rule_set,
            max_distill_weight=distill_weight,
            experiment_name=experiment_name,
            results_dir=Path(__file__).resolve().parent / "results" / experiment_name,
        )

        metrics = run_experiment(config, save_artifacts=True, save_plots=args.save_plots)
        rows.append(
            {
                "run_index": idx,
                "rule_set_name": rule_set,
                "seed": seed,
                "num_labeled": num_labeled,
                "rule_strength": rule_strength,
                "max_distill_weight": distill_weight,
                "baseline_accuracy": metrics["baseline"]["accuracy"],
                "logic_accuracy": metrics["logic_guided"]["accuracy"],
                "delta_accuracy": metrics["delta_accuracy"],
                "baseline_rule_agreement": metrics["baseline"]["prediction_rule_agreement"],
                "logic_rule_agreement": metrics["logic_guided"]["prediction_rule_agreement"],
                "delta_rule_agreement": metrics["delta_rule_agreement"],
                "results_dir": str(config.results_dir),
            }
        )
        print(
            f"[{idx}/{len(combinations)}] {rule_set} n={num_labeled} "
            f"rs={rule_strength:.2f} dw={distill_weight:.2f} seed={seed} "
            f"delta={metrics['delta_accuracy']:.4f}"
        )

    rows.sort(key=lambda item: item["delta_accuracy"], reverse=True)
    save_summary(rows, sweep_root)

    if rows:
        with (sweep_root / "best_run.json").open("w", encoding="utf-8") as f:
            json.dump(rows[0], f, indent=2, ensure_ascii=False)
        print("Best run:", rows[0])
    print("Sweep summary saved to:", sweep_root)


if __name__ == "__main__":
    main()
