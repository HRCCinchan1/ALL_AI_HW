"""Driver that runs every classifier on every dataset for training
fractions 10, 20, ..., 100 and prints / saves mean and std of
prediction error plus mean training time.

The output of this script is what you should use to produce the
learning curve plots in the report (Q4).

Usage:
    python3 q2q3_run_all_stats.py              # all six experiments, default iters
    python3 q2q3_run_all_stats.py -i 3         # 3 iterations per fraction
    python3 q2q3_run_all_stats.py -w perceptron_digits scratch_faces
    python3 q2q3_run_all_stats.py -o results.json

You do not need to edit this file; it imports from the six classifier
modules you wrote.
"""

import argparse
import json
import sys

import numpy as np

# Import each classifier module's main() once it has been implemented.
# If you have not implemented a file yet, comment out the matching line.
from q1a_perceptron_digits import main as perceptron_digits_main
from q1a_perceptron_faces import main as perceptron_faces_main
from q1b_neural_net_scratch_digits import main as scratch_digits_main
from q1b_neural_net_scratch_faces import main as scratch_faces_main
from q1c_neural_net_pytorch_digits import main as pytorch_digits_main
from q1c_neural_net_pytorch_faces import main as pytorch_faces_main


EXPERIMENTS = {
    "perceptron_digits": perceptron_digits_main,
    "perceptron_faces": perceptron_faces_main,
    "scratch_digits": scratch_digits_main,
    "scratch_faces": scratch_faces_main,
    "pytorch_digits": pytorch_digits_main,
    "pytorch_faces": pytorch_faces_main,
}

TRAINING_FRACTIONS = list(range(10, 101, 10))


def run(which: list[str], iters: int) -> dict:
    """Run the selected experiments and return a results dict."""
    results: dict[str, list[dict]] = {}
    for name in which:
        if name not in EXPERIMENTS:
            print(f"[warn] unknown experiment: {name}", file=sys.stderr)
            continue
        print(f"\n########## {name.upper()} ##########")
        results[name] = []
        for pct in TRAINING_FRACTIONS:
            stats = EXPERIMENTS[name](pct, num_iterations=iters)
            results[name].append(stats)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run learning curve experiments.")
    parser.add_argument(
        "-w",
        dest="which",
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        help="Subset of experiments to run (default: all six).",
    )
    parser.add_argument("-i", dest="iters", type=int, default=5,
                        help="Number of random sample repetitions per fraction.")
    parser.add_argument("-o", dest="out", type=str, default=None,
                        help="Optional path to dump JSON results.")
    args = parser.parse_args()

    results = run(args.which, args.iters)

    if args.out:
        with open(args.out, "w") as fp:
            json.dump(results, fp, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
