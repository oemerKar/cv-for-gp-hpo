
#After the Experiment run (run_carps_experiments.py) to aggregate the results
#How to use: python gather_results.py --results-dir results --out combined_logs.csv

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from carps.analysis.gather_data_utils import read_trial_log


def gather(results_dir: Path) -> pd.DataFrame:
    """
    Gathers the results at the given path and turns them into a DataFrame which is then saved as a .csv in main

    """
    dfs = []
    skipped = []

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metadata_fn = run_dir / "metadata.json"
        if not metadata_fn.exists():
            skipped.append((run_dir.name, "no metadata.json"))
            continue

        df = read_trial_log(run_dir, log_fn="trial_logs.jsonl")
        if df is None or df.empty:
            skipped.append((run_dir.name, "no/empty trial_logs.jsonl (Run may have never ended)"))
            continue

        with open(metadata_fn) as f:
            meta = json.load(f)

        df["optimizer_id"] = meta["optimizer_id"]
        df["task_id"] = meta["task_id"]
        df["task_category"] = meta["task_category"]
        df["seed"] = meta["seed"]
        df["run_dir"] = run_dir.name

        dfs.append(df)

    if skipped:
        print(f"{len(skipped)} Run(s) skipped:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not dfs:
        raise RuntimeError(f"No runs found in {results_dir}.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n{len(dfs)} Runs read, {len(combined)} Trials in total.")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--out", type=str, default="combined_logs.csv")
    args = parser.parse_args()

    combined = gather(Path(args.results_dir))
    combined.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()