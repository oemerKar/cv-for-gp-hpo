"""
Analysis of combined_logs.csv (Output of gather_results.py):

How to use: python carps_analyze.py --combined-logs combined_logs.csv --out autorank_matrix.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import ioh


def get_f_min(task_id: str) -> float | None:
    """
    gets the minimum of the function used, to later compute regret
    """
    match = re.match(r"bbob_f(\d+)_i(\d+)_d(\d+)", task_id)
    if match is None:
        return None
    fid, instance, dim = (int(g) for g in match.groups())
    problem = ioh.get_problem(fid=fid, instance=instance, dimension=dim)
    return problem.optimum.y


def add_regret_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    adds a 'f_min'- and a "regret" col, if there is no known optimum (yahpo/hpobench) 'regret' = 'trial_value__cost'
    """
    df = df.copy()
    f_min_lookup = {tid: get_f_min(tid) for tid in df["task_id"].unique()}
    df["f_min"] = df["task_id"].map(f_min_lookup)
    df["regret"] = np.where(
        df["f_min"].notna(),
        df["trial_value__cost"] - df["f_min"],
        df["trial_value__cost"],
    )
    return df


def compute_cumulative_best(df: pd.DataFrame) -> pd.DataFrame:
    """adds 'cum_best_regret' col, showing the smallest regrets of each (task_id, optimizer_id, seed) run
    """
    df = df.copy()
    df["cum_best_regret"] = (
        df.sort_values("n_trials")
        .groupby(["task_id", "optimizer_id", "seed"])["regret"]
        .cummin()
    )
    return df


def final_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    returns the final best regret for each (task_id, optimizer_id, seed).
    """
    df = add_regret_column(df)
    df = compute_cumulative_best(df)
    final = (
        df.sort_values("n_trials")
        .groupby(["task_id", "optimizer_id", "seed"])["cum_best_regret"]
        .last()
        .reset_index()
    )
    return final


def autorank_matrix(final: pd.DataFrame, agg: str = "median") -> pd.DataFrame:
    """
    Builds matrix for autorank
    """
    agg_per_task = (
        final.groupby(["task_id", "optimizer_id"])["cum_best_regret"]
        .agg(agg)
        .reset_index()
    )
    matrix = agg_per_task.pivot(index="task_id", columns="optimizer_id", values="cum_best_regret")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-logs", type=str, default="combined_logs.csv")
    parser.add_argument("--out", type=str, default="autorank_matrix.csv")
    parser.add_argument("--agg", type=str, default="median", choices=["median", "mean"])
    args = parser.parse_args()

    df = pd.read_csv(args.combined_logs)
    final = final_performance_table(df)
    final.to_csv(Path(args.out).with_name("final_per_run.csv"), index=False)

    matrix = autorank_matrix(final, agg=args.agg)
    matrix.to_csv(args.out)

    print(f"final_per_run.csv saved ({len(final)} Runs)")
    print(f"{args.out} saved (autorank-Matrix, {matrix.shape[0]} Tasks x {matrix.shape[1]} Optimierer)")
    print(matrix)


if __name__ == "__main__":
    main()