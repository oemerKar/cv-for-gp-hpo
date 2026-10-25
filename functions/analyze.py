"""
Analysis script for experiment results: checks plausability and anomalies
over all Seeds/functions/methods in a experiment folder
"""
import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
import random
from scipy.stats import wilcoxon


def load_seed_result(filepath: Path) -> dict:
    with gzip.open(filepath, 'rt') as f:
        return json.load(f)


def analyze_seed_method(regrets, lengthscales, noises, fit_times, convergence_relative_threshold: float = 0.01) -> dict:
    """ 
    Checks a single (seed, method) combination on anomalies
    These anomalies include: 
    stagnant regret, no improvement, increasing regret (which should be impossible), if there are any nan or inf values, 
    extreme lenghtscale and noise values (lenghtscale > 1000 or < 1e-6; noise > 100 or < 0), if the lenghtscale varies  at all,
    if there is a fit time outlier and whether the optimization converges and if it is near the optimum
    Further metrics for analysis purposes are also provided
    """
    regrets = np.array(regrets, dtype=float)
    lengthscales = np.array(lengthscales, dtype=float)
    noises = np.array(noises, dtype=float)
    fit_times = np.array(fit_times, dtype=float)

    initial_regret = float(regrets[0])
    final_regret = float(regrets[-1])

    #If Regret has sunken by atleast 99% is near optimal is set to true
    convergence_threshold = max(initial_regret * convergence_relative_threshold, 1e-6)
    is_near_optimal = final_regret < convergence_threshold

    #find the iteration at which we count it as converged (regret lower then 1e-3)
    if is_near_optimal:
        close_enough =  np.abs(regrets - final_regret) < convergence_threshold
        convergence_iteration = int(np.argmax(close_enough)) if np.any(close_enough) else None
    else:
        convergence_iteration = None

    flags = {
        "stagnant_regret": bool(np.all(regrets == regrets[0]) and final_regret > convergence_threshold),
        "no_improvement": bool(np.isclose(regrets[0], regrets[-1], atol=1e-9) and final_regret > convergence_threshold),
        "regret_increases": bool(np.any(np.diff(regrets) > 1e-9)),
        "has_nan_inf": bool(
            np.any(~np.isfinite(regrets)) or
            np.any(~np.isfinite(lengthscales)) or
            np.any(~np.isfinite(noises)) or
            np.any(~np.isfinite(fit_times))
        ),
        "lengthscale_extreme": bool(np.any(lengthscales > 1000) or np.any(lengthscales < 1e-6)),
        "noise_extreme": bool(np.any(noises > 100) or np.any(noises < 0)),
        "lengthscale_no_variation": bool(np.ptp(lengthscales) < 1e-8 and len(lengthscales) > 1),
        "fit_time_outlier": bool(
            len(fit_times) > 3 and
            np.max(fit_times) > np.median(fit_times) * 10
        ),
        "final_regret": float(regrets[-1]),
        "median_fit_time": float(np.median(fit_times)),
        "is_near_optimal": is_near_optimal,
        "convergence_iteration": convergence_iteration,
    }
    return flags


def analyze_seed(filepath: Path) -> list[dict]:
    """
    analyzses all methods of a  single seed and returns a list of them
    Return: list of rows filled with data of seed and calculated anomalie
    """
    data = load_seed_result(filepath)
    rows = []
    for method_name, metrics in data["methods"].items():
        flags = analyze_seed_method(
            metrics["regrets"], metrics["lengthscales"],
            metrics["noises"], metrics["fit_times"],
        )
        flags["seed"] = data["seed"]
        flags["method"] = method_name
        flags["file"] = filepath.name
        rows.append(flags)
    return rows


def analyze_folder(folder: Path, expected_seeds: int = 10) -> pd.DataFrame:
    """
    analyses all seeds in in a function folder. 
    Returns: DataFrame with, one row per (seed, method).
    """
    folder = Path(folder)
    seed_files = sorted(folder.glob("seed_*.json.gz"))

    rows = []
    for fp in seed_files:
        rows.extend(analyze_seed(fp))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["folder"] = folder.name
    df["missing_seeds"] = expected_seeds - len(seed_files)

    # cross seed anomalies: is there a seed that is anormal in contrast to the other seeds for this experiment
    for method in df["method"].unique():
        mask = df["method"] == method
        med = df.loc[mask, "final_regret"].median()
        mad = (df.loc[mask, "final_regret"] - med).abs().median()  # Median Absolute Deviation
        threshold = med + 5 * mad if mad > 0 else med * 3
        df.loc[mask, "cross_seed_outlier"] = df.loc[mask, "final_regret"] > threshold

    return df


def find_leaf_folders(experiment_path : Path) -> list[Path]:
    """
    recursively finds the all folders containing seed_*.json.gz files
    """
    leaf_folders = []
    for path in sorted(experiment_path.rglob("seed_*.json.gz")):
        leaf_folders.append(path.parent)
    return sorted(set(leaf_folders))

def analyze_experiment(experiment_path: str, expected_seeds: int = 10) -> pd.DataFrame:
    """
    Calculates Analysis for a full experiment and builds one big table
    """
    experiment_path = Path(experiment_path)
    all_dfs = []

    for folder in find_leaf_folders(experiment_path):
        df = analyze_folder(folder, expected_seeds=expected_seeds)

        if not df.empty:
            # So that names such as d10/Sphere contain the d10 in the name
            df["folder"] = str(folder.relative_to(experiment_path))
            all_dfs.append(df)

    if not all_dfs:
        print(f"No analyzable data in {experiment_path}.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def summarize_issues(df: pd.DataFrame, min_issue_to_print_fraction : float = 0.3) -> pd.DataFrame:
    """
    Summarizes the detailed table of analyze experiment
    """
    issue_cols = [
        "stagnant_regret", "no_improvement", "regret_increases", "has_nan_inf",
        "lengthscale_extreme", "noise_extreme", "lengthscale_no_variation",
        "fit_time_outlier", "cross_seed_outlier",
    ]
    summary = df.groupby(["folder", "method"])[issue_cols].sum()
    summary["n_seeds"] = df.groupby(["folder", "method"]).size()
    summary["any_issue"] = (summary[issue_cols].sum(axis=1) / summary["n_seeds"]) >= min_issue_to_print_fraction

    #average convergence iteration
    conv = df[df["is_near_optimal"]].groupby(["folder", "method"])["convergence_iteration"].median()
    summary["median_convergence_iter"] = conv
    summary["pct_near_optimal"] = df.groupby(["folder", "method"])["is_near_optimal"].mean()

    return summary.reset_index()

def bootstrap_resampling(folder : Path, sample_size : int = 1000) -> dict:
    """
    Samples from the given experiment sample_size times to compute statistical significance of the test results of that folder
    """
    regrets_mll = []
    regrets_loocv = []
    regrets_mcmc = []

    seed_count = 0

    #get the last regret of all seeds for all methods
    for seed in sorted(folder.glob("seed_*.json.gz")):
        data = load_seed_result(seed)
        seed_count += 1

        for method_name, metrics in data["methods"].items():
            if method_name == "MLL":
                regrets_mll.append(metrics["regrets"][-1])
            elif method_name == "LOOCV":
                regrets_loocv.append(metrics["regrets"][-1])
            elif method_name == "MCMC":
                regrets_mcmc.append(metrics["regrets"][-1])

    seed_results_mll = []
    seed_results_loocv = []
    seed_results_mcmc = []

    #draw sample_size many samples of the regret
    for i in range(sample_size):
        indices = [random.randrange(0, seed_count) for _ in range(seed_count)]

        mll_temp = [regrets_mll[idx] for idx in indices]
        loocv_temp = [regrets_loocv[idx] for idx in indices]
        mcmc_temp = [regrets_mcmc[idx] for idx in indices]

        seed_results_mll.append(np.median(mll_temp))
        seed_results_loocv.append(np.median(loocv_temp))
        seed_results_mcmc.append(np.median(mcmc_temp))

    #Change lists to numpy arrays
    seed_results_mll = np.array(seed_results_mll)
    seed_results_loocv = np.array(seed_results_loocv)
    seed_results_mcmc = np.array(seed_results_mcmc)

    #Compute distances between the results
    diff_mll_loocv = seed_results_mll - seed_results_loocv
    diff_mll_mcmc = seed_results_mll - seed_results_mcmc
    diff_loocv_mcmc = seed_results_loocv - seed_results_mcmc

    return {
        "mll": seed_results_mll,
        "loocv": seed_results_loocv,
        "mcmc": seed_results_mcmc,
        "diff_mll_loocv": diff_mll_loocv,
        "diff_mll_mcmc": diff_mll_mcmc,
        "diff_loocv_mcmc": diff_loocv_mcmc,
    }

def summarize_bootstrap(diff: np.array, method_a: str, method_b: str, folder_name: str) -> pd.DataFrame:
    """
    summarizes bootstrap results for a given pair of methods
    """
    lower = np.percentile(diff, 2.5)
    upper = np.percentile(diff, 97.5)

    pct_a_lower = np.mean(diff < 0)

    return {
        "folder": folder_name,
        "method_a": method_a,
        "method_b": method_b,
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "pct_a_better": round(pct_a_lower, 4),
    }


def test_significance(folder: Path, sample_size: int = 1000):
    """
    Tests for the significance of the results using bootstrap and returning a df which covers all 3 
    comparisons with their lower and upper percentile aswell as the percentage in which one of the 
    methods dominates
    """
    results = bootstrap_resampling(folder, sample_size)

    rows = [
        summarize_bootstrap(results["diff_mll_loocv"], "MLL", "LOOCV", folder.name),
        summarize_bootstrap(results["diff_mll_mcmc"], "MLL", "MCMC", folder.name),
        summarize_bootstrap(results["diff_loocv_mcmc"], "LOOCV", "MCMC", folder.name),
    ]
    return pd.DataFrame(rows)


def aggregate(exp_path, n_runs=10):
    """
    Aggregate function copied out of run_experiments, as we dont want to save data here
    It is only used to aggregate the data for a given exp_path
    """
    all_regrets = {}
    all_fit_times = {}
    all_lengthscales = {}
    all_noises = {}

    for seed in range(n_runs):
        path = f"{exp_path}/seed_{seed}.json.gz"
        with gzip.open(path, "rb") as file:
            seed_result = json.load(file)
        for loop_name, metrics in seed_result["methods"].items():
            if loop_name not in all_regrets:
                all_regrets[loop_name] = []
                all_fit_times[loop_name] = []
                all_lengthscales[loop_name] = []
                all_noises[loop_name] = []


            all_regrets[loop_name].append(metrics["regrets"])
            all_fit_times[loop_name].append(metrics["fit_times"])
            all_lengthscales[loop_name].append(metrics["lengthscales"])
            all_noises[loop_name].append(metrics["noises"])

    results = {
        "regrets":   {n: np.array(r) for n, r in all_regrets.items()},
        "fit_times": {n: np.array(t) for n, t in all_fit_times.items()},
        "lengthscales": {n: np.array(t) for n, t in all_lengthscales.items()},
        "noises": {n: np.array(t) for n, t in all_noises.items()},
    }
    return results


def sensitivity_over_iterations(regrets_a, regrets_b, method_a: str, method_b: str, checkpoints=(10, 20, 30, 40, 50)):
    """
    Tests the effect of the amount of iterations done by evaluating the wilcoxon rank sum test after
    the given checkpoints
    """
    rows = []

    for k in checkpoints:
        idx = min(k, regrets_a.shape[-1]) -1
        final_a = regrets_a[:, idx]
        final_b = regrets_b[:, idx]
        stat, p = wilcoxon(final_a, final_b)
        rows.append({
            "methods": f"{method_a} vs. {method_b}",
            "num_loop": k,
            "median_a": np.median(final_a),
            "median_b": np.median(final_b),
            "p_value": p,
            "significant": p < 0.05,
        })
    return pd.DataFrame(rows)

def test_sensitivity_over_iterations(results_path, checkpoints=(10, 20, 30, 40, 50)):
    """
    Tests the effect of the amount of iterations done by evaluating the wilcoxon rank sum test after
    the given checkpoints
    """
    results = aggregate(results_path)

    all_dfs = []
    for method_a, method_b in (("MLL", "LOOCV"), ("MLL", "MCMC"), ("LOOCV", "MCMC")):
        all_dfs.append(sensitivity_over_iterations(results["regrets"][method_a], results["regrets"][method_b], method_a, method_b))

    combined = pd.concat(all_dfs, ignore_index= True)

    return combined

def compute_all_wins(csv_path: str) -> pd.DataFrame:
    """
    Computes the amount of times a Method wins, significantly and non-significantly and returns a 
    table containing the methods and their amount of wins, that are sig. and non-sig. 
    """

    df = pd.read_csv(csv_path)
    df = df.copy()

    #adds winner and loser to each comparison
    df["Winner"] = np.where(
        df["Log 10 Median A"] < df["Log 10 Median B"],
        df["Method A"],
        df["Method B"]
    )
    df["Loser"] = np.where(
        df["Log 10 Median A"] < df["Log 10 Median B"],
        df["Method B"],
        df["Method A"]
    )

    all_wins = df["Winner"].value_counts().rename("all_wins")    
    sig_wins = df[df["Signifikant?"] == "Ja"]["Winner"].value_counts().rename("significant_wins")

    summary = pd.concat([all_wins, sig_wins], axis = 1).fillna(0).astype(int)
    summary["non_significant_wins"] = summary["all_wins"] - summary["significant_wins"]
    
    n_as_a = df["Method A"].value_counts()
    n_as_b = df["Method B"].value_counts()
    summary["total_comparisons"] = n_as_a.add(n_as_b, fill_value=0).astype(int)

    #Add wins by opponent
    wins_by_opponent = df.groupby(["Winner", "Loser"]).size().unstack(fill_value=0)
    wins_by_opponent.columns = [f"wins_vs_{col}" for col in wins_by_opponent.columns]
    summary = summary.join(wins_by_opponent)

    summary["Conv. Rate"] = (summary["significant_wins"] / summary["all_wins"] * 100).round(1)
    
    return summary.sort_values("all_wins", ascending=False)


def compute_all_filtered(csv_path: str, functions):
    """
    filters the csv provided by selecting only the given functions and saving it in the tmp folder
    and than computes compute_all_wins on this filtered data

    should only be called in the notebook as ../tmp is relative to the working directory
    """
    Path("../tmp").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df_filtered = df[df["Experiment"].isin(functions)]
    df_filtered.to_csv("../tmp/all_wins_tmp.csv", index= False)

    return compute_all_wins("../tmp/all_wins_tmp.csv")


def summarize_lengthscale_extreme(df):
    """
    Compares the amount of extreme legnthscales that LOOCV/MLL have to the ones MCMC has
    Is done for Experiment 2 Version 2, to show the lenghtscale degeneration without a prior vs. an uninformative one
    """
    mll_loocv = df[df["method"].isin(["MLL","LOOCV"])]
    mcmc = df[df["method"]=="MCMC"]
    
    return {
        "MLL/LOOCV median seeds affected": mll_loocv["lengthscale_extreme"].median(),
        "MLL/LOOCV mean seeds affected": mll_loocv["lengthscale_extreme"].mean(),
        "MLL/LOOCV % of rows with ≥6/10 affected": (mll_loocv["lengthscale_extreme"] >= 6).mean() * 100,
        "MCMC median seeds affected": mcmc["lengthscale_extreme"].median(),
        "MCMC mean seeds affected": mcmc["lengthscale_extreme"].mean(),
        "MCMC % of rows with 0/10 affected": (mcmc["lengthscale_extreme"] == 0).mean() * 100,
    }
    
