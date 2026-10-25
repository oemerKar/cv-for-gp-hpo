from pathlib import Path
import numpy as np
import json
import gzip
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse


def show_significant(csv_path: str) -> pd.DataFrame:
    """
    Loads a summary_table.csv and returns only the significant values
    with (Experiment, Methods, Median, Std).
    """
    df = pd.read_csv(csv_path)
    
    sig = df[df["Signifikant?"] == "Ja"].copy()

    sig["Winner"] = np.where(sig["Log 10 Median A"] < sig["Log 10 Median B"], sig["Method A"], sig["Method B"])

    sig = sig.sort_values(by="p-value", ascending= True)
    
    cols = [
        "Experiment", 
        "Method A", "Log 10 Median A", "Log 10 Std A",
        "Method B", "Log 10 Median B", "Log 10 Std B",
        "Fit Time Ratio A", "Fit Time Ratio B",
        "p-value", "Winner"
    ]
    
    return sig[cols].reset_index(drop=True)

#Loads a single .json.gz file and extracts the important information
def load_seed_result(filepath: Path) -> dict:
    """
    returns: dict with the structure {'experiment': str, 'seed': int, 
        'methods': {'MLL': {...}, 'LOOCV': {...}, 'MCMC': {...}}}
    Each method contains: regrets, fit_times, best_vals, iter_times, 
                            lengthscales, noises (each in form of a list)
    """
    with gzip.open(filepath) as f:
        data = json.load(f)

    return data

def load_all_seeds(folder: Path)-> dict:
    """
        returns: list of dicts, each representing a seed
    """
    folder = Path(folder)
    results = []

    for filepath in sorted(folder.glob("seed_*.json.gz")):
        results.append(load_seed_result(filepath))

    return results

def extract_metric(seed_results: list[dict], method: str, metric: str) -> np.ndarray:
    """
    extracts given metric out of given method in given list of seed results
    """
    return np.array([r["methods"][method][metric] for r in seed_results])

def plot_all_seeds_with_median(results: dict, title: str, save_path: str, log_scale: bool = True):
    """
    Plots the curves for all seeds (thin and transparently) plus the Median (thick)
    per method. Can also save via save_path, but the save functionality was turned off for 
    purposes of only displaying the plot in the notebook

    Args:
        results: dict such as {"MLL": array(n_seeds, n_iters), "LOOCV": ..., "MCMC": ...}
        title: title of plot
        save_path: save path
        log_scale: y axis logaritghmic?
    """
    colors = {"MLL": "blue", "LOOCV": "orange", "MCMC": "green"}
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, regrets in results.items():
        regrets = np.array(regrets)
        iters = range(1, regrets.shape[1] + 1)
        color = colors.get(name, "gray")

        #Seed Curve
        for seed_curve in regrets:
            ax.plot(iters, seed_curve, color=color, alpha=0.5, linewidth=0.8, zorder=2)

        #Median Curve
        median = np.median(regrets, axis=0)
        ax.plot(iters, median, color=color, linewidth=2, label=name, zorder=5)

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    ax.legend()
    ax.grid(True, alpha=0.3)

    #For saving generated plot

    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #fig.savefig(save_path, bbox_inches="tight")
    #plt.close(fig)

def argument_for_path(path: str, mcmc_included = True, arg = "regrets"):
    """
    returns metric arrays (one for each seed) for the given metric $arg$ for the given path, includes mcmc by default
    """
    folder = Path(path)
    seeds = load_all_seeds(folder)
    if mcmc_included:
        mcmc_regrets = extract_metric(seeds, "MCMC", arg)

    mll_regrets = extract_metric(seeds, "MLL", arg)
    loocv_regrets = extract_metric(seeds, "LOOCV", arg)

    if mcmc_included:
        return {"MLL": mll_regrets,
            "LOOCV": loocv_regrets,
            "MCMC": mcmc_regrets,
        }
    return {"MLL": mll_regrets,
        "LOOCV": loocv_regrets,
    }

def plot_metric_on_ax(ax, seeds: dict, metric: str):
    """
    Plots one metric on given ax 

    Args:
        ax: ax the metric is saved on
        seeds: list of per seed results from load_all_seeds
        metric: the metric that is to be plotted
    """
    colors = {"MLL": "blue", "LOOCV": "orange", "MCMC": "green"}

    mll_metric = extract_metric(seeds, "MLL", metric)
    loocv_metric = extract_metric(seeds, "LOOCV", metric)
    mcmc_metric = extract_metric(seeds, "MCMC", metric)

    results = {"MLL": mll_metric,
        "LOOCV": loocv_metric,
        "MCMC": mcmc_metric,
    }
    
    for name, regrets in results.items():
        regrets = np.array(regrets)
        iters = range(1, regrets.shape[1] + 1)
        color = colors.get(name, "gray")

        #Seed Curve
        for seed_curve in regrets:
            ax.plot(iters, seed_curve, color=color, alpha=0.2, linewidth=0.8, zorder=2)

        #Median Curve
        median = np.median(regrets, axis=0)
        ax.plot(iters, median, color=color, linewidth=2, label=name, zorder=5)


def plot_overview(seeds: list[dict], title: str, save_path: str = None):
    """
    Plots regret, lengthscale and noise together as one plot
    """

    fig, axes = plt.subplots(3,1, figsize = (8,10), sharex = True)

    metrics = [("regrets", "Regret"), ("lengthscales", "Lengthscale"), ("noises", "Noise")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        plot_metric_on_ax(ax= ax, seeds = seeds, metric= metric)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")

    axes[0].set_title(title)
    axes[-1].set_xlabel("Iteration")
    axes[0].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)


def plot_significant_cases(exp_path: str, n_runs: int = 10):
    """
    Reads summary_table.csv of a experiment, filters all significant cases and 
    generates an overview plot for each of those cases
    """

    csv_path = os.path.join(exp_path, "summary_table.csv")
    sig = show_significant(csv_path)

    all_exp_paths = sig["Experiment"].unique()

    for name in all_exp_paths:
        folder = Path(exp_path) / name
        seeds = load_all_seeds(folder)
        save_path = folder/ "plot_overview.pdf"

        plot_overview(seeds, title = name, save_path = save_path)


#AI GENERATED PLOT FUNCTION
def plot_significant_distribution(exp_path: Path, save_path: Path = None, save_name : str = "sig_distr.png"):
    """
    plots a diagramm showing for all methods how many significant wins they have against each other method
    """

    methods = ["MLL", "LOOCV", "MCMC"]

    method_colors = {"MLL": "blue", "LOOCV": "orange", "MCMC": "green"}

    if save_path is None:
        save_path = exp_path

    save_path = save_path / save_name

    csv_path = exp_path/"summary_table.csv"
    df = pd.read_csv(csv_path)
    sig = df[df["Signifikant?"] == "Ja"].copy()
    sig["Winner"] = np.where(sig["Log 10 Median A"] < sig["Log 10 Median B"], sig["Method A"], sig["Method B"])
    sig["Loser"] = np.where(sig["Log 10 Median A"] < sig["Log 10 Median B"], sig["Method B"], sig["Method A"])

    counts = sig.groupby(["Winner", "Loser"]).size().unstack(fill_value=0)

    counts = counts.reindex(index = methods, columns= methods, fill_value= 0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_opponents = 2  # each method has exactly 2 possible methods to be compared to 
    group_width = 0.6
    bar_width = group_width / n_opponents
    x = np.arange(len(methods))

    for i, winner in enumerate(methods):
        opponents = [m for m in methods if m != winner]
        offsets = np.linspace(-group_width/2 + bar_width/2, group_width/2 - bar_width/2, n_opponents)
        for off, opp in zip(offsets, opponents):
            val = counts.loc[winner, opp]
            ax.bar(x[i] + off, val, width=bar_width, color=method_colors[opp])

    # building the legend manually
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=method_colors[m]) for m in methods]
    ax.legend(legend_handles, methods, title="Against")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Significant wins")
    ax.set_xlabel("Winner")
    ax.set_title(f"Significant wins by opponent ({exp_path.name})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str)

    args = parser.parse_args()

    plot_significant_cases(exp_path=f"experiments/{args.experiment}")