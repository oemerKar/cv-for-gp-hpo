import matplotlib.pyplot as plt
import numpy as np
import os

#AI GENERATED PLOT FUNCTION
def plot_bo_comparison(results, title="BO Comparison", labels=None, colors=None):
    """
    Plots a comparison between the methods in the given results, regarding regrets, best_vals, fit times, iteration times
    lenghtscales and noises
    """
    default_colors = ["blue", "orange", "green", "red"]
    
    if isinstance(results, dict):
        # {label: result_dict} übergeben
        items = list(results.items())
    else:
        # Liste von result_dicts übergeben, labels separat
        items = list(zip(labels or [f"Method {i+1}" for i in range(len(results))], results))

    colors = colors or default_colors

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    iterations = range(1, len(items[0][1]["regrets"]) + 1)

    plots = [
        ("regrets",      "Regret (log scale)",  "Iteration", "Regret"),
        ("best_vals",    "Best Value",           "Iteration", "Best Value"),
        ("fit_times",    "Fit Time (s)",         "Iteration", "Time (s)"),
        ("iter_times",   "Iteration Time (s)",   "Iteration", "Time (s)"),
        ("lengthscales", "Lengthscale (mean)",   "Iteration", "Lengthscale"),
        ("noises",       "Noise",                "Iteration", "Noise"),
    ]

    for ax, (key, title_sub, xlabel, ylabel) in zip(axes.flat, plots):
        for (name, result), color in zip(items, colors):
            ax.plot(iterations, result[key], label=name, color=color)
        if key == "regrets":
            ax.set_yscale("log")
        ax.set_title(title_sub)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_regret_with_std(results, title):
    """
    Only plots the regret plots of all methods in the results in one plot 
    """
    colors = ["blue", "orange", "green", "red"]
    iters = range(1, len(next(iter(results.values()))[0]) + 1)

    plt.figure(figsize=(8, 5))

    for (name, regrets), color in zip(results.items(), colors):
        regrets_array = np.array(regrets)
        mean = regrets_array.mean(axis=0)
        std  = regrets_array.std(axis=0)

        # Varianz nicht ins Negative
        lower = np.maximum(mean - std, 0)

        plt.plot(iters, mean, label=name, color=color)
        plt.fill_between(iters, lower, mean + std, alpha=0.2, color=color)

    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def save_regret_with_std(results, title, save_path):
    """
        Makes and saves the regret plots of all methods in the results in one plot at given save_path
    """
    colors = ["blue", "orange", "green", "red"]
    iters = range(1, len(next(iter(results.values()))[0]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, regrets), color in zip(results.items(), colors):
        regrets_array = np.array(regrets)
        mean = regrets_array.mean(axis=0)
        std  = regrets_array.std(axis=0)

        # Varianz nicht ins Negative
        lower = np.maximum(mean - std, 0)

        ax.plot(iters, mean, label=name, color=color)
        ax.fill_between(iters, lower, mean + std, alpha=0.2, color=color)

    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok= True)
    fig.savefig(save_path, bbox_inches = "tight")
    plt.close(fig)