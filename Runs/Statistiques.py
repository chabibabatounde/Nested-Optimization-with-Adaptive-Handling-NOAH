import json
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================
cllass = 'CEC2017'
Dim = '30'

SPLIT_CEC = True
N_SPLITS = 2

# =========================
# Load JSON
# =========================
INPUT_JSON = os.getcwd() + '/Tests/Optimizations_data/' + cllass + '/Dim_' + Dim + '_Statistics.json'

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

functions = list(data.keys())
algorithms = list(data[functions[0]].keys())

# =========================
# Build mean performance matrix
# =========================
mean_scores = pd.DataFrame(index=functions, columns=algorithms)

for f in functions:
    for algo in algorithms:
        scores = data[f][algo]["scores"]
        mean_scores.loc[f, algo] = np.mean(scores)

mean_scores = mean_scores.astype(float)

# =========================
# ANALYSIS FUNCTION
# =========================
def run_analysis(sub_functions, suffix=""):

    sub_mean_scores = mean_scores.loc[sub_functions]

    # =========================
    # Friedman test
    # =========================
    friedman_stat, friedman_p = stats.friedmanchisquare(
        *[sub_mean_scores[algo] for algo in algorithms]
    )

    print(f"\n[{suffix}] Friedman statistic:", friedman_stat)
    print(f"[{suffix}] p-value:", friedman_p)

    # =========================
    # Compute ranks
    # =========================
    ranks = sub_mean_scores.rank(axis=1, method='average', ascending=True)
    mean_ranks = ranks.mean().sort_values()

    print(f"\n[{suffix}] Mean ranks:")
    print(mean_ranks)

    # =========================
    # Plot mean ranks
    # =========================
    plt.figure(figsize=(10, 5))
    sns.barplot(x=mean_ranks.values, y=mean_ranks.index, palette="viridis")
    plt.title(f"Friedman Mean Ranking {suffix}")
    plt.xlabel("Average Rank (lower is better)")
    plt.ylabel("Algorithms")
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_ranking{suffix}.png'))
    plt.close()

    # =========================
    # Nemenyi test
    # =========================
    nemenyi = sp.posthoc_nemenyi_friedman(sub_mean_scores.values)
    nemenyi.index = algorithms
    nemenyi.columns = algorithms

    print(f"\n[{suffix}] Nemenyi p-values:")
    print(nemenyi)

    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f"Nemenyi Test {suffix}")
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_heatmap{suffix}.png'))
    plt.close()

    # =========================
    # Boxplot
    # =========================
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sub_mean_scores, orient="h")
    plt.title(f"Algorithm Performance {suffix}")
    plt.xlabel("Score")
    plt.ylabel("Algorithms")
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_boxplot{suffix}.png'))
    plt.close()

    # =========================
    # Radar plot
    # =========================
    radar_data = ranks.copy()

    labels = sub_functions
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    linestyles = ['-', '--', '-.', ':']
    palette = sns.color_palette("tab10", len(algorithms))

    for i, algo in enumerate(algorithms):
        values = radar_data[algo].values.tolist()
        values += values[:1]

        marker = None
        linestyle = None
        color = palette[i]
        linewidth = 1

        # Highlight NOAH
        if algo.upper() == 'NOAH':
            marker = 'o'
            linestyle = '-.'
            color = 'red'
            linewidth = 1.5

        ax.plot(
            angles,
            values,
            linewidth=linewidth,
            label=algo,
            marker=marker,
            linestyle=linestyle,
            color=color
        )

        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)

    ax.set_ylim(len(algorithms), 1)

    ax.set_title(f"Radar Plot (Ranks) {suffix}", size=14)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_radar{suffix}.png'))
    plt.close()


# =========================
# SPLIT LOGIC
# =========================
if cllass == 'CEC2020RW' and SPLIT_CEC:
    split_size = int(np.ceil(len(functions) / N_SPLITS))

    for i in range(N_SPLITS):
        sub_functions = functions[i * split_size:(i + 1) * split_size]
        run_analysis(sub_functions, suffix=f"_part{i+1}")
elif cllass == 'CEC2017' and SPLIT_CEC:
    split_size = int(np.ceil(len(functions) / N_SPLITS))
    for i in range(N_SPLITS):
        sub_functions = functions[i * split_size:(i + 1) * split_size]
        run_analysis(sub_functions, suffix=f"_part{i + 1}")
else:
    run_analysis(functions)
