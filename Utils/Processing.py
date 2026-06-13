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
cllass = 'CEC2020RW'
Dim = 'X'

SPLIT_CEC = True
N_SPLITS = 2

# =========================
# Load JSON
# =========================
INPUT_JSON = os.getcwd() + '/Optimizations_data/Optimizations_data/' + cllass + '/Dim_' + Dim + '_Statistics.json'

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

functions = list(data.keys())
algorithms_raw = list(data[functions[0]].keys())

# =========================
# Ordre d'affichage imposé
# =========================
DISPLAY_ORDER = ["NOAH", "PSO", "GA", "SADE", "GWO", "WarSO",
                 "WOA", "DOA", "NIADE", "SOA", "CCCO", "DBO", "BKA"]

# Réordonner selon DISPLAY_ORDER (insensible à la casse),
# en conservant uniquement les algorithmes réellement présents
def reorder_algorithms(raw_algos, desired_order):
    # mapping casse-insensible : NOM_UPPER -> nom réel dans les données
    lookup = {a.upper(): a for a in raw_algos}
    ordered = [lookup[name.upper()] for name in desired_order
               if name.upper() in lookup]
    # ajouter à la fin tout algo présent dans les données mais absent de l'ordre
    remaining = [a for a in raw_algos if a not in ordered]
    if remaining:
        print(f"[WARN] Algorithmes hors DISPLAY_ORDER ajoutés à la fin : {remaining}")
    return ordered + remaining

algorithms = reorder_algorithms(algorithms_raw, DISPLAY_ORDER)

print("Ordre final des algorithmes :", algorithms)


# =========================
# Build mean performance matrix
# =========================
mean_scores = pd.DataFrame(index=functions, columns=algorithms)

for f in functions:
    for algo in algorithms:
        scores = data[f][algo]["scores"]
        mean_scores.loc[f, algo] = np.mean(scores)

mean_scores = mean_scores.astype(float)


def fmt_pm(mean, std):
    return f"{mean:.2e} $\\pm$ {std:.2e}"


def esc(s):
    return s.replace('_', r'\_')


def generate_results_table(sub_functions, suffix=""):
    n_algos = len(algorithms)
    col_spec = "cl" + "c" * n_algos
    algo_headers = " & ".join([esc(a) for a in algorithms])
    caption = f"Results on {cllass} ($D={Dim}$) {suffix.replace('_', ' ')}"
    label = f"tab:results_{cllass}_{Dim}{suffix}"

    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(rf"\begin{{longtable}}{{{col_spec}}}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}} \\")
    lines.append(r"\toprule")
    lines.append(rf"Function & Metric & {algo_headers} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(rf"Function & Metric & {algo_headers} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for idx, f in enumerate(sub_functions, start=1):
        mins, means, stds = {}, {}, {}
        for algo in algorithms:
            scores = np.array(data[f][algo]["scores"], dtype=float)
            mins[algo] = np.min(scores)
            means[algo] = np.mean(scores)
            stds[algo] = np.std(scores)

        best_min = min(mins.values())
        best_mean = min(means.values())

        min_row = []
        for algo in algorithms:
            cell = f"{mins[algo]:.2e}"
            if np.isclose(mins[algo], best_min):
                cell = rf"\textbf{{{cell}}}"
            min_row.append(cell)
        lines.append(rf"\multirow{{3}}{{*}}{{$F_{{{idx}}}$}} & Min & " + " & ".join(min_row) + r" \\")

        mean_row = []
        for algo in algorithms:
            cell = f"{means[algo]:.2e}"
            if np.isclose(means[algo], best_mean):
                cell = rf"\textbf{{{cell}}}"
            mean_row.append(cell)
        lines.append(r" & Mean & " + " & ".join(mean_row) + r" \\")

        std_row = [f"{stds[algo]:.2e}" for algo in algorithms]
        lines.append(r" & Std & " + " & ".join(std_row) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\end{center}")

    out_path = INPUT_JSON.replace('.json', f'_results_table{suffix}.tex')
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"[{suffix}] LaTeX results longtable saved: {out_path}")


def generate_friedman_table(mean_ranks, friedman_stat, friedman_p, suffix=""):
    caption = (f"Friedman mean ranking on {cllass} ($D={Dim}$) "
               f"{suffix.replace('_', ' ')}. "
               f"Statistic $={friedman_stat:.4f}$, $p={friedman_p:.4e}$.")
    label = f"tab:friedman_{cllass}_{Dim}{suffix}"

    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(r"\begin{longtable}{clc}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}} \\")
    lines.append(r"\toprule")
    lines.append(r"Rank & Algorithm & Mean Rank \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Rank & Algorithm & Mean Rank \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    best_algo = mean_ranks.index[0]
    for pos, (algo, rank) in enumerate(mean_ranks.items(), start=1):
        algo_disp = esc(algo)
        if algo == best_algo:
            lines.append(rf"{pos} & \textbf{{{algo_disp}}} & \textbf{{{rank:.4f}}} \\")
        else:
            lines.append(rf"{pos} & {algo_disp} & {rank:.4f} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\end{center}")

    out_path = INPUT_JSON.replace('.json', f'_friedman_table{suffix}.tex')
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"[{suffix}] LaTeX Friedman longtable saved: {out_path}")


def generate_wilcoxon_table(sub_functions, suffix="", reference="NOAH"):
    ref_name = None
    for a in algorithms:
        if a.upper() == reference.upper():
            ref_name = a
            break
    if ref_name is None:
        print(f"[{suffix}] Référence '{reference}' introuvable, Wilcoxon ignoré.")
        return

    others = [a for a in algorithms if a != ref_name]
    alpha = 0.05

    results = pd.DataFrame(index=sub_functions, columns=others)
    counts = {a: {"+": 0, "=": 0, "-": 0} for a in others}

    for f in sub_functions:
        ref_scores = np.array(data[f][ref_name]["scores"], dtype=float)
        for algo in others:
            algo_scores = np.array(data[f][algo]["scores"], dtype=float)
            try:
                stat, p = stats.wilcoxon(ref_scores, algo_scores)
            except ValueError:
                p = 1.0

            if p < alpha:
                sign = "+" if np.median(ref_scores) < np.median(algo_scores) else "-"
            else:
                sign = "="

            results.loc[f, algo] = sign
            counts[algo][sign] += 1

    col_spec = "c" * (len(others) + 1)
    algo_headers = " & ".join([esc(a) for a in others])
    ref_disp = esc(ref_name)
    caption = (f"Wilcoxon signed-rank test ({ref_disp} vs others) on "
               f"{cllass} ($D={Dim}$) {suffix.replace('_', ' ')}")
    label = f"tab:wilcoxon_{cllass}_{Dim}{suffix}"

    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(rf"\begin{{longtable}}{{{col_spec}}}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}} \\")
    lines.append(r"\toprule")
    lines.append(rf"Function & {algo_headers} \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(rf"Function & {algo_headers} \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for idx, f in enumerate(sub_functions, start=1):
        row = [rf"$F_{{{idx}}}$"] + [str(results.loc[f, a]) for a in others]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\midrule")
    overall_cells = ["Overall (+/=/-)"]
    for a in others:
        c = counts[a]
        overall_cells.append(f"{c['+']}/{c['=']}/{c['-']}")
    lines.append(" & ".join(overall_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\end{center}")

    out_path = INPUT_JSON.replace('.json', f'_wilcoxon_table{suffix}.tex')
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"[{suffix}] LaTeX Wilcoxon longtable saved: {out_path}")


def generate_nemenyi_table(nemenyi, mean_ranks, sub_functions, friedman_stat, friedman_p, suffix="", reference="NOAH"):
    """
    Generate LaTeX table for Nemenyi post-hoc test with pairwise differences.
    Shows only reference algorithm vs others comparisons.
    """
    # q_alpha values for alpha=0.05 (studentized range critical values / sqrt(2))
    q_alpha_table = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.314, 14: 3.356, 15: 3.394, 16: 3.429,
        17: 3.461, 18: 3.491, 19: 3.519, 20: 3.544
    }

    k = len(algorithms)  # number of algorithms
    N = len(sub_functions)  # number of functions

    # Get q_alpha for k algorithms
    q_alpha = q_alpha_table.get(k, 2.576)  # default to 2.576 if k > 20

    # Calculate Critical Difference
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

    # Find reference algorithm
    ref_name = None
    for a in algorithms:
        if a.upper() == reference.upper():
            ref_name = a
            break

    if ref_name is None:
        print(f"[{suffix}] Référence '{reference}' introuvable, Nemenyi ignoré.")
        return

    ref_rank = mean_ranks[ref_name]
    ref_disp = esc(ref_name)

    # Build comparison rows
    comparisons = []
    for algo in algorithms:
        if algo == ref_name:
            continue

        algo_rank = mean_ranks[algo]
        diff = abs(algo_rank - ref_rank)
        is_significant = diff > cd

        comp_name = f"{ref_disp} vs {esc(algo)}"
        diff_str = f"{diff:.4f}"
        interp = r"\textbf{Significatif}" if is_significant else "Non significatif"

        comparisons.append({
            'comparison': comp_name,
            'difference': diff_str,
            'interpretation': interp,
            'significant': is_significant
        })

    # Build LaTeX table
    caption = (f"Nemenyi post-hoc test ({ref_disp} vs others) on "
               f"{cllass} ($D={Dim}$) {suffix.replace('_', ' ')}. "
               f"CD = {cd:.4f}, Friedman $={friedman_stat:.4f}$, $p={friedman_p:.4e}$.")
    label = f"tab:nemenyi_{cllass}_{Dim}{suffix}"

    lines = []
    lines.append(r"\begin{center}")
    lines.append(r"\small")
    lines.append(r"\renewcommand{\arraystretch}{1.25}")
    lines.append(r"\begin{longtable}{lcc}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}} \\")
    lines.append(r"\toprule")
    lines.append(r"Algorithms & Difference & Interprétation \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Algorithms & Difference & Interprétation \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    # Add comparison rows
    for comp in comparisons:
        if comp['significant']:
            lines.append(rf"{comp['comparison']} & \textbf{{{comp['difference']}}} & {comp['interpretation']} \\")
        else:
            lines.append(rf"{comp['comparison']} & {comp['difference']} & {comp['interpretation']} \\")

    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{3}}{{l}}{{\textit*{{Note:}} CD = $q_\alpha \sqrt{{k(k+1)/(6N)}}$ = {cd:.4f} (k={k}, N={N})}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\end{center}")

    out_path = INPUT_JSON.replace('.json', f'_nemenyi_table{suffix}.tex')
    with open(out_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"[{suffix}] LaTeX Nemenyi longtable saved: {out_path}")


def run_analysis(sub_functions, suffix=""):
    sub_mean_scores = mean_scores.loc[sub_functions]

    friedman_stat, friedman_p = stats.friedmanchisquare(
        *[sub_mean_scores[algo] for algo in algorithms]
    )

    print(f"\n[{suffix}] Friedman statistic:", friedman_stat)
    print(f"[{suffix}] p-value:", friedman_p)

    ranks = sub_mean_scores.rank(axis=1, method='average', ascending=True)
    mean_ranks = ranks.mean().sort_values()

    print(f"\n[{suffix}] Mean ranks:")
    print(mean_ranks)

    generate_results_table(sub_functions, suffix=suffix)
    generate_friedman_table(mean_ranks, friedman_stat, friedman_p, suffix=suffix)
    generate_wilcoxon_table(sub_functions, suffix=suffix, reference="NOAH")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=mean_ranks.values, y=mean_ranks.index, palette="viridis")
    plt.title(f"Friedman Mean Ranking {suffix}")
    plt.xlabel("Average Rank (lower is better)")
    plt.ylabel("Algorithms")
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_ranking{suffix}.png'))
    plt.close()

    nemenyi = sp.posthoc_nemenyi_friedman(sub_mean_scores.values)
    nemenyi.index = algorithms
    nemenyi.columns = algorithms

    print(f"\n[{suffix}] Nemenyi p-values:")
    print(nemenyi)

    # Generate Nemenyi table
    generate_nemenyi_table(nemenyi, mean_ranks, sub_functions, friedman_stat, friedman_p, suffix=suffix,
                           reference="NOAH")

    plt.figure(figsize=(10, 8))
    sns.heatmap(nemenyi, annot=True, cmap="coolwarm", fmt=".3f")
    plt.title(f"Nemenyi Test {suffix}")
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_heatmap{suffix}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=sub_mean_scores, orient="h")
    plt.title(f"Algorithm Performance {suffix}")
    plt.xlabel("Score")
    plt.ylabel("Algorithms")
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_boxplot{suffix}.png'))
    plt.close()

    radar_data = ranks.copy()
    labels = sub_functions
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", len(algorithms))

    for i, algo in enumerate(algorithms):
        values = radar_data[algo].values.tolist()
        values += values[:1]
        marker = None
        linestyle = None
        color = palette[i]
        linewidth = 1
        if algo.upper() == 'NOAH':
            marker = 'o'
            linestyle = '-.'
            color = 'red'
            linewidth = 1.5
        ax.plot(angles, values, linewidth=linewidth, label=algo, marker=marker, linestyle=linestyle, color=color)
        ax.fill(angles, values, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(len(algorithms), 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(INPUT_JSON.replace('.json', f'_radar{suffix}.png'))
    plt.close()


if cllass == 'CEC2020RW' and SPLIT_CEC:
    split_size = int(np.ceil(len(functions) / N_SPLITS))
    for i in range(N_SPLITS):
        sub_functions = functions[i * split_size:(i + 1) * split_size]
        run_analysis(sub_functions, suffix=f"_part{i + 1}")
elif cllass == 'CEC2017' and SPLIT_CEC:
    split_size = int(np.ceil(len(functions) / N_SPLITS))
    for i in range(N_SPLITS):
        sub_functions = functions[i * split_size:(i + 1) * split_size]
        run_analysis(sub_functions, suffix=f"_part{i + 1}")
else:
    run_analysis(functions)