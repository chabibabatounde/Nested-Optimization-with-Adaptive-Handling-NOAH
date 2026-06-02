'''

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# =========================
# 1. LOAD DATA
# =========================

INPUT_JSON = os.getcwd() + "/Tests/NOAH/CEC2022/20260530020113__dim20_opti50_it100/Statistics.json"
INPUT_JSON = os.getcwd() + "/Tests/NOAH/CEC2020RW/20260530020113__dimX_opti30_it100/Statistics.json"
INPUT_JSON = os.getcwd() + "/Tests/NOAH/CEC2022/20260531150310__dim20_opti30_it100/Statistics.json"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)
'''
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import scikit_posthocs as sp
import os

# ============================================================
# 1. Charger les données
# ============================================================
INPUT_JSON = os.getcwd() + "/Tests/NOAH/CEC2022/STATS_DATASET.json"

with open(INPUT_JSON, "r") as f:
    data = json.load(f)

functions = list(data.keys())
algorithms = list(data[functions[0]].keys())

df = pd.DataFrame(index=functions, columns=algorithms, dtype=float)
for f in functions:
    for algo in algorithms:
        df.loc[f, algo] = np.mean(data[f][algo]["scores"])

print("=== Scores moyens par fonction ===")
print(df)

# ============================================================
# 2. Friedman test + Mean Ranking
# ============================================================
ranks = df.apply(lambda row: rankdata(row.values), axis=1, result_type="expand")
ranks.columns = df.columns
mean_ranks = ranks.mean(axis=0).sort_values()

print("\n=== Friedman Mean Ranking ===")
print(mean_ranks)

stat, p_value = friedmanchisquare(*[df[a].values for a in algorithms])
print(f"\nFriedman statistic = {stat:.4f}, p-value = {p_value:.6f}")

# ➜ Référence = meilleur algo selon Friedman (rang moyen le plus faible)
reference = mean_ranks.index[0]
print(f"\n>>> Algorithme de référence (1er du classement) : {reference}")

# ============================================================
# 3. Test post-hoc de Nemenyi
# ============================================================
nemenyi = sp.posthoc_nemenyi_friedman(df.values)
nemenyi.index = algorithms
nemenyi.columns = algorithms
print("\n=== Nemenyi pairwise p-values ===")
print(nemenyi.round(4))

# ============================================================
# 4. Wilcoxon signed-rank test (référence vs autres)
# ============================================================
wilcoxon_results = []
plus_total, equal_total, minus_total = 0, 0, 0
alpha = 0.05

for algo in algorithms:
    if algo == reference:
        continue
    x = df[reference].values
    y = df[algo].values
    try:
        stat_w, p_w = wilcoxon(x, y)
    except ValueError:
        stat_w, p_w = np.nan, 1.0

    # + : référence meilleure significativement
    # - : référence pire significativement
    # = : non significatif
    if p_w < alpha:
        if np.mean(x) < np.mean(y):
            sign = "+"; plus_total += 1
        else:
            sign = "-"; minus_total += 1
    else:
        sign = "="; equal_total += 1

    wilcoxon_results.append({
        "Algorithme": algo,
        "Statistique W": round(stat_w, 4) if not np.isnan(stat_w) else "NA",
        "p-value": round(p_w, 6),
        f"Signe ({reference} vs autre)": sign
    })

wilcoxon_df = pd.DataFrame(wilcoxon_results)
overall = f"{plus_total}/{equal_total}/{minus_total}"
print(f"\n=== Wilcoxon signed-rank test (référence = {reference}) ===")
print(wilcoxon_df.to_string(index=False))
print(f"\nOverall (+/=/-): {overall}")

# ============================================================
# 5. Figure résumé du classement
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Bar plot des rangs moyens — référence mise en évidence
colors = []
for algo in mean_ranks.index:
    colors.append("#d62728" if algo == reference else "#4c72b0")

axes[0].barh(mean_ranks.index, mean_ranks.values, color=colors, edgecolor="black")
axes[0].invert_yaxis()
axes[0].set_xlabel("Rang moyen (Friedman)")
axes[0].set_title(f"Classement Friedman\n(p-value = {p_value:.2e}) — Référence : {reference}")
for i, v in enumerate(mean_ranks.values):
    axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontweight="bold")
axes[0].grid(axis="x", linestyle="--", alpha=0.6)

# (b) Heatmap Nemenyi
nemenyi_sorted = nemenyi.loc[mean_ranks.index, mean_ranks.index]
im = axes[1].imshow(nemenyi_sorted.values, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")
axes[1].set_xticks(range(len(algorithms)))
axes[1].set_yticks(range(len(algorithms)))
axes[1].set_xticklabels(nemenyi_sorted.columns, rotation=45, ha="right")
axes[1].set_yticklabels(nemenyi_sorted.index)
axes[1].set_title("Nemenyi - p-values\n(rouge = significatif, p<0.05)")
for i in range(len(algorithms)):
    for j in range(len(algorithms)):
        val = nemenyi_sorted.values[i, j]
        axes[1].text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="black", fontsize=8)
plt.colorbar(im, ax=axes[1], label="p-value")

plt.suptitle(f"Analyse statistique — Overall {reference} vs autres (+/=/-): {overall}",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("statistical_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 6. Export résultats
# ============================================================
with pd.ExcelWriter("statistical_results.xlsx") as writer:
    df.to_excel(writer, sheet_name="Scores_moyens")
    ranks.to_excel(writer, sheet_name="Rangs")
    mean_ranks.to_frame("Rang_moyen").to_excel(writer, sheet_name="Friedman_ranking")
    nemenyi.to_excel(writer, sheet_name="Nemenyi")
    wilcoxon_df.to_excel(writer, sheet_name="Wilcoxon", index=False)

print("\n✅ Résultats exportés : statistical_results.xlsx + statistical_analysis.png")
