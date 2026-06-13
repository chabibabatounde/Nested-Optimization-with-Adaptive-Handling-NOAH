import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import rankdata

# ==============================
# 🔹 Lecture des données
# ==============================

def lire_fichiers_data_json(repertoire_racine):
    data = []
    algo_names = []

    for racine, dossiers, fichiers in sorted(os.walk(repertoire_racine)):
        if 'dataset.json' in fichiers:
            chemin_fichier = os.path.join(racine, 'dataset.json')
            try:
                with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
                    nom_algo = os.path.basename(racine)
                    data.append(json.load(fichier))
                    algo_names.append(nom_algo)
            except Exception as e:
                print(f"Erreur lors de la lecture de {chemin_fichier} : {e}")

    return data, algo_names

# ==============================
# 🔹 Chargement
# ==============================

parametre = "Version"
repertoire = os.getcwd() + "/Tests/CEC2022/" + parametre

data, algo_names = lire_fichiers_data_json(repertoire)

processed_data = {}
functions = list(data[0].keys())

# ==============================
# 🔹 Structuration des données
# ==============================

for fn in functions:
    processed_data[fn] = dict()
    for i in range(len(data)):
        lst = []
        values = data[i][fn]['NOAH']['optimizations']
        for v in values:
            lst.append(v['score'])
        processed_data[fn][algo_names[i]] = lst.copy()

# sauvegarde JSON
with open(repertoire + "/" + parametre + ".json", "w", encoding="utf-8") as file:
    json.dump(processed_data, file, indent=1, ensure_ascii=False)

# ==============================
# 🔹 Ranking
# ==============================

algorithms = list(processed_data["F1"].keys())

mean_scores = {alg: [] for alg in algorithms}
ranks_per_function = []

for func in functions:
    scores = {}
    for alg in algorithms:
        values = np.array(processed_data[func][alg])
        mean_val = np.mean(values)
        scores[alg] = mean_val
        mean_scores[alg].append(mean_val)

    vals = list(scores.values())
    ranks = rankdata(vals, method='average')
    ranks_dict = dict(zip(algorithms, ranks))
    ranks_per_function.append(ranks_dict)

# ranking global
global_ranks = {}
for alg in algorithms:
    avg_rank = np.mean([r[alg] for r in ranks_per_function])
    global_ranks[alg] = avg_rank

# tri pour affichage classement (performance)
final_ranking = sorted(global_ranks.items(), key=lambda x: x[1])

# ==============================
# 🔹 Affichage classement
# ==============================

print("\n=== Classement Global (plus petit = meilleur) ===")
for i, (alg, score) in enumerate(final_ranking, 1):
    print(f"{i}. {alg} - Rank moyen: {score:.3f}")

# ==============================
# 🔹 Ordre alphabétique (plots)
# ==============================

sorted_algorithms = sorted(algorithms, key=lambda x: x.lower())
sorted_ranks = [global_ranks[alg] for alg in sorted_algorithms]
best_algo = min(global_ranks, key=global_ranks.get)

# ==============================
# 🔹 Barplot
# ==============================

plt.figure(figsize=(5, 5))
alogox = [parametre + ' = ' + i for i in sorted_algorithms]
bars = plt.bar(alogox, sorted_ranks)

for i, alg in enumerate(sorted_algorithms):
    if alg == best_algo:
        bars[i].set_color('green')

plt.xlabel("Valeurs de " + parametre)
plt.ylabel("Rang moyen")
plt.xticks(rotation=45)

for i, v in enumerate(sorted_ranks):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.tight_layout()
plt.savefig(repertoire + "/" + parametre + "_bar.png")
plt.close()

# ==============================
# 🔹 Heatmap
# ==============================

rank_matrix = []
for r in ranks_per_function:
    rank_matrix.append([r[alg] for alg in sorted_algorithms])

df = pd.DataFrame(rank_matrix, columns=sorted_algorithms, index=functions)

plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap="viridis_r")
plt.title("Rangs par fonction (ordre alphabétique)")
plt.xlabel("Algorithmes")
plt.ylabel("Fonctions")
plt.tight_layout()
plt.savefig(repertoire + "/" + parametre + "_heat.png")
plt.close()

# ==============================
# 🔹 Calcul mean/std pour tableau LaTeX
# ==============================

stats_data = {}
for func in functions:
    stats_data[func] = {}
    for alg in algorithms:
        values = np.array(processed_data[func][alg])
        stats_data[func][alg] = (np.mean(values), np.std(values))

latex_algorithms = sorted(algorithms, key=lambda x: x.lower())

# ==============================
# 🔹 Génération LaTeX résultats (mean ± std)
# ==============================

latex = []
latex.append("\\begin{center}")
latex.append("\\small")
latex.append("\\begin{longtable}{c" + "c" * len(latex_algorithms) + "}")
latex.append("\\caption{Results for $" + parametre + "$}")
latex.append("\\label{tab:sensity_" + parametre.lower() + "} \\\\")
latex.append("\\toprule")

header = ""
for alg in latex_algorithms:
    header += '$' + parametre + ' = ' + alg + '$ & '
header = header[:-2]
latex.append(" & " + header + " \\\\")
latex.append("\\midrule")
latex.append("\\endfirsthead")
latex.append("\\toprule")
latex.append(" & " + header + " \\\\")
latex.append("\\midrule")
latex.append("\\endhead")

for func in functions:
    row = [f"$F_{{{func[1:]}}}$"]
    means = [stats_data[func][alg][0] for alg in latex_algorithms]
    best_mean = min(means)

    for alg in latex_algorithms:
        mean, std = stats_data[func][alg]
        cell = f"{mean:.2e} $\\pm$ {std:.2e}"
        if np.isclose(mean, best_mean):
            cell = "\\textbf{" + cell + "}"
        row.append(cell)

    latex.append(" & ".join(row) + " \\\\")

latex.append("\\bottomrule")
latex.append("\\end{longtable}")
latex.append("\\end{center}")

latex_path = os.path.join(repertoire, parametre + "_table.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex))
print(f"\n✅ Tableau LaTeX généré : {latex_path}")

# ==============================
# 🔹 ANALYSE STATISTIQUE DES VERSIONS (ABLATION)
# ==============================
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

print("\n" + "=" * 60)
print("ANALYSE STATISTIQUE DE L'ÉTUDE D'ABLATION")
print("=" * 60)

friedman_matrix = []
for func in functions:
    row = [np.mean(processed_data[func][alg]) for alg in algorithms]
    friedman_matrix.append(row)

friedman_matrix = np.array(friedman_matrix)

stat, p_value = friedmanchisquare(*[friedman_matrix[:, i] for i in range(len(algorithms))])
print(f"\n[Friedman] statistic = {stat:.4f}, p-value = {p_value:.4e}")

df_friedman = pd.DataFrame(friedman_matrix, columns=algorithms)
nemenyi = sp.posthoc_nemenyi_friedman(df_friedman.values)
nemenyi.index = algorithms
nemenyi.columns = algorithms

reference = "V0"
print(f"\n[Wilcoxon] {reference} vs variantes (global) :")

wilcoxon_global = {}
ref_scores_global = np.array([np.mean(processed_data[f][reference]) for f in functions])

for alg in algorithms:
    if alg == reference:
        continue
    var_scores = np.array([np.mean(processed_data[f][alg]) for f in functions])
    try:
        w_stat, w_p = wilcoxon(ref_scores_global, var_scores)
    except ValueError:
        w_stat, w_p = np.nan, 1.0

    symbol = "+" if (w_p < 0.05 and np.mean(ref_scores_global) < np.mean(var_scores)) else \
             "-" if (w_p < 0.05) else "="
    wilcoxon_global[alg] = (w_p, symbol)
    print(f"  {reference} vs {alg} : p = {w_p:.4e}  [{symbol}]")

# ==============================
# 🔹 Wilcoxon PAR FONCTION
# ==============================

print(f"\n[Wilcoxon par fonction] {reference} vs variantes :")

# wilcoxon_per_func[func][alg] = (p_value, symbol)
wilcoxon_per_func = {}

others = [alg for alg in algorithms if alg != reference]

for func in functions:
    wilcoxon_per_func[func] = {}
    ref_runs = np.array(processed_data[func][reference])  # scores bruts des runs

    for alg in others:
        alg_runs = np.array(processed_data[func][alg])

        # s'assurer que les deux vecteurs ont la même taille
        min_len = min(len(ref_runs), len(alg_runs))
        r = ref_runs[:min_len]
        a = alg_runs[:min_len]

        try:
            w_stat, w_p = wilcoxon(r, a)
        except ValueError:
            # différences toutes nulles
            w_stat, w_p = np.nan, 1.0

        if np.isnan(w_p) or w_p >= 0.05:
            symbol = "="
        elif np.median(r) < np.median(a):
            symbol = "+"
        else:
            symbol = "-"

        wilcoxon_per_func[func][alg] = (w_p, symbol)

# ==============================
# 🔹 LaTeX : Wilcoxon par fonction (longtable)
# ==============================

def fmt_p(p):
    """Formate la p-value en notation scientifique LaTeX."""
    if np.isnan(p):
        return "--"
    return f"${p:.2e}$".replace("e-0", "e-").replace("e+0", "e+")

# comptage des +/=/- par version
counts = {alg: {"+": 0, "=": 0, "-": 0} for alg in others}
for func in functions:
    for alg in others:
        _, sym = wilcoxon_per_func[func][alg]
        counts[alg][sym] += 1

# construction du tableau
col_spec = "c" + "cc" * len(others)  # Fonction | (p, sym) x n_others
latex_w = []
latex_w.append("\\begin{center}")
latex_w.append("\\small")
latex_w.append("\\renewcommand{\\arraystretch}{1.25}")
latex_w.append(f"\\begin{{longtable}}{{{col_spec}}}")
latex_w.append(f"\\caption{{Wilcoxon signed-rank test per function: "
               f"{reference} vs ablated variants (CEC~2022, $D=20$). "
               f"$+$: {reference} significantly better; "
               f"$-$: {reference} significantly worse; "
               f"$=$: no significant difference ($\\alpha=0.05$).}}")
latex_w.append("\\label{tab:wilcoxon_ablation_per_func} \\\\")
latex_w.append("\\toprule")

# header ligne 1 : noms des versions
h1 = "Function"
for alg in others:
    h1 += f" & \\multicolumn{{2}}{{c}}{{{reference} vs {alg}}}"
latex_w.append(h1 + " \\\\")

# header ligne 2 : p-value / result
h2 = ""
for alg in others:
    h2 += " & $p$-value & Result"
latex_w.append(h2 + " \\\\")
latex_w.append("\\cmidrule(lr){1-1}")
for idx, alg in enumerate(others):
    start_col = 2 + idx * 2
    latex_w.append(f"\\cmidrule(lr){{{start_col}-{start_col + 1}}}")

latex_w.append("\\midrule")
latex_w.append("\\endfirsthead")

# repeat header on next pages
latex_w.append("\\toprule")
latex_w.append(h1 + " \\\\")
latex_w.append(h2 + " \\\\")
latex_w.append("\\midrule")
latex_w.append("\\endhead")

latex_w.append("\\midrule")
latex_w.append(f"\\multicolumn{{{1 + 2 * len(others)}}}{{r}}{{\\textit{{Continued on next page}}}} \\\\")
latex_w.append("\\endfoot")
latex_w.append("\\bottomrule")
latex_w.append("\\endlastfoot")

# lignes par fonction
for func in functions:
    row = [f"$F_{{{func[1:]}}}$"]
    for alg in others:
        w_p, sym = wilcoxon_per_func[func][alg]
        # mettre en gras si significatif
        p_str = fmt_p(w_p)
        if sym == "+":
            sym_str = "\\textbf{$+$}"
            p_str = "\\textbf{" + fmt_p(w_p) + "}"
        elif sym == "-":
            sym_str = "$-$"
        else:
            sym_str = "$=$"
        row.append(p_str)
        row.append(sym_str)
    latex_w.append(" & ".join(row) + " \\\\")

# ligne de comptage
latex_w.append("\\midrule")
count_row = ["\\textbf{Count}"]
for alg in others:
    p_count = counts[alg]["+"]
    e_count = counts[alg]["="]
    m_count = counts[alg]["-"]
    count_row.append(f"\\multicolumn{{2}}{{c}}{{$+$:{p_count} / $=$:{e_count} / $-$:{m_count}}}")
latex_w.append(" & ".join(count_row) + " \\\\")

latex_w.append("\\end{longtable}")
latex_w.append("\\end{center}")

wilcoxon_func_path = os.path.join(repertoire, parametre + "_wilcoxon_per_func.tex")
with open(wilcoxon_func_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_w))
print(f"\n✅ Wilcoxon par fonction généré : {wilcoxon_func_path}")

# ==============================
# 🔹 Friedman + Wilcoxon global LaTeX
# ==============================

latex_ablation = []
latex_ablation.append("\\begin{center}")
latex_ablation.append("\\small")
latex_ablation.append("\\renewcommand{\\arraystretch}{1.25}")
latex_ablation.append("\\begin{longtable}{clc}")
latex_ablation.append(f"\\caption{{Friedman mean ranking for ablation study. "
                      f"Statistic $={stat:.4f}$, $p={p_value:.4e}$.}}")
latex_ablation.append("\\label{tab:friedman_ablation} \\\\")
latex_ablation.append("\\toprule")
latex_ablation.append("Rank & Version & Mean Rank \\\\")
latex_ablation.append("\\midrule")
latex_ablation.append("\\endfirsthead")
latex_ablation.append("\\toprule")
latex_ablation.append("Rank & Version & Mean Rank \\\\")
latex_ablation.append("\\midrule")
latex_ablation.append("\\endhead")

for i, (alg, score) in enumerate(final_ranking, 1):
    if alg == reference:
        latex_ablation.append(f"{i} & \\textbf{{{alg}}} & \\textbf{{{score:.4f}}} \\\\")
    else:
        latex_ablation.append(f"{i} & {alg} & {score:.4f} \\\\")

latex_ablation.append("\\bottomrule")
latex_ablation.append("\\end{longtable}")
latex_ablation.append("\\end{center}")

friedman_path = os.path.join(repertoire, parametre + "_friedman_ablation.tex")
with open(friedman_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_ablation))
print(f"✅ Tableau Friedman (ablation) généré : {friedman_path}")

# Wilcoxon global
latex_wilcoxon = []
latex_wilcoxon.append("\\begin{table}[h]")
latex_wilcoxon.append("\\centering")
latex_wilcoxon.append("\\renewcommand{\\arraystretch}{1.25}")
latex_wilcoxon.append(f"\\caption{{Wilcoxon signed-rank test: {reference} "
                      f"(full version) vs ablated variants (global, CEC~2022, $D=20$).}}")
latex_wilcoxon.append("\\label{tab:wilcoxon_ablation_global}")
latex_wilcoxon.append("\\begin{tabular}{lcc}")
latex_wilcoxon.append("\\toprule")
latex_wilcoxon.append("Comparison & $p$-value & Result \\\\")
latex_wilcoxon.append("\\midrule")

for alg in algorithms:
    if alg == reference:
        continue
    w_p, symbol = wilcoxon_global[alg]
    sym_str = "\\textbf{$+$}" if symbol == "+" else ("$-$" if symbol == "-" else "$=$")
    latex_wilcoxon.append(f"{reference} vs {alg} & ${w_p:.4e}$ & {sym_str} \\\\")

latex_wilcoxon.append("\\bottomrule")
latex_wilcoxon.append("\\end{tabular}")
latex_wilcoxon.append("\\end{table}")

wilcoxon_path = os.path.join(repertoire, parametre + "_wilcoxon_ablation.tex")
with open(wilcoxon_path, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_wilcoxon))
print(f"✅ Tableau Wilcoxon global (ablation) généré : {wilcoxon_path}")

# ==============================
# 🔹 Heatmap Nemenyi
# ==============================

plt.figure(figsize=(7, 6))
sns.heatmap(nemenyi, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
plt.title("Nemenyi post-hoc p-values (ablation)")
plt.tight_layout()
plt.savefig(repertoire + "/" + parametre + "_nemenyi_heatmap.png", dpi=300)
plt.close()
print(f"✅ Heatmap Nemenyi générée.")
