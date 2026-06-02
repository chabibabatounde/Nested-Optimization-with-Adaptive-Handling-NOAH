import json
import os
import random
import numpy as np
import importlib

from matplotlib import pyplot as plt
from mealpy import ABC, GA, DE, BBO, GWO, PSO, FPA, SHADE, SMA, WarSO, WOA, HS, EHO, HHO, SCA, SA, NRO, ASO, ES

from Algorithms.NIADE import NIADE
from Algorithms.NOAH import Noah
from Algorithms.SequoiaOA import SequoiaOA
from Algorithms.DBO import DBO
from Algorithms.CGO import CGO
from Algorithms.BKA import BKA
from Algorithms.DOA import DOA
from Algorithms.CCCO import CCCO


def get_functions(class_name):
    fn_list = []
    if 'CEC2017' == class_name:
        n = 29
        for n in range(1, 1 + n):
            module_name = f"Benchmark.CEC2017.F{n}"
            class_name = f"F{n}"
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            fn_list.append(cls)
    elif 'CEC2020RW' == class_name:
        n = 50
        lrange = list(range(1, 1 + n))
        lrange.remove(40)
        lrange.remove(41)
        lrange.remove(42)
        lrange.remove(43)
        lrange.remove(44)

        for n in lrange:
            module_name = f"Benchmark.CEC2020.RW{n}"
            class_name = f"RW{n}"
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            fn_list.append(cls)

    elif 'CEC2022' == class_name:
        n = 12
        # for n in [11, 10, 4, 5, 1, 2, 3, 6, 7, 8, 9, 12]:
        for n in range(1, 1 + n):
            module_name = f"Benchmark.CEC2022.F{n}"
            class_name = f"F{n}"
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            fn_list.append(cls)
    else:
        exit("Class name not recognized")
    return fn_list


def generate_data(size, dimension, domain):
    initials = []
    for s in range(size):
        proposal = []
        for d in range(dimension):
            proposal.append(np.random.uniform(domain[d][0], domain[d][1]))
        initials.append(proposal)

    return np.array(initials, dtype=float)


def file_management(function, dimension, size, domain, class_name, directory):
    function_name = function.name()
    if dimension > function.max_dimension:
        dimension = function.max_dimension

    os.makedirs(os.getcwd() + '/' + directory, exist_ok=True)

    file_path = os.getcwd() + "/../Benchmark/InitialsData/" + class_name + '_' + function_name + "_" + str(
        dimension) + "x" + str(size) + "_.json"
    initials = None
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        initials = np.array(data)
    else:
        initials = generate_data(size, dimension, domain)
        with open(file_path, "w", encoding="utf-8") as fl:
            json.dump(initials.tolist(), fl, indent=2)
    return initials


def get_algorithm_instance(m, iteration, pop_size, window_length, seed):
    models = {
        'NOAH': Noah(
            epoch=iteration,
            pop_size=pop_size,
            window_length=window_length,
            save_population=True,
            sort_flag=True,
            seed=seed
        ),

        'BKA': BKA(
            epoch=iteration,
            pop_size=pop_size,
            p=0.9,
            seed=seed
        ),

        'DOA': DOA(
            epoch=iteration,
            pop_size=pop_size,
            p=0.9,
            seed=seed
        ),

        'CGO': CGO(
            epoch=iteration,
            pop_size=pop_size,
            seed=seed
        ),

        'CCCO': CCCO(
            epoch=iteration,
            pop_size=pop_size,
            seed=seed
        ),

        'ABC': ABC.OriginalABC(epoch=iteration, pop_size=pop_size, n_limits=50,
                               seed=seed),

        'NIADE': NIADE(
            epoch=iteration,
            pop_size=pop_size,
            f=0.4,
            cr=0.9,
            seed=seed
        ),

        'SOA': SequoiaOA(
            epoch=iteration,
            pop_size=pop_size,
            f=0.4,
            cr=0.9,
            seed=seed
        ),

        'DBO': DBO(
            epoch=iteration,
            pop_size=pop_size,
            seed=seed
        ),

        'CMA_ES': ES.CMA_ES(
            epoch=iteration,
            pop_size=pop_size,
            seed=seed
        ),
        'GWO': GWO.OriginalGWO(
            epoch=iteration,
            pop_size=pop_size,
            save_population=True,
            sort_flag=True,
            seed=seed
        ),
        "WarSO": WarSO.OriginalWarSO(
            epoch=iteration,
            pop_size=pop_size,
            rr=0.1,
            save_population=True,
            sort_flag=True,
            seed=seed
        ),
        'PSO': PSO.OriginalPSO(
            epoch=iteration,
            pop_size=pop_size,
            sort_flag=True,
            seed=seed
        ),

        'HHO': HHO.OriginalHHO(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),
        'SCA': SCA.OriginalSCA(epoch=iteration, pop_size=pop_size, seed=seed),
        'WOA': WOA.OriginalWOA(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),
        'EHO': EHO.OriginalEHO(epoch=iteration, pop_size=pop_size, alpha=0.5, beta=0.5, n_clans=5, sort_flag=True,
                               seed=seed),
        'SMA': SMA.OriginalSMA(epoch=iteration, pop_size=pop_size, p_t=0.03, sort_flag=True, seed=seed),
        'SHADE': SHADE.L_SHADE(epoch=iteration, pop_size=pop_size, miu_f=0.5, miu_cr=0.5, seed=seed),
        'GA': GA.BaseGA(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),  # Genetic*

        'DE': DE.JADE(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),  # Differential
        'SADE': DE.SADE(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),  # Differential self adapt
        'SA': SA.OriginalSA(epoch=iteration, pop_size=pop_size, temp_init=100, step_size=0.1, sort_flag=True,
                            seed=seed),
        'ASO': ASO.OriginalASO(epoch=iteration, pop_size=pop_size, alpha=10, beta=0.2, sort_flag=True, seed=seed),
        'NRO': NRO.OriginalNRO(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),

        'HS': HS.OriginalHS(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),  # Harmonic
        'BBO': BBO.OriginalBBO(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),
        'FPA': FPA.OriginalFPA(epoch=iteration, pop_size=pop_size, sort_flag=True, seed=seed),
    }
    return models[m]


def algorithm_abv(algorithm):
    key = algorithm.get_name()

    tab_key = {
        'SequoiaOA': 'SOA',
        'Noah': 'NOAH',
        'DBO': 'DBO',
        'CGO': 'CGO',
        'CCCO': 'CCCO',
        'BKA': 'BKA',
        'DOA': 'DOA',
        'MealpyCCE': 'CCEO',
        'CMA_ES': 'CMA_ES',
        'OriginalGWO': 'GWO',
        'OriginalWarSO': 'WarSO',
        'OriginalPSO': 'PSO',
        'BaseGA': 'GA',
        'JADE': 'DE',
        'OriginalSHADE': 'SHADE',
        'L_SHADE': 'SHADE',
        'SADE': 'SADE',
        'OriginalSA': 'SA',
        'OriginalHS': 'HS',
        'OriginalASO': 'ASO',
        'OriginalNRO': 'NRO',
        'OriginalHHO': 'HHO',
        'OriginalBRO': 'BRO',
        'OriginalSCA': 'SCA',
        'OriginalWOA': 'WOA',
        'OriginalBBO': 'BBO',
        'OriginalFPA': 'FPA',
        'OriginalEHO': 'EHO',
        'OriginalMFO': 'MFO',
        'OriginalAOA': 'AOA',
        'OriginalALO': 'ALO',
        'OriginalSMA': 'SMA'
    }

    return tab_key[key]


def arrange_table(data):
    arr = np.array(data['mean'])
    sorted_indices = np.argsort(arr)
    all = dict()
    for k in data.keys():
        if type(data[k]) is list and len(data[k]) != 0:
            all[k] = []
    for i in sorted_indices:
        for k in all.keys():
            if len(data[k]) != 0:
                all[k].append(data[k][i])
    return all


def plot_summerize(fn, plot_tab, dimension, directory):
    plot_tab = arrange_table(plot_tab)
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.4)
    a = 0.5

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax3 = fig.add_subplot(gs[1, :])

    d1, d2 = fn.domain(), fn.domain()
    if isinstance(fn.domain(), list):
        d1 = fn.domain()[0]
        d2 = fn.domain()[1]
    '''
    x1 = np.linspace(d1[0], d1[1], 100)
    x2 = np.linspace(d2[0], d2[1], 100)
    X, Y = np.meshgrid(x1, x2)
    Z = np.array([[fn.eval([x, y]) for x in x1] for y in x2])
    ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    '''

    # --- Sous-graphique 2 : Barres (moyenne et minimum) ---
    ax1.bar(plot_tab['algo'], plot_tab['mean'], label='Mean', alpha=a)
    ax1.bar(plot_tab['algo'], plot_tab['min'], label='Min', alpha=a)
    ax1.plot(plot_tab['algo'], plot_tab['optimums'], linestyle='--', marker='*', label='Optimal')
    ax1.set_ylabel('Fitness')
    ax1.set_title(f"{fn.name()} : dimension {dimension}")
    ax1.legend()

    # --- Sous-graphique 3 : Évolution des scores ---
    for j in range(len(plot_tab['scores_evolution'])):
        i = len(plot_tab['scores_evolution']) - j - 1
        if plot_tab['algo'][i] == "CCEO":
            ax3.plot(
                plot_tab['itrs'][i], plot_tab['scores_evolution'][i],
                label=plot_tab['algo'][i], linestyle='--'
            )
        else:
            ax3.plot(
                plot_tab['itrs'][i], plot_tab['scores_evolution'][i],
                label=plot_tab['algo'][i],
            )
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Fitness')
    ax3.set_title(f"{fn.name()} : dimension {dimension}")
    ax3.legend()
    plt.savefig(f"{directory}{fn.name()}/___a_summerize.png")
    plt.clf()


def plot_algorithme(fn, plot_tab, data, data_sum, algo, directory):
    a = 0.8

    plt.plot(data['itrs'], data['scores_evolution'], label=algo)
    plt.plot(
        data_sum[fn.name()]['CCEO']['itrs'],
        data_sum[fn.name()]['CCEO']['scores_evolution'],
        label='CCEO',
        linestyle='--'
    )
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/" + algo + "_score_evolution_compared.png")
    plt.clf()

    plt.bar(
        ['CCEO', algo],
        [data_sum[fn.name()]['CCEO']['score_mean'], data['score_mean']],
        label='Mean',
        alpha=a
    )
    plt.bar(
        ['CCEO', algo],
        [data_sum[fn.name()]['CCEO']['score_min'], data['score_min']],
        label='Min',
        alpha=a
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/_" + algo + "_score_compared.png")
    plt.clf()

    e = data_sum[fn.name()][algo]
    plot_tab['optimums'].append(e['optimum'])
    plot_tab['min'].append(e['score_min'])
    plot_tab['mean'].append(e['score_mean'])
    plot_tab['med'].append(e['score_median'])
    plot_tab['std'].append(e['score_std'])
    plot_tab['it_min'].append(e['it_min'])
    plot_tab['it_mean'].append(e['it_mean'])
    plot_tab['it_max'].append(e['it_max'])
    plot_tab['duration_min'].append(e['duration_min'])
    plot_tab['duration_mean'].append(e['duration_mean'])
    plot_tab['scores_evolution'].append(e['scores_evolution'])
    plot_tab['itrs'].append(e['itrs'])
    plot_tab['reach_p'].append(e['reach_p'])
    plot_tab['ax'].append(random.choice(e['ax']))
    plot_tab['algo'].append(algo)
    plt.clf()


def plot_single_compare(fn, plot_tab, dimension, directory):
    a = 0.6
    plt.bar(plot_tab['algo'], plot_tab['mean'], label='Mean', alpha=a)
    plt.bar(plot_tab['algo'], plot_tab['min'], label='Min', alpha=a)
    plt.plot(plot_tab['algo'], plot_tab['optimums'], linestyle='--', marker='*', label='Optimal')
    plt.ylabel('Score')
    plt.title(fn.name() + ' : dimension ' + str(dimension))
    plt.legend()
    plt.savefig(directory + fn.name() + "/___score_compared.png")
    plt.clf()

    plt.bar(plot_tab['algo'], plot_tab['mean'], label='Mean', alpha=a)
    plt.bar(plot_tab['algo'], plot_tab['med'], label='Median', alpha=a)
    plt.bar(plot_tab['algo'], plot_tab['min'], label='Min', alpha=a)
    plt.bar(plot_tab['algo'], plot_tab['std'], label='std', alpha=a)
    plt.plot(plot_tab['algo'], plot_tab['optimums'], linestyle='--', marker='*', label='Optimal')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/___score_compared_all.png")
    plt.clf()

    plt.bar(plot_tab['algo'], plot_tab['reach_p'], alpha=a)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/___targeted_compared.png")
    plt.clf()

    plt.bar(plot_tab['algo'], plot_tab['duration_mean'], label='Mean', alpha=a)
    plt.bar(plot_tab['algo'], plot_tab['duration_min'], label='Min', alpha=a)
    plt.ylabel('Duration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/___duration_compared.png")
    plt.clf()

    plt.bar(plot_tab['algo'], plot_tab['it_max'], label='Max', alpha=a)
    plt.bar(plot_tab['algo'], plot_tab['it_mean'], label='Mean', alpha=a)
    plt.ylabel('Generation')
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/___generation_compared.png")
    plt.clf()

    for j in range(len(plot_tab['scores_evolution'])):
        i = len(plot_tab['scores_evolution']) - j - 1
        if plot_tab['algo'][i] == "CCEO":
            plt.plot(
                plot_tab['itrs'][i], plot_tab['scores_evolution'][i],
                label=plot_tab['algo'][i], linestyle='--')
        else:
            plt.plot(
                plot_tab['itrs'][i], plot_tab['scores_evolution'][i],
                label=plot_tab['algo'][i],
            )
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title(fn.name() + ' : dimension ' + str(dimension))
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory + fn.name() + "/___evolution_compared.png")
    plt.clf()

    try:
        plt.figure(figsize=(8, 8))
        plt.plot(
            plot_tab['itrs'][0][:len(plot_tab['ax'][0])],
            plot_tab['ax'][0],
            label=plot_tab['algo'][0],
            linestyle='--'
        )
        plt.xlabel('Iteration')
        plt.ylabel('$a_{x}$')
        plt.title(fn.name() + ' : dimension ' + str(dimension))
        plt.legend()
        plt.tight_layout()
        plt.savefig(directory + fn.name() + "/___ax.png")
        plt.clf()
    except Exception as e:
        print("Plot ax failed:", e)
    finally:
        plt.clf()


def all_score_compare(data_sum, directory):
    summerize = dict()
    for fn in data_sum:
        fn_summerize = dict()
        for algo in data_sum[fn]:
            if algo not in list(summerize.keys()):
                summerize[algo] = []
            summerize[algo].append(data_sum[fn][algo]['score_min'])
    tab = []
    for algo in summerize:
        tab.append(np.mean(summerize[algo]))
    plt.bar(list(summerize.keys()), tab)
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(directory + "___All_mean_score.png")
    plt.clf()
