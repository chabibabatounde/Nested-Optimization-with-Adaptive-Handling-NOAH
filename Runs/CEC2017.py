import json
import time
from datetime import datetime
from mealpy import FloatVar
from Utils import utils
import numpy as np
from matplotlib import pyplot as plt

models = ['GA', 'PSO', 'DE', 'SADE', 'GWO', 'WarSO', 'NIADE', 'CGO', 'SOA', 'NOAH']
# ABC

models = [
    'NOAH',
    'GA', 'PSO', 'ABC',
    'GWO', 'WarSO', 'WOA',
    'NIADE', 'BKA', 'CGO', 'SOA'
]

for dimension in [10, 50, 100]:
    fn_class = 'CEC2017'
    nb_opti = 30
    generation = 100  # Nombre de génération
    pop_size = 50  # Taille de la population
    window_length = 5
    # ========================================================== #

    directory = '_dim' + str(dimension) + '_opti' + str(nb_opti) + '_it' + str(generation)
    directory = datetime.now().strftime("Tests/NOAH/" + fn_class + "/%Y%m%d%H%M%S_" + directory + '/')
    functions = utils.get_functions(fn_class)
    print('---------------------------------------')
    print('Optimisation for', len(functions), 'functions')
    print('---------------------------------------\n')
    optimizations = dict()
    data_sum = dict()
    counter = 0
    for function in functions:
        counter += 1
        fn = function(dimension)
        domain = fn.domain()
        initial_data = utils.file_management(fn, dimension, pop_size, domain, fn_class, directory)
        print('Optimisation', str(counter) + '/' + str(len(functions)), 'for', fn.name(), 'in D = ', dimension)
        optimizations[fn.name()] = {}
        data_sum[fn.name()] = dict()
        scorus = {}
        for m in models:
            scorus[m] = []
            start_score = np.inf
            for k in initial_data.copy():
                if fn.eval(k) < start_score:
                    start_score = fn.eval(k)
            print('\t', m, 'is running...')
            optimizations[fn.name()][m] = {'optimizations': []}
            for i in range(nb_opti):
                seed = i
                algorithm = utils.get_algorithm_instance(
                    m,
                    generation,
                    pop_size,
                    window_length,
                    seed
                )

                lb = [b[0] for b in domain]
                ub = [b[1] for b in domain]

                problem = {
                    "obj_func": fn.eval, "minmax": "min", "log_to": None, "target": fn.optimum() * dimension,
                    "bounds": FloatVar(lb=lb, ub=ub),
                }
                start_time = time.perf_counter()
                g_best = algorithm.solve(problem, mode="thread", starting_solutions=initial_data.copy(), seed=seed)
                end_time = time.perf_counter()
                ax = []
                pops = []
                if hasattr(algorithm.history, "ax"):
                    ax = algorithm.history.ax
                data = [start_score]
                if hasattr(algorithm.history, "pops"):
                    debut = []
                    for el in initial_data.copy().tolist():
                        debut.append(el)
                    pops = [debut] + algorithm.history.pops
                for element in list(algorithm.history.list_current_best_fit):
                    data.append(element)
                    if element == fn.optimum() * dimension:
                        break
                result = {
                    'score': min(data),
                    'scores_evolution': data,
                    'optimum': fn.optimum() * dimension,
                    'ax': ax,
                    'iteration': len(data),
                    'value': algorithm.g_best.solution.tolist(),
                    'pops': pops,
                }
                scorus[m].append(result['score'])
                optimizations[fn.name()][m]['optimizations'].append(result)
            scorus[m] = {'min': min(scorus[m]), 'mean': np.mean(scorus[m]), 'max': max(scorus[m])}
            best = 10e100
            best_index = 0
            tic = 0
            scores = {}
            for op in optimizations[fn.name()][m]['optimizations']:
                if op['score'] < best:
                    best_index = tic
                tic += 1
            optimizations[fn.name()][m]['best_index'] = best_index
        scorus = dict(
            sorted(
                scorus.items(),
                key=lambda x: (x[1]['mean'], 0 if x[0] == "CCEO" else 1)
            )
        )

        mins = []
        maxs = []
        means = []
        for k in list(scorus.keys()):
            mins.append(scorus[k]['min'])
            maxs.append(scorus[k]['max'])
            means.append(scorus[k]['mean'])
        plt.bar(list(scorus.keys()), maxs, label='max')
        plt.bar(list(scorus.keys()), means, label='mean')
        plt.bar(list(scorus.keys()), mins, label='min')
        plt.xlabel('Algorithm')
        plt.ylabel('Score')
        plt.legend()
        plt.savefig(directory + fn.name() + ".png")
        plt.clf()
        with open(directory + fn.name() + "_dataset.json", "w", encoding="utf-8") as file:
            json.dump(scorus, file, indent=1, ensure_ascii=False)

    if len(functions) != 0:
        with open(directory + "dataset.json", "w", encoding="utf-8") as file:
            json.dump(optimizations, file, ensure_ascii=False)
