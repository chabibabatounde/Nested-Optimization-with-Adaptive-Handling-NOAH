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
        for j in range(1, 1 + n):
            i = n - j + 1
            module_name = f"Benchmark.CEC2017.F{i}"
            class_name = f"F{i}"
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
        for i in lrange:
            module_name = f"Benchmark.CEC2020.RW{i}"
            class_name = f"RW{i}"
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            fn_list.append(cls)

    elif 'CEC2022' == class_name:
        n = 12
        for i in range(1, 1 + n):
            module_name = f"Benchmark.CEC2022.F{i}"
            class_name = f"F{i}"
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
    file_path = os.getcwd() + "/Benchmark/InitialsData/" + class_name + '_' + function_name + "_" + str(
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
    return initials, dimension


def get_algorithm_instance(m, iteration, pop_size, seed, p_levy, levy_k, alpha_max):
    models = {
        'NOAH': Noah(
            epoch=iteration,
            pop_size=pop_size,

            p_levy = p_levy,
            levy_k = levy_k,
            alpha_max = alpha_max,

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
        'CCCO': CCCO(
            epoch=iteration,
            pop_size=pop_size,
            seed=seed
        ),
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
        'WOA': WOA.OriginalWOA(
            epoch=iteration,
            pop_size=pop_size,
            sort_flag=True,
            seed=seed
        ),
        'GA': GA.BaseGA(
            epoch=iteration,
            pop_size=pop_size,
            sort_flag=True,
            seed=seed
        ),
        'SADE': DE.SADE(
            epoch=iteration,
            pop_size=pop_size,
            sort_flag=True,
            seed=seed
        )
    }
    return models[m]