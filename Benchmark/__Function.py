import os
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import re


class Function:
    __dimension = None
    __domain = None
    __optimum = None
    __optimum_variable = None
    __name = None
    __types = []
    __round = None

    def __init__(self, dim, domain, x_star, f_x_start, max_dimension, name, round_nb=None):
        if (max_dimension is not None) and (dim > max_dimension):
            dim = max_dimension
        self.__dimension = dim
        self.max_dimension = max_dimension
        self.__domain = domain
        self.__optimum = f_x_start
        self.__optimum_variable = x_star
        self.__name = name
        self.__round = round_nb

    def types(self, t):
        self.__types = t

    def my_types(self):
        return self.__types

    def show(self, path=None):
        d1, d2 = self.__domain, self.__domain
        if type(self.__domain) is list:
            d1 = self.__domain[0]
            d2 = self.__domain[1]
        x1 = np.linspace(d1[0], d1[1], 100)
        x2 = np.linspace(d2[0], d2[1], 100)
        X, Y = np.meshgrid(x1, x2)
        Z = np.array([[self.eval([x1, x2]) for x1 in x1] for x2 in x2])
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x1, x2)")
        ax.set_title(self.__name)
        ax.figure.tight_layout()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.clf()

    def plot_2D(self):
        d1, d2 = self.__domain, self.__domain
        if type(self.__domain) is list:
            d1 = self.__domain[0]
            d2 = self.__domain[1]
        x1 = np.linspace(d1[0], d1[1], 100)
        x2 = np.linspace(d2[0], d2[1], 100)
        X, Y = np.meshgrid(x1, x2)
        Z = np.array([[self.eval([x1, x2]) for x1 in x1] for x2 in x2])
        fig = plt.figure(figsize=(10, 10))
        contour = plt.contour(X, Y, Z, levels=30, cmap='viridis')
        plt.clabel(contour, inline=True, fontsize=8)
        plt.colorbar(contour, label='f(x1, x2)')
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Projection 2D (courbes de niveau) de la fonction d'Ackley")
        plt.grid()
        plt.show()

    def generate_proposal(self):
        proposal = []
        pos = 0
        domain = self.__domain
        for j in range(self.__dimension):
            if type(domain) is list:
                domain = self.__domain[pos]
                pos += 1
            if random.choice([-1, 1]) == 1:
                proposal.append(np.random.uniform(domain[0], domain[1]))
            else:
                proposal.append(np.random.randint(domain[0], domain[1]))
        return tuple(proposal)

    def round_nb(self):
        return self.__round

    def get_types(self):
        return self.__types

    def eval(self, variables_values):
        exit("Please define " + self.__name + " Class")

    def dimension(self):
        return self.__dimension

    def domain(self):
        return self.__domain

    def optimum(self):
        return self.__optimum

    def optimum_variable(self):
        return self.__optimum_variable

    def name(self):
        return self.__name

    # Helper functions
    def sphere(self, x):
        return np.sum(x ** 2)

    def rastrigin(self, x):
        return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

    def griewank(self, x):
        n = len(x)
        return 1 + np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))

    def ackley(self, x):
        n = len(x)
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + 20 + np.e

    def rosenbrock(self, x):
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

    def schwefel(self, x):
        return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
