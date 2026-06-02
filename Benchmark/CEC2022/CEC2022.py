import os

import numpy as np


class CEC2022:

    def __init__(self, dim, func_num):
        self.dim = dim
        self.func_num = func_num
        base_path = "/Applications/Python/CCE_Mealpy_Implement/Benchmark/Publication/CEC2022/data2"
        self.shift = self.load_shift(base_path)
        self.rotation = self.load_rotation(base_path)
        self.shuffle = self.load_shuffle(base_path)

    def apply_shuffle(self, x):
        return x[self.shuffle]

    def transform(self, x):
        z = x - self.shift
        z = self.rotation @ z
        return z

    def load_shuffle(self, base_path):
        file = os.path.join(
            base_path,
            f"shuffle_data_{self.func_num}_D{self.dim}.txt"
        )
        if os.path.exists(file):
            s = np.loadtxt(file).astype(int)
            return s - 1  # index python
        else:
            return np.arange(self.dim)

    def load_shift(self, base_path):
        file = os.path.join(base_path, f"shift_data_{self.func_num}.txt")
        data = np.loadtxt(file)
        data = data.reshape(-1)  # transforme en vecteur
        return data[:self.dim]

    def load_rotation(self, base_path):
        file = os.path.join(
            base_path,
            f"M_{self.func_num}_D{self.dim}.txt"
        )
        M = np.loadtxt(file)
        return M.reshape(self.dim, self.dim)

    def rosenbrock(self, x):
        x = np.asarray(x)

        return np.sum(
            100 * (x[1:] - x[:-1] ** 2) ** 2 +
            (x[:-1] - 1) ** 2
        )

    def ackley(self, x):
        x = np.asarray(x)
        d = len(x)

        return (
                -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
                - np.exp(np.sum(np.cos(2 * np.pi * x)) / d)
                + 20
                + np.e
        )

    def griewank(self, x):
        x = np.asarray(x)

        sum_term = np.sum(x ** 2) / 4000

        prod_term = np.prod(
            np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))
        )

        return sum_term - prod_term + 1

    def schwefel(self, x):
        x = np.asarray(x)

        return 418.9829 * len(x) - np.sum(
            x * np.sin(np.sqrt(np.abs(x)))
        )
