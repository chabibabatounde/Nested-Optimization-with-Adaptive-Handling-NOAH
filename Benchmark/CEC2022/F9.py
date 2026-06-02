from Benchmark.__Function import Function
from opfunu.cec_based.cec2022 import *
import numpy as np


class F9(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        domain = [(-100, 100)] * dim
        name = "F9"
        max_dimension = 20
        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            f_x_start,
            max_dimension,
            name
        )
        self.opfunu = F92022(ndim=self.dimension())
        self.types(['CEC2022'])

    def eval(self, variables_values):
        return self.opfunu.evaluate(variables_values)
