
from Benchmark.Functions.__Function import Function
from opfunu.cec_based.cec2017 import *
import numpy as np

class F28(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        domain = [(-100, 100)] * dim

        name = "F28"
        max_dimension = 10
        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            f_x_start,
            max_dimension,
            name
        )
        self.opfunu = F282017(ndim=10)
        self.types(['CEC2017'])

    def eval(self, variables_values):
        return self.opfunu.evaluate(variables_values)
