import os

template = """
from Benchmark.Functions.__Function import Function
from opfunu.cec_based.cec2017 import *
import numpy as np

class F{n}(Function):
    def __init__(self, dim):
        x_start = np.zeros(dim)
        f_x_start = 0
        domain = (-100, 100)
        name = "F{n}"
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
        self.opfunu = F{n}2017(ndim=self.dimension())
        self.types(['CEC2017'])

    def eval(self, variables_values):
        return self.opfunu.evaluate(variables_values)
"""

output_dir = "."

for i in range(10, 31):
    filename = os.path.join(output_dir, f"F{i}.py")
    with open(filename, "w") as f:
        f.write(template.format(n=i))

print("Files F10.py to F30.py generated.")
