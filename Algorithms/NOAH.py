import numpy as np
from mealpy import Optimizer


class Noah(Optimizer):
    def __init__(
            self,
            epoch=100,  # Maximum generation
            pop_size=100,  # Population size
            **kwargs
    ):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        # Main parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])

        # Internal parameters (Computational variable)
        self.dimension = None
        self.lb = None
        self.ub = None

        # --- Initialization -----------------------------------------------------
    def initialize_variables(self):
        self.dimension = len(self.problem.ub)
        self.history.pops = []
        self.lb, self.ub = np.array(self.problem.lb), np.array(self.problem.ub)

    def __scale_cosine(self, value, max_in=100, out_values=(0.1, 0.4), k=1, reverse=True):
        x = value / max_in
        base = (1 - np.cos(np.pi * (x ** k))) / 2
        if not reverse:
            base = 1 - base
        return out_values[0] + (out_values[1] - out_values[0]) * base

    def __scale(self, value, in_values, out_values=(0, 1), reverse=False, is_inter=False):
        normalized = (value - in_values[0]) / (in_values[1] - in_values[0])

        if reverse:
            normalized = (in_values[1] - value) / (in_values[1] - in_values[0])
        normalized = normalized * (out_values[1] - out_values[0]) + out_values[0]

        if is_inter:
            normalized = int(normalized)
        return normalized

    def __select_candidates_solutions(self, itr):
        nb_candidates = self.__scale(
            itr,
            (1, self.epoch),
            (
                int(self.pop_size / 100 * 90),
                self.pop_size - 1
            ),
            reverse=False,
            is_inter=True
        )
        candidates = self.pop[:nb_candidates].copy()

        for _ in range(self.pop_size - nb_candidates):
            candidates.append(self.generate_new_agent())
        return candidates

    def __levy_steps(self, alpha):
        u = np.random.normal(0, 1, self.dimension)
        v = np.random.normal(0, 1, self.dimension)
        steps = u / (np.abs(v) ** (1 / alpha))
        steps = steps
        return steps

    def evolve(self, itr):
        solutions = []
        pos = -1
        current_best = self.pop[0]
        for p in self.__select_candidates_solutions(itr):
            pos += 1
            candidate = p.solution.copy()
            best = current_best.solution.copy()
            explore = int(self.__scale_cosine(pos, self.pop_size, (1, 10), k=self.levy_k, reverse=False))
            for _ in range(explore):
                # (1) Leader guided
                q = self.__scale_cosine(itr, self.epoch, (0.00001, 0.5), k=self.levy_k, reverse=False)
                solution = candidate + ((best - candidate) * q)

                # (2) Escape from local optimum
                if np.random.rand() < self.p_levy:
                    alpha = 1.5 + self.__scale_cosine(itr, self.epoch, (0, self.alpha_max), k=self.levy_k, reverse=True)
                    steps = self.__levy_steps(alpha)
                    solution += steps
                solution = self.generate_agent(np.array(self.amend_solution(solution)))
                solutions.append(solution)

                if solution.target.fitness < current_best.target.fitness:
                    current_best = solution

            # (3) Local search
            for d in range(self.problem.n_dims):
                q = self.__scale_cosine(itr, self.epoch, (0, 0.1), k=self.levy_k, reverse=True)
                q *= (self.problem.ub[d] - self.problem.lb[d])
                new_sol = current_best.solution.copy()
                new_sol[d] = best[d] + (np.random.rand() * 2 - 1) * q
                new_sol = self.amend_solution(new_sol)
                new_agent = self.generate_agent(new_sol)
                if new_agent.target.fitness < current_best.target.fitness:
                    current_best = new_agent
                    solutions.append(new_agent)

        self.pop += solutions
        self.pop = sorted(self.pop, key=lambda agent: agent.target.fitness)
        self.pop = self.pop[:self.pop_size]

    def generate_new_agent(self, as_array=False):
        proposal = []
        for i in range(self.dimension):
            r = self.lb[i] + np.random.uniform(0, 1) * (self.ub[i] - self.lb[i])
            proposal.append(r)
        if as_array:
            return np.array(proposal)
        return self.generate_agent(proposal)
