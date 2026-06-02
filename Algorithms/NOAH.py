import numpy as np
from mealpy import Optimizer


# NOAH: Nested Optimization with Adaptive Handling
class Noah(Optimizer):
    """
        NOAH: Nested Optimization with Adaptive Handling
        A population-based metaheuristic that combines:
          - Leader-guided exploration (attraction toward current best)
          - Lévy-flight escape mechanism (with cosine-modulated alpha)
          - Dimensional local search (partial coordinate perturbation)
          - Adaptive handling via:
              * Cosine modulation of n_candidates and n_explore
              * Elitist truncation
        Gannet Optimization Algorithm - 2024
        Growth Optimizer - 2024
        Orchid Optimization Algorithm - 2025
        Sea Horse Optimizer - 2025


        Teamwork Optimization Algorithm - 2024
        Nurse Optimization Algorithm - 2024
        Chef-Based Optimization - 2024

        Koala Optimization Algorithm - 2025
        Sea Horse Optimizer - 2025
    """

    def __init__(
            self,
            epoch=100,  # Maximum generation
            pop_size=100,  # Population size
            omega=5,
            **kwargs
    ):
        super().__init__(**kwargs)
        np.random.seed(self.seed)
        # Main parameters
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.p_levy = 0.1

        self.omega = omega

        # Internal parameters (Computational variable)
        self.__ax = None
        self.__epsilon = 1e-10
        self.__rho_fuel = None
        self.__rho_air = None
        self.dimension = None
        self.lb = None
        self.ub = None

        # --- Initialization -----------------------------------------------------

    def initialize_variables(self):
        self.dimension = len(self.problem.ub)
        self.__ax = 5
        self.history.ax = []
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

    def ___select_candidates_solutions(self, itr, p_min):
        nb_candidates = self.__scale(
            itr,
            (1, self.epoch),
            (
                int(self.pop_size / 100 * p_min),
                self.pop_size - 1
            ),
            reverse=False,
            is_inter=True
        )
        candidates = self.pop[1:nb_candidates + 1].copy()

        for _ in range(self.pop_size - nb_candidates):
            candidates.append(self.generate_new_agent())
        return candidates

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

    def __ratio(self, itr, r_min=0.1, r_max=0.2):
        return r_min + (r_max - r_min) * (1 + np.cos(np.pi * itr / self.epoch)) / 2

    '''
    def differential_mutation(self, population, i, F=0.5):
        pop_size = len(population)
        indices = list(range(pop_size))
        indices.remove(i)
        r1, r2, r3 = np.random.choice(indices, size=3, replace=False)
        Xr1 = population[r1].solution.copy()
        Xr2 = population[r2].solution.copy()
        Xr3 = population[r3].solution.copy()
        Vi = Xr1 + F * (Xr2 - Xr3)
        return Vi
    '''

    def evolve(self, itr):
        p_levy = 0.2
        levy_k = 3
        p_min = 90
        alpha_max = 0.4

        solutions = []
        pos = -1
        current_best = self.pop[0]
        # for p in self.pop[1:]:
        for p in self.__select_candidates_solutions(itr):
            pos += 1
            candidate = p.solution.copy()
            best = current_best.solution.copy()
            explore = int(self.__scale_cosine(pos, self.pop_size, (1, 10), k=levy_k, reverse=False))
            for _ in range(explore):
                # (1) Leader guided
                q = self.__scale_cosine(itr, self.epoch, (0.00001, 0.5), k=levy_k, reverse=False)
                solution = candidate + ((best - candidate) * q)

                '''
                F = self.__scale_cosine(itr, self.epoch, (0.2, 0.9), k=levy_k, reverse=False)
                solution = self.differential_mutation(self.pop, pos, F=F)
                '''

                # (2) Escape from local optimum
                if np.random.rand() < p_levy:
                    alpha = 1.5 + self.__scale_cosine(itr, self.epoch, (0, alpha_max), k=levy_k, reverse=True)
                    steps = self.__levy_steps(alpha)
                    solution += steps
                solution = self.generate_agent(np.array(self.amend_solution(solution)))
                solutions.append(solution)

                if solution.target.fitness < current_best.target.fitness:
                    current_best = solution

            # (3) Local search
            for d in range(self.problem.n_dims):

                q = 0.1 * (self.problem.ub[d] - self.problem.lb[d])

                q = self.__scale_cosine(itr, self.epoch, (0, 0.1), k=levy_k, reverse=True)
                q *= (self.problem.ub[d] - self.problem.lb[d])

                new_sol = current_best.solution.copy()
                new_sol[d] = best[d] + (np.random.rand() * 2 - 1) * q
                new_sol = self.amend_solution(new_sol)
                new_agent = self.generate_agent(new_sol)
                if new_agent.target.fitness < current_best.target.fitness:
                    current_best = new_agent
                    solutions.append(new_agent)

                '''
                        if epoch % 10 == 0:
                            n_restart = int(0.1 * self.pop_size)
                            for i in np.random.choice(range(self.pop_size), n_restart, replace=False):
                                self.pop[i] = self.generate_agent(self.problem.generate_solution())
                
                
                partial = self.generate_new_agent(as_array=True)
                n = max(1, int(len(best) * self.__scale_cosine(itr, self.epoch, (0.1, 1), 4, reverse=False) / 2))
                index = np.random.choice(best.shape[0], size=n, replace=False)
                print(n)
                solution[index] = (1 - p_best * partial[:n]) + (p_best * current_best.solution[index])

                # (3) Local search
                n = max(1, int(len(solution) * self.__scale_cosine(itr, self.epoch, (0.1, 1), 4, reverse=False) / 2))
                index = np.random.choice(solution.shape[0], size=n, replace=False)
                best = current_best.solution.copy()
                if np.random.uniform(0, 1) > 1.5:
                    pass
                else:
                    partial = self.generate_new_agent(as_array=True)
                    best[index] = (1 - p_best * partial[:n]) + (p_best * best[index])
                best = self.generate_agent(best)
                solutions.append(best)
                if np.random.uniform(0, 1) > 1.5:
                    perturb = 0.1 * (self.domain - self.problem.lb[d])
                    new_sol = best.copy()
                    new_sol[d] = best[d] + (np.random.rand() * 2 - 1) * perturb
                    new_sol = self.amend_solution(new_sol)
                    new_agent = self.generate_agent(new_sol)
                    partial = self.generate_new_agent(as_array=True)
                    solution[index] = (1 - p_best * partial[:n]) + (p_best * current_best.solution[index])
                else:
                    phi = (1 + np.sqrt(5)) / 2
                    w = (0.5 + np.random.rand() / 2) * phi
                    solution = w * current_best.solution + (1 - w)
                    for d in range(len(solution)):
                        q = np.random.choice([-1, 1]) * np.random.uniform(0.0001, 1)
                        q = 0.1 * (self.ub[d] - self.lb[d]) * np.random.choice([-1, 1])
                        solution[d] = current_best.solution[d] + q
                '''
        self.pop += solutions
        self.pop = sorted(self.pop, key=lambda agent: agent.target.fitness)
        self.pop = self.pop[:self.pop_size]

    # --- Main steps -----------------------------------------------------------

    def __ecu_delta_f(self):
        size = len(self.history.list_current_best_fit)
        ax = self.__ax
        if size >= self.omega:
            window = self.history.list_current_best_fit[-self.omega:]
            improvement = np.mean([
                abs(window[i] - window[i - 1]) / (abs(window[i - 1]) + self.__epsilon)
                for i in range(1, len(window))
            ]) * 100

            if improvement >= 75:
                ax -= 1
            elif 25 <= improvement <= 75:
                ax = 5
            else:
                ax += 1
        return int(np.clip(ax, 1, 10))

    def generate_new_agent(self, as_array=False):
        # (0) Random generation
        proposal = []
        for i in range(self.dimension):
            r = self.lb[i] + np.random.uniform(0, 1) * (self.ub[i] - self.lb[i])
            proposal.append(r)
        if as_array:
            return np.array(proposal)
        return self.generate_agent(proposal)
