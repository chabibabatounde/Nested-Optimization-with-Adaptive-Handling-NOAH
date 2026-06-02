import numpy as np
from mealpy.optimizer import Optimizer


class DOA(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, dream_rate=0.3, memory_rate=0.5, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size

        # paramètres DOA
        self.dream_rate = dream_rate      # exploration (rêve)
        self.memory_rate = memory_rate    # exploitation mémoire

    def _get_best(self, pop):
        best = pop[0]
        for agent in pop:
            if self.problem.minmax == "min":
                if agent.target.fitness < best.target.fitness:
                    best = agent
            else:
                if agent.target.fitness > best.target.fitness:
                    best = agent
        return best

    def initialize_variables(self):
        self.pop = self.generate_population(self.pop_size)
        self.g_best = self._get_best(self.pop)

    def evolve(self, epoch):
        pop_new = []

        for i in range(self.pop_size):
            xi = self.pop[i].solution.copy()

            # =========================
            # 🧠 DREAM PHASE (exploration)
            # =========================
            if np.random.rand() < self.dream_rate:
                # combinaison aléatoire + perturbation forte
                r1, r2 = np.random.choice(range(self.pop_size), 2, replace=False)
                x1 = self.pop[r1].solution
                x2 = self.pop[r2].solution

                alpha = np.random.uniform(-1, 1, self.problem.n_dims)
                beta = np.random.uniform(0, 1, self.problem.n_dims)

                x_new = xi + alpha * (x1 - x2) + beta * (self.g_best.solution - xi)

            # =========================
            # 🧠 MEMORY PHASE (guidage)
            # =========================
            else:
                best = self.g_best.solution
                rand_agent = self.pop[np.random.randint(self.pop_size)].solution

                w1 = np.random.rand()
                w2 = np.random.rand()

                x_new = xi + w1 * (best - xi) + w2 * (rand_agent - xi)

            # =========================
            # 🔧 REFINEMENT (local search)
            # =========================
            if np.random.rand() < self.memory_rate:
                perturb = 0.1 * (self.problem.ub - self.problem.lb)
                noise = (np.random.rand(self.problem.n_dims) * 2 - 1) * perturb
                x_new = x_new + noise

            # correction bornes
            x_new = self.amend_solution(x_new)

            agent = self.generate_agent(x_new)
            pop_new.append(agent)

        # =========================
        # ✅ SELECTION
        # =========================
        self.pop = self.greedy_selection_population(self.pop, pop_new)

        # update best
        self.g_best = self._get_best(self.pop)

        # =========================
        # 🔄 DIVERSIFICATION (rêve chaotique)
        # =========================
        if epoch % 10 == 0:
            n_restart = int(0.1 * self.pop_size)
            for i in np.random.choice(range(self.pop_size), n_restart, replace=False):
                self.pop[i] = self.generate_agent(self.problem.generate_solution())

        # =========================
        # 🔍 LOCAL SEARCH autour du best
        # =========================
        best = self.g_best.solution.copy()
        for d in range(self.problem.n_dims):
            step = 0.05 * (self.problem.ub[d] - self.problem.lb[d])
            new_sol = best.copy()
            new_sol[d] = best[d] + (np.random.rand() * 2 - 1) * step

            new_sol = self.amend_solution(new_sol)
            new_agent = self.generate_agent(new_sol)

            if self.problem.minmax == "min":
                if new_agent.target.fitness < self.g_best.target.fitness:
                    self.g_best = new_agent
            else:
                if new_agent.target.fitness > self.g_best.target.fitness:
                    self.g_best = new_agent
