import numpy as np
from mealpy.optimizer import Optimizer


class DBO(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size

        # paramètres recommandés (papier)
        self.k = 0.1
        self.b = 0.3
        self.S = 0.5

    # ---------- SAFE BEST ----------
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

    def _get_worst(self, pop):
        worst = pop[0]
        for agent in pop:
            if self.problem.minmax == "min":
                if agent.target.fitness > worst.target.fitness:
                    worst = agent
            else:
                if agent.target.fitness < worst.target.fitness:
                    worst = agent
        return worst

    # ---------- INIT ----------
    def initialize_variables(self):
        self.pop = self.generate_population(self.pop_size)
        self.g_best = self._get_best(self.pop)

    # ---------- MAIN ----------
    def evolve(self, epoch):

        pop_new = []
        pop_old = self.pop.copy()

        best = self._get_best(pop_old)
        worst = self._get_worst(pop_old)

        Xb = best.solution
        Xw = worst.solution

        R = 1 - epoch / self.epoch

        for i in range(self.pop_size):
            xi = pop_old[i].solution.copy()

            # =====================================================
            # 1. BALL-ROLLING (Eq.1)
            # =====================================================
            if i < int(0.25 * self.pop_size):

                if epoch > 1:
                    x_prev = pop_old[np.random.randint(self.pop_size)].solution
                else:
                    x_prev = xi

                delta_x = np.abs(xi - Xw)
                direction = 1 if np.random.rand() < 0.5 else -1

                x_new = xi + direction * self.k * (xi - x_prev) + self.b * delta_x

            # =====================================================
            # 2. BROOD (Eq.3-4)
            # =====================================================
            elif i < int(0.5 * self.pop_size):

                lb_star = np.maximum(Xb * (1 - R), self.problem.lb)
                ub_star = np.minimum(Xb * (1 + R), self.problem.ub)

                b1 = np.random.rand(self.problem.n_dims)
                b2 = np.random.rand(self.problem.n_dims)

                x_new = Xb + b1 * (xi - lb_star) + b2 * (xi - ub_star)

            # =====================================================
            # 3. FORAGING (Eq.5-6)
            # =====================================================
            elif i < int(0.75 * self.pop_size):

                lb_b = np.maximum(Xb * (1 - R), self.problem.lb)
                ub_b = np.minimum(Xb * (1 + R), self.problem.ub)

                C1 = np.random.randn()
                C2 = np.random.rand(self.problem.n_dims)

                x_new = xi + C1 * (xi - lb_b) + C2 * (xi - ub_b)

            # =====================================================
            # 4. THIEF (Eq.7)
            # =====================================================
            else:
                g = np.random.randn(self.problem.n_dims)
                x_new = Xb + self.S * g * (np.abs(xi - Xb) + np.abs(xi - Xb))

            # =====================================================
            # BOUNDS
            # =====================================================
            x_new = self.amend_solution(x_new)

            pop_new.append(self.generate_agent(x_new))

        # update population
        self.pop = pop_new

        # update best
        current_best = self._get_best(self.pop)

        if self.problem.minmax == "min":
            if current_best.target.fitness < self.g_best.target.fitness:
                self.g_best = current_best
        else:
            if current_best.target.fitness > self.g_best.target.fitness:
                self.g_best = current_best
