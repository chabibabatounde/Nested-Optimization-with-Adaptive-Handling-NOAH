import numpy as np
from mealpy.optimizer import Optimizer


class BKA(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, p=0.9, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size

        # param clé du papier (switch attaque)
        self.p = p

    # =========================
    # SAFE BEST
    # =========================
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

    # =========================
    # INIT
    # =========================
    def initialize_variables(self):
        self.pop = self.generate_population(self.pop_size)
        self.g_best = self._get_best(self.pop)

    # =========================
    # MAIN LOOP
    # =========================
    def evolve(self, epoch):

        pop_new = []

        # best
        g_best = self._get_best(self.pop)
        Xbest = g_best.solution

        # moyenne population
        pop_matrix = np.array([agent.solution for agent in self.pop])
        Xmean = np.mean(pop_matrix, axis=0)

        for i in range(self.pop_size):
            xi = self.pop[i].solution

            r = np.random.rand()

            # ==========================================
            # 1. GLOBAL EXPLORATION (soaring)
            # ==========================================
            if r > self.p:
                # vol large (exploration)
                rand_vec = np.random.randn(self.problem.n_dims)
                step = rand_vec * (xi - Xmean)

                x_new = xi + step

            # ==========================================
            # 2. ATTACK BEHAVIOR
            # ==========================================
            else:
                r2 = np.random.rand()

                # ---------- ATTACK TYPE 1 ----------
                if r2 < 0.5:
                    # attaque directe vers best
                    rand_vec = np.random.randn(self.problem.n_dims)
                    x_new = xi + rand_vec * (Xbest - abs(xi))

                # ---------- ATTACK TYPE 2 ----------
                else:
                    # attaque avec oscillation (escape local)
                    rand_vec = np.random.randn(self.problem.n_dims)
                    x_new = Xbest + rand_vec * (abs(xi - Xbest))

            # ==========================================
            # 3. LOCAL EXPLOITATION (perching)
            # ==========================================
            beta = 2 * (1 - epoch / self.epoch)  # décroissance

            local_step = beta * np.random.uniform(-1, 1, self.problem.n_dims)
            x_new = x_new + local_step * (Xbest - xi)

            # ==========================================
            # BOUNDS
            # ==========================================
            x_new = self.amend_solution(x_new)

            pop_new.append(self.generate_agent(x_new))

        # =========================
        # UPDATE
        # =========================
        self.pop = pop_new

        current_best = self._get_best(self.pop)

        if self.problem.minmax == "min":
            if current_best.target.fitness < self.g_best.target.fitness:
                self.g_best = current_best
        else:
            if current_best.target.fitness > self.g_best.target.fitness:
                self.g_best = current_best