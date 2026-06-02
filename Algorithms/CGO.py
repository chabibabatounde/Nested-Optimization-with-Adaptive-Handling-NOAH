import numpy as np
from mealpy.optimizer import Optimizer


class CGO(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size

        # paramètres papier
        self.vmax = 1.0
        self.vmin = 0.15
        self.b = 1
        self.alpha = 0.2
        self.Dis = 0.1

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

    # ---------- INIT ----------
    def initialize_variables(self):
        self.pop = self.generate_population(self.pop_size)
        self.g_best = self._get_best(self.pop)

    # ---------- MAIN ----------
    def evolve(self, epoch):

        pop_old = self.pop.copy()

        # === tri par fitness
        pop_old.sort(key=lambda a: a.target.fitness)

        best = pop_old[0]
        Xbest = best.solution

        pop_matrix = np.array([a.solution for a in pop_old])
        Xcent = np.mean(pop_matrix, axis=0)

        # === vitesse (Eq.2)
        V = self.vmax + (self.vmin - self.vmax) / (
            1 + np.exp(-10 * self.b * (2 * epoch / self.epoch - 1))
        )

        # === split population
        Ns = int(0.618 * self.pop_size)
        Ng = self.pop_size - Ns

        pop_new = []

        # =====================================================
        # 1. GROWING
        # =====================================================
        for i in range(Ng):
            xi = pop_old[i].solution

            g1 = Xbest - xi
            g2 = xi - Xcent

            r1 = np.random.rand()
            GR = np.random.randn(self.problem.n_dims)

            x_new = xi + V * GR * (r1 * g1 + (1 - r1) * g2)

            # ---------- Repulsion ----------
            rep_force = np.zeros_like(xi)

            for j in range(self.pop_size):
                if j == i:
                    continue
                dist = np.linalg.norm(x_new - pop_old[j].solution)
                if dist < self.Dis:
                    rep_force += (x_new - pop_old[j].solution)

            if np.linalg.norm(rep_force) > 0:
                rep_force = self.alpha * rep_force

            x_new = x_new + rep_force
            x_new = self.amend_solution(x_new)

            pop_new.append(self.generate_agent(x_new))

        # =====================================================
        # 2. ELITE POOL
        # =====================================================
        half = int(self.pop_size / 2)
        Xhalf = np.mean(pop_matrix[:half], axis=0)

        elite_pool = [
            pop_old[0].solution,
            pop_old[1].solution,
            pop_old[2].solution,
            Xhalf,
            Xcent
        ]

        # =====================================================
        # 3. SPROUTING
        # =====================================================
        for i in range(Ng, self.pop_size):
            xi = pop_old[i].solution

            elite = elite_pool[np.random.randint(len(elite_pool))]

            s1 = Xbest - xi
            s2 = xi - Xcent

            r2 = np.random.rand()
            GR = np.random.randn(self.problem.n_dims)

            x_new = elite + GR * (r2 * s1 + (1 - r2) * s2)

            x_new = self.amend_solution(x_new)
            pop_new.append(self.generate_agent(x_new))

        # =====================================================
        # 4. PRUNING
        # =====================================================
        tcut = int(self.epoch / self.pop_size) + 1

        if epoch % tcut == 0:
            n_prune = int(0.382 * self.pop_size)

            pop_new.sort(key=lambda a: a.target.fitness, reverse=True)

            for i in range(n_prune):
                rand_sol = np.random.uniform(self.problem.lb, self.problem.ub)
                pop_new[i] = self.generate_agent(rand_sol)

        # =====================================================
        # UPDATE
        # =====================================================
        self.pop = pop_new

        current_best = self._get_best(self.pop)

        if self.problem.minmax == "min":
            if current_best.target.fitness < self.g_best.target.fitness:
                self.g_best = current_best
        else:
            if current_best.target.fitness > self.g_best.target.fitness:
                self.g_best = current_best
