import numpy as np
from mealpy.optimizer import Optimizer


class SequoiaOA(Optimizer):
    def __init__(self, epoch=1000, pop_size=50, elite_size=2, **kwargs):
        super().__init__(**kwargs)
        np.random.seed(self.seed)

        self.epoch = epoch
        self.pop_size = pop_size
        self.elite_size = elite_size

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

    # ---------- MAIN LOOP ----------
    def evolve(self, epoch):

        # === paramètres dynamiques (paper Eq.5 + Eq.8)
        fire_prob = max(0.3 - 0.15 * (epoch / self.epoch), 0.1)
        mutation_rate = max(0.2 - 0.1 * (epoch / self.epoch), 0.02)

        # === tri population (élitisme)
        self.pop.sort(key=lambda agent: agent.target.fitness)
        elites = [agent.solution.copy() for agent in self.pop[:self.elite_size]]

        pop_matrix = np.array([agent.solution for agent in self.pop])

        # =========================================================
        # 1. Collective growth (Eq.4)
        # =========================================================
        half = int(self.pop_size / 2)
        mean_top = np.mean(pop_matrix[:half], axis=0)

        pop_matrix = pop_matrix + np.random.randn(*pop_matrix.shape) * (mean_top - pop_matrix)

        # =========================================================
        # 2. Fire disturbance (Eq.5)
        # =========================================================
        if np.random.rand() < fire_prob:
            pop_matrix = pop_matrix + np.random.randn(*pop_matrix.shape) * 0.5

        # =========================================================
        # 3. Reproduction (crossover + mutation)
        # =========================================================
        new_pop = []

        for i in range(0, self.pop_size, 2):
            if i + 1 >= self.pop_size:
                break

            p1 = pop_matrix[i]
            p2 = pop_matrix[i + 1]

            alpha = np.random.rand()

            # crossover
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = alpha * p2 + (1 - alpha) * p1

            # mutation
            if np.random.rand() < mutation_rate:
                child1 += np.random.randn(self.problem.n_dims) * 0.3
                child2 += np.random.randn(self.problem.n_dims) * 0.3

            # bounds
            child1 = self.amend_solution(child1)
            child2 = self.amend_solution(child2)

            new_pop.append(self.generate_agent(child1))
            new_pop.append(self.generate_agent(child2))

        # si taille impaire
        while len(new_pop) < self.pop_size:
            idx = np.random.randint(0, self.pop_size)
            new_pop.append(self.generate_agent(self.amend_solution(pop_matrix[idx])))

        self.pop = new_pop

        # =========================================================
        # 4. Local search (Eq.9)
        # =========================================================
        best = self.g_best.solution.copy()
        local = best + 0.1 * np.random.randn(self.problem.n_dims)
        local = self.amend_solution(local)

        local_agent = self.generate_agent(local)

        if self.problem.minmax == "min":
            if local_agent.target.fitness < self.g_best.target.fitness:
                self.g_best = local_agent
        else:
            if local_agent.target.fitness > self.g_best.target.fitness:
                self.g_best = local_agent

        # =========================================================
        # 5. Elite preservation
        # =========================================================
        self.pop.sort(key=lambda agent: agent.target.fitness, reverse=False)

        for i in range(self.elite_size):
            self.pop[-(i + 1)] = self.generate_agent(elites[i])

        # =========================================================
        # 6. Update global best
        # =========================================================
        current_best = self._get_best(self.pop)

        if self.problem.minmax == "min":
            if current_best.target.fitness < self.g_best.target.fitness:
                self.g_best = current_best
        else:
            if current_best.target.fitness > self.g_best.target.fitness:
                self.g_best = current_best
