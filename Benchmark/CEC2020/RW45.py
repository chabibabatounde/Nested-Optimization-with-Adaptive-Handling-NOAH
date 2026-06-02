from Benchmark.__Function import Function
import numpy as np


class RW45(Function):
    def __init__(self, dim):

        # ===== PROBLEM PARAMETERS =====
        self.D = dim  # Number of switching angles (typically 25-31)
        self.m = 0.32  # Modulation index (fundamental voltage target)

        # Sign sequence: s_k = (-1)^(k+1)
        # s = [+1, -1, +1, -1, +1, -1, ...]
        self.s = np.array([(-1) ** (k + 1) for k in range(1, self.D + 1)], dtype=float)

        # 31 selected harmonic orders (exclude 1, 3, 9, triplen)
        self.k = np.array([
            5, 7, 11, 13, 17, 19, 23, 25, 29, 31,
            35, 37, 41, 43, 47, 49, 53, 55, 59, 61,
            65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97
        ], dtype=float)

        self.num_harmonics = len(self.k)

        # Normalization factor: √(∑ 1/k⁴)
        self.norm_factor = np.sqrt(np.sum(1.0 / (self.k ** 4)))

        # Initial solution: uniformly distributed angles
        x_start = np.linspace(5, 85, self.D)

        # Domain: 0° to 90° for each angle
        domain = [(0.0, 90.0) for _ in range(self.D)]

        f_x_start = 0.0  # Will be computed
        name = "RW45"
        max_dimension = self.D

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            f_x_start,
            max_dimension,
            name
        )
        self.types(['CEC2020RW'])

    def eval(self, variables_values):
        """
        Evaluate SHE-PWM configuration

        Returns:
        f: Objective (THD minimization)
        g: Inequality constraints (monotonicity)
        h: Equality constraint (fundamental voltage)
        """

        x = np.array(variables_values, dtype=float)  # Angles in degrees

        # ===== OBJECTIVE FUNCTION =====
        # f = √(∑[j=1:31] [1/k_j⁴ × (∑[l=1:D] s_l × cos(k_j×θ_l))²]) / √(∑ 1/k⁴)

        harmonic_sum = 0.0

        for j in range(self.num_harmonics):
            k_j = self.k[j]

            # S_j = ∑[l=1:D] s_l × cos(k_j × θ_l)
            harmonic_content = 0.0

            for l in range(self.D):
                angle_rad = np.radians(x[l])
                harmonic_content += self.s[l] * np.cos(k_j * angle_rad)

            # Weighted harmonic: (S_j / k_j²)²
            weighted_harmonic = (harmonic_content ** 2) / (k_j ** 4)
            harmonic_sum += weighted_harmonic

        # Objective: normalized THD
        objective = np.sqrt(harmonic_sum) / self.norm_factor

        # ===== INEQUALITY CONSTRAINTS (Monotonicity) =====
        # g_i: θ_i - θ_{i+1} + ε ≤ 0
        # Ensures: θ_1 < θ_2 < ... < θ_D

        g = np.zeros(self.D - 1)
        epsilon = 1e-6

        for i in range(self.D - 1):
            g[i] = x[i] - x[i + 1] + epsilon

        # Penalty for violated inequality constraints
        g_penalty = 0.0
        for i in range(self.D - 1):
            if g[i] > 0:
                g_penalty += (1e4 * g[i]) ** 2

        # ===== EQUALITY CONSTRAINT (Fundamental Voltage) =====
        # h: ∑[l=1:D] s_l × cos(θ_l) - m = 0

        fundamental = 0.0
        for l in range(self.D):
            angle_rad = np.radians(x[l])
            fundamental += self.s[l] * np.cos(angle_rad)

        h = fundamental - self.m

        # Penalty for violated equality constraint
        h_penalty = (1e5 * h) ** 2

        # ===== TOTAL FITNESS =====
        fitness = objective + g_penalty + h_penalty

        return fitness if np.isfinite(fitness) else 1e10


class RW45_Constrained(Function):
    """
    Alternative formulation with explicit constraint handling
    """

    def __init__(self, dim):
        """RC45 with constraint information tracking"""

        self.D = dim
        self.m = 0.32
        self.s = np.array([(-1) ** (k + 1) for k in range(1, self.D + 1)], dtype=float)
        self.k = np.array([
            5, 7, 11, 13, 17, 19, 23, 25, 29, 31,
            35, 37, 41, 43, 47, 49, 53, 55, 59, 61,
            65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97
        ], dtype=float)

        self.num_harmonics = len(self.k)
        self.norm_factor = np.sqrt(np.sum(1.0 / (self.k ** 4)))

        # Initial solution
        x_start = np.linspace(5, 85, self.D)

        domain = [(0.0, 90.0) for _ in range(self.D)]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0.0,
            max_dimension=self.D,
            name="RW45_Constrained"
        )
        self.types(['CEC2020RW'])

        # Storage for constraint violations
        self.last_h = 0.0
        self.last_g_violations = 0

    def eval_constraints(self, variables_values):
        """
        Evaluate constraints separately

        Returns: (h, g) where
        - h: Equality constraint (should be ~0)
        - g: Inequality constraints vector (should be ≤ 0)
        """

        x = np.array(variables_values, dtype=float)

        # Equality constraint
        fundamental = np.sum(self.s * np.cos(np.radians(x)))
        h = fundamental - self.m

        # Inequality constraints
        g = np.zeros(self.D - 1)
        for i in range(self.D - 1):
            g[i] = x[i] - x[i + 1] + 1e-6

        self.last_h = h
        self.last_g_violations = np.sum(np.maximum(0, g) ** 2)

        return h, g

    def eval(self, variables_values):
        """Main evaluation with constraint handling"""

        x = np.array(variables_values, dtype=float)

        # Objective
        harmonic_sum = 0.0
        for j in range(self.num_harmonics):
            k_j = self.k[j]
            S_j = np.sum(self.s * np.cos(k_j * np.radians(x)))
            harmonic_sum += (S_j ** 2) / (k_j ** 4)

        objective = np.sqrt(harmonic_sum) / self.norm_factor

        # Constraints
        h, g = self.eval_constraints(x)

        # Penalties
        h_penalty = (1e5 * h) ** 2
        g_penalty = 1e4 * self.last_g_violations

        fitness = objective + h_penalty + g_penalty

        return fitness if np.isfinite(fitness) else 1e10
