from Benchmark.__Function import Function
import numpy as np


class RW46(Function):
    def __init__(self, dim):

        # ===== PROBLEM PARAMETERS =====
        self.D = 25  # Number of switching angles (typically 25)
        self.m = 0.32  # Modulation index
        self.target_fundamental = 2 * self.m  # 0.64 (5-level target)

        # Sign sequence for 5-level (explicit, not simple pattern)
        self.s = np.array([
            1, -1, 1, 1, -1, 1, -1, 1, -1, -1,
            1, -1, 1, 1, -1, 1, -1, 1, -1, -1,
            1, -1, 1, 1, -1
        ], dtype=float)

        # Verify sign array length matches dimension
        if len(self.s) != self.D:
            raise ValueError(f"Sign array length {len(self.s)} != D {self.D}")

        # 31 selected harmonic orders (same as RC45)
        self.k = np.array([
            5, 7, 11, 13, 17, 19, 23, 25, 29, 31,
            35, 37, 41, 43, 47, 49, 53, 55, 59, 61,
            65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97
        ], dtype=float)

        self.num_harmonics = len(self.k)

        # Normalization factor: √(∑ 1/k⁴)
        self.norm_factor = np.sqrt(np.sum(1.0 / (self.k ** 4)))

        # Scaling factor for 5-level
        self.scaling_factor = 0.5

        # Initial solution: uniformly distributed angles
        x_start = np.linspace(5, 85, self.D)

        # Domain: 0° to 90° for each angle
        domain = [(0.0, 90.0) for _ in range(self.D)]

        f_x_start = 0.0
        name = "RW46"
        max_dimension = 25

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
        Evaluate SHE-PWM configuration for 5-level inverter

        Returns:
        fitness: Total cost (objective + penalties for constraint violations)
        """

        x = np.array(variables_values, dtype=float)  # Angles in degrees

        # ===== OBJECTIVE FUNCTION =====
        # f = 0.5 × √(∑[j=1:31] [1/k_j⁴ × (∑[l=1:D] s_l × cos(k_j×θ_l))²]) / √(∑ 1/k⁴)

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

        # Objective: normalized THD with 0.5 scaling for 5-level
        objective = self.scaling_factor * np.sqrt(harmonic_sum) / self.norm_factor

        # ===== INEQUALITY CONSTRAINTS (Monotonicity) =====
        # g_i: θ_i - θ_{i+1} + ε ≤ 0
        # Ensures: θ_1 < θ_2 < ... < θ_D

        g_penalty = 0.0
        epsilon = 1e-6

        for i in range(self.D - 1):
            violation = x[i] - x[i + 1] + epsilon
            if violation > 0:
                g_penalty += (1e4 * violation) ** 2

        # ===== EQUALITY CONSTRAINT (Fundamental Voltage) =====
        # h: ∑[l=1:D] s_l × cos(θ_l) - 2m = 0
        # Note: 2m for 5-level (vs m for 3-level)

        fundamental = 0.0
        for l in range(self.D):
            angle_rad = np.radians(x[l])
            fundamental += self.s[l] * np.cos(angle_rad)

        h = fundamental - self.target_fundamental

        # Penalty for violated equality constraint
        h_penalty = (1e5 * h) ** 2

        # ===== TOTAL FITNESS =====
        fitness = objective + g_penalty + h_penalty

        return fitness if np.isfinite(fitness) else 1e10


class RW46_Detailed(Function):
    """
    RC46 with detailed constraint tracking and analysis
    """

    def __init__(self, dim):
        """RC46 with constraint information"""

        self.D = dim
        self.m = 0.32
        self.target_fundamental = 2 * self.m

        self.s = np.array([
            1, -1, 1, 1, -1, 1, -1, 1, -1, -1,
            1, -1, 1, 1, -1, 1, -1, 1, -1, -1,
            1, -1, 1, 1, -1
        ], dtype=float)

        if len(self.s) != self.D:
            raise ValueError(f"Sign array length {len(self.s)} != D {self.D}")

        self.k = np.array([
            5, 7, 11, 13, 17, 19, 23, 25, 29, 31,
            35, 37, 41, 43, 47, 49, 53, 55, 59, 61,
            65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97
        ], dtype=float)

        self.num_harmonics = len(self.k)
        self.norm_factor = np.sqrt(np.sum(1.0 / (self.k ** 4)))
        self.scaling_factor = 0.5

        x_start = np.linspace(5, 85, self.D)
        domain = [(0.0, 90.0) for _ in range(self.D)]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0.0,
            max_dimension=self.D,
            name="RW46_Detailed"
        )
        self.types(['CEC2020RW'])

        # Storage for constraint analysis
        self.last_h = 0.0
        self.last_g_violations = 0
        self.last_thd = 0.0
        self.last_harmonics = None

    def eval_harmonics(self, angles):
        """
        Compute all harmonic components

        Returns: Dict with S_j values
        """

        x = np.array(angles, dtype=float)
        harmonics = {}

        for j in range(self.num_harmonics):
            k_j = self.k[j]
            S_j = np.sum(self.s * np.cos(k_j * np.radians(x)))
            harmonics[int(k_j)] = S_j

        return harmonics

    def eval_constraints(self, variables_values):
        """
        Evaluate constraints separately

        Returns: (h, g) where
        - h: Equality constraint (should be ~0)
        - g: Inequality constraints vector (should be ≤ 0)
        """

        x = np.array(variables_values, dtype=float)

        # Equality constraint: ∑ s_l cos(θ_l) = 2m
        fundamental = np.sum(self.s * np.cos(np.radians(x)))
        h = fundamental - self.target_fundamental

        # Inequality constraints: monotonicity
        g = np.zeros(self.D - 1)
        for i in range(self.D - 1):
            g[i] = x[i] - x[i + 1] + 1e-6

        self.last_h = h
        self.last_g_violations = np.sum(np.maximum(0, g) ** 2)

        return h, g

    def eval(self, variables_values):
        """Main evaluation with detailed tracking"""

        x = np.array(variables_values, dtype=float)

        # Objective
        harmonic_sum = 0.0
        self.last_harmonics = {}

        for j in range(self.num_harmonics):
            k_j = self.k[j]
            S_j = np.sum(self.s * np.cos(k_j * np.radians(x)))
            self.last_harmonics[int(k_j)] = S_j
            harmonic_sum += (S_j ** 2) / (k_j ** 4)

        self.last_thd = self.scaling_factor * np.sqrt(harmonic_sum) / self.norm_factor

        # Constraints
        h, g = self.eval_constraints(x)

        # Penalties
        h_penalty = (1e5 * h) ** 2
        g_penalty = 1e4 * self.last_g_violations

        fitness = self.last_thd + h_penalty + g_penalty

        return fitness if np.isfinite(fitness) else 1e10

    def get_analysis(self):
        """Return detailed problem analysis"""
        return {
            'thd': self.last_thd,
            'h_constraint': self.last_h,
            'g_violations': self.last_g_violations,
            'harmonics': self.last_harmonics,
            'feasible': abs(self.last_h) < 1e-3 and self.last_g_violations < 1e-6
        }
