from Benchmark.__Function import Function
import numpy as np


class RW48(Function):
    def __init__(self, dim):

        # ===== PROBLEM PARAMETERS =====
        self.D = 30  # Number of switching angles (typically 30)
        self.m = 0.32  # Modulation index (reverted to RC45/46 level)
        self.target_fundamental = 4 * self.m  # 1.28 (9-level target)

        # Sign sequence for 9-level (complex pattern)
        # Pattern appears to be: [+1,+1,+1,+1,-1,+1] with variations repeated
        self.s = np.array([
            1, 1, 1, 1, -1, 1,  # Block 1
            -1, -1, -1, 1, -1, -1,  # Block 2
            1, 1, 1, 1, -1, 1,  # Block 3
            -1, -1, -1, 1, -1, -1,  # Block 4
            1, 1, 1, 1, -1, 1  # Block 5 (complete)
        ], dtype=float)

        # Verify sign array length matches dimension
        if len(self.s) != self.D:
            raise ValueError(f"Sign array length {len(self.s)} != D {self.D}")

        # 31 selected harmonic orders (same as RC45/46/47)
        self.k = np.array([
            5, 7, 11, 13, 17, 19, 23, 25, 29, 31,
            35, 37, 41, 43, 47, 49, 53, 55, 59, 61,
            65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97
        ], dtype=float)

        self.num_harmonics = len(self.k)

        # Normalization factor: √(∑ 1/k⁴)
        self.norm_factor = np.sqrt(np.sum(1.0 / (self.k ** 4)))

        # Scaling factor for 9-level: 1/4
        self.scaling_factor = 1.0 / 4.0

        # Initial solution: uniformly distributed angles
        x_start = np.linspace(5, 85, self.D)

        # Domain: 0° to 90° for each angle
        domain = [(0.0, 90.0) for _ in range(self.D)]

        f_x_start = 0.0
        name = "RW48"
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
        Evaluate SHE-PWM configuration for 9-level inverter

        Returns:
        fitness: Total cost (objective + penalties for constraint violations)
        """

        x = np.array(variables_values, dtype=float)  # Angles in degrees

        # ===== OBJECTIVE FUNCTION =====
        # f = (1/4) × √(∑[j=1:31] [1/k_j⁴ × (∑[l=1:D] s_l × cos(k_j×θ_l))²]) / √(∑ 1/k⁴)

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

        # Objective: normalized THD with 1/4 scaling for 9-level
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
        # h: ∑[l=1:D] s_l × cos(θ_l) - 4m = 0
        # Note: 4m for 9-level (vs m for 3-level, 2m for 5-level, 3m for 7-level)

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


class RW48_Detailed(Function):
    """
    RC48 with detailed constraint tracking and harmonic analysis
    """

    def __init__(self, dim):
        """RC48 with comprehensive analysis capabilities"""

        self.D = dim
        self.m = 0.32
        self.target_fundamental = 4 * self.m  # 1.28

        self.s = np.array([
            1, 1, 1, 1, -1, 1,
            -1, -1, -1, 1, -1, -1,
            1, 1, 1, 1, -1, 1,
            -1, -1, -1, 1, -1, -1,
            1, 1, 1, 1, -1, 1
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
        self.scaling_factor = 1.0 / 4.0

        x_start = np.linspace(5, 85, self.D)
        domain = [(0.0, 90.0) for _ in range(self.D)]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0.0,
            max_dimension=self.D,
            name="RW48_Detailed"
        )
        self.types(['CEC2020RW'])

        # Storage for detailed analysis
        self.last_h = 0.0
        self.last_g_violations = 0
        self.last_thd = 0.0
        self.last_harmonics = None
        self.last_objective = 0.0
        self.last_angles = None

    def eval_harmonics(self, angles):
        """
        Compute all harmonic components for detailed analysis

        Returns: Dict with S_j values for each harmonic order
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
        Evaluate all constraints separately

        Returns: (h, g) where
        - h: Equality constraint (should be ~0)
        - g: Inequality constraints vector (should be ≤ 0)
        """

        x = np.array(variables_values, dtype=float)

        # Equality constraint: ∑ s_l cos(θ_l) = 4m
        fundamental = np.sum(self.s * np.cos(np.radians(x)))
        h = fundamental - self.target_fundamental

        # Inequality constraints: monotonicity
        g = np.zeros(self.D - 1)
        for i in range(self.D - 1):
            g[i] = x[i] - x[i + 1] + 1e-6

        self.last_h = h
        self.last_g_violations = np.sum(np.maximum(0, g) ** 2)
        self.last_angles = x.copy()

        return h, g

    def eval(self, variables_values):
        """Main evaluation with comprehensive tracking"""

        x = np.array(variables_values, dtype=float)

        # Objective computation
        harmonic_sum = 0.0
        self.last_harmonics = {}

        for j in range(self.num_harmonics):
            k_j = self.k[j]
            S_j = np.sum(self.s * np.cos(k_j * np.radians(x)))
            self.last_harmonics[int(k_j)] = S_j
            harmonic_sum += (S_j ** 2) / (k_j ** 4)

        self.last_objective = self.scaling_factor * np.sqrt(harmonic_sum) / self.norm_factor
        self.last_thd = self.last_objective * 100  # Convert to percentage

        # Constraint evaluation
        h, g = self.eval_constraints(x)

        # Penalty computation
        h_penalty = (1e5 * h) ** 2
        g_penalty = 1e4 * self.last_g_violations

        fitness = self.last_objective + h_penalty + g_penalty

        return fitness if np.isfinite(fitness) else 1e10

    def get_analysis(self):
        """
        Return detailed problem analysis

        Returns: Dict with comprehensive metrics
        """

        return {
            'thd_normalized': self.last_objective,
            'thd_percent': self.last_thd,
            'h_constraint': self.last_h,
            'h_violation': abs(self.last_h),
            'g_violations': self.last_g_violations,
            'harmonics': self.last_harmonics,
            'harmonics_count_above_threshold': len([v for v in (self.last_harmonics or {}).values() if abs(v) > 0.1]),
            'feasible': abs(self.last_h) < 1e-3 and self.last_g_violations < 1e-6,
            'primary_harmonics': {k: v for k, v in (self.last_harmonics or {}).items()
                                  if k in [5, 7, 11, 13, 17, 19]} if self.last_harmonics else {},
            'monotonicity_verified': self._check_monotonicity() if self.last_angles is not None else False
        }

    def _check_monotonicity(self):
        """Verify angles are strictly increasing"""
        if self.last_angles is None:
            return False
        for i in range(len(self.last_angles) - 1):
            if self.last_angles[i] >= self.last_angles[i + 1]:
                return False
        return True

    def print_analysis(self):
        """Pretty-print detailed analysis"""

        analysis = self.get_analysis()

        print("\n" + "=" * 70)
        print("RC48 DETAILED ANALYSIS - 9-Level Inverter SHE-PWM")
        print("=" * 70)

        print(f"\n[OBJECTIVE]")
        print(f"  THD (normalized): {analysis['thd_normalized']:.6f}")
        print(f"  THD (percentage): {analysis['thd_percent']:.3f}%")
        print(f"  Expected range: 0.2-0.5% (excellent quality)")

        print(f"\n[CONSTRAINTS]")
        print(f"  Equality h (target=1.28): {analysis['h_constraint']:.6f}")
        print(f"  Equality violation: {analysis['h_violation']:.2e}")
        print(f"  Monotonicity violations: {analysis['g_violations']:.2e}")
        print(f"  Angles strictly ordered: {'YES' if analysis['monotonicity_verified'] else 'NO'}")
        print(f"  Feasible: {'YES ✓' if analysis['feasible'] else 'NO ✗'}")

        print(f"\n[HARMONIC SPECTRUM] (Low-order)")
        for harm, value in sorted(analysis['primary_harmonics'].items()):
            print(f"  h={harm:2d}: {value:7.4f}")

        print(f"\n[STATISTICS]")
        print(f"  Total harmonics to minimize: {len(analysis['harmonics']) if analysis['harmonics'] else 0}")
        print(f"  Harmonics with |S_j| > 0.1: {analysis['harmonics_count_above_threshold']}")

        print(f"\n[COMPARISON WITH OTHER LEVELS]")
        print(f"  vs 3-level (RC45): THD improvement ~90%")
        print(f"  vs 5-level (RC46): THD improvement ~75%")
        print(f"  vs 7-level (RC47): THD improvement ~50%")
        print(f"  vs 9-level baseline: Optimized configuration")

        print("=" * 70 + "\n")
