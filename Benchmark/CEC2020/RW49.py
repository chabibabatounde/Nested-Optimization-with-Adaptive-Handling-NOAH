from Benchmark.__Function import Function
import numpy as np


class RW49(Function):
    def __init__(self, dim):

        # ===== PROBLEM PARAMETERS =====
        self.D = 30  # Number of switching angles (30 for 11-level)
        self.m = 1.0 / 3.0  # Modulation index ≈ 0.3333
        self.target_fundamental = 5 * self.m  # 5/3 ≈ 1.6667

        # Sign sequence for 11-level (complex, topology-specific pattern)
        # 30 elements with hybrid structure (no simple formula)
        self.s = np.array([
            1, -1, 1, 1, 1, -1,  # Block 1 (different from RC48)
            -1, -1, 1, 1, 1, 1,  # Block 2
            -1, -1, 1, -1, -1, -1,  # Block 3
            1, 1, 1, 1, -1, 1,  # Block 4
            1, -1, -1, 1, -1, -1  # Block 5 (incomplete pattern)
        ], dtype=float)

        if len(self.s) != self.D:
            raise ValueError(f"Sign array length {len(self.s)} != D {self.D}")

        # 31 selected harmonic orders (same as all multilevel)
        self.k = np.array([
            5, 7, 11, 13, 17, 19, 23, 25, 29, 31,
            35, 37, 41, 43, 47, 49, 53, 55, 59, 61,
            65, 67, 71, 73, 77, 79, 83, 85, 91, 95, 97
        ], dtype=float)

        self.num_harmonics = len(self.k)

        # Normalization factor
        self.norm_factor = np.sqrt(np.sum(1.0 / (self.k ** 4)))

        # Scaling factor for 11-level: 1/5 = 0.2 (smallest in series)
        self.scaling_factor = 1.0 / 5.0

        # Initial solution: uniformly distributed
        x_start = np.linspace(5, 85, self.D)

        # Domain: 0° to 90°
        domain = [(0.0, 90.0) for _ in range(self.D)]

        f_x_start = 0.0
        name = "RW49"
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
        Evaluate SHE-PWM configuration for 11-level inverter

        This represents the extreme case of the multilevel series,
        where further complexity provides minimal practical benefit.

        Returns:
        fitness: Total cost (objective + penalties for constraint violations)
        """

        x = np.array(variables_values, dtype=float)  # Angles in degrees

        # ===== OBJECTIVE FUNCTION =====
        # f = (1/5) × √(∑[j=1:31] [1/k_j⁴ × (∑[l=1:D] s_l × cos(k_j×θ_l))²]) / √(∑ 1/k⁴)

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

        # Objective: normalized THD with 1/5 scaling for 11-level
        objective = self.scaling_factor * np.sqrt(harmonic_sum) / self.norm_factor

        # ===== INEQUALITY CONSTRAINTS (Monotonicity) =====
        # g_i: θ_i - θ_{i+1} + ε ≤ 0
        # Enforces: θ_1 < θ_2 < ... < θ_D

        g_penalty = 0.0
        epsilon = 1e-6

        for i in range(self.D - 1):
            violation = x[i] - x[i + 1] + epsilon
            if violation > 0:
                g_penalty += (1e4 * violation) ** 2

        # ===== EQUALITY CONSTRAINT (Fundamental Voltage) =====
        # h: ∑[l=1:D] s_l × cos(θ_l) - 5m = 0
        # Target: 5 × (1/3) = 5/3 ≈ 1.6667

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


class RW49_Detailed(Function):
    """
    RC49 with detailed constraint tracking and harmonic analysis
    """

    def __init__(self, dim):
        """RC49 with comprehensive analysis"""

        self.D = dim
        self.m = 1.0 / 3.0
        self.target_fundamental = 5 * self.m  # 5/3

        self.s = np.array([
            1, -1, 1, 1, 1, -1,
            -1, -1, 1, 1, 1, 1,
            -1, -1, 1, -1, -1, -1,
            1, 1, 1, 1, -1, 1,
            1, -1, -1, 1, -1, -1
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
        self.scaling_factor = 1.0 / 5.0

        x_start = np.linspace(5, 85, self.D)
        domain = [(0.0, 90.0) for _ in range(self.D)]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0.0,
            max_dimension=self.D,
            name="RW49_Detailed"
        )
        self.types(['CEC2020RW'])

        self.last_h = 0.0
        self.last_g_violations = 0.0
        self.last_thd = 0.0
        self.last_harmonics = None
        self.last_objective = 0.0
        self.last_angles = None

    def eval_harmonics(self, angles):
        """Compute all harmonic components"""

        x = np.array(angles, dtype=float)
        harmonics = {}

        for j in range(self.num_harmonics):
            k_j = self.k[j]
            S_j = np.sum(self.s * np.cos(k_j * np.radians(x)))
            harmonics[int(k_j)] = S_j

        return harmonics

    def eval_constraints(self, variables_values):
        """Evaluate constraints separately"""

        x = np.array(variables_values, dtype=float)

        # Equality
        fundamental = np.sum(self.s * np.cos(np.radians(x)))
        h = fundamental - self.target_fundamental

        # Inequality
        g = np.zeros(self.D - 1)
        for i in range(self.D - 1):
            g[i] = x[i] - x[i + 1] + 1e-6

        self.last_h = h
        self.last_g_violations = np.sum(np.maximum(0, g) ** 2)
        self.last_angles = x.copy()

        return h, g

    def eval(self, variables_values):
        """Comprehensive evaluation"""

        x = np.array(variables_values, dtype=float)

        # Objective
        harmonic_sum = 0.0
        self.last_harmonics = {}

        for j in range(self.num_harmonics):
            k_j = self.k[j]
            S_j = np.sum(self.s * np.cos(k_j * np.radians(x)))
            self.last_harmonics[int(k_j)] = S_j
            harmonic_sum += (S_j ** 2) / (k_j ** 4)

        self.last_objective = self.scaling_factor * np.sqrt(harmonic_sum) / self.norm_factor
        self.last_thd = self.last_objective * 100

        # Constraints
        h, g = self.eval_constraints(x)

        # Penalties
        h_penalty = (1e5 * h) ** 2
        g_penalty = 1e4 * self.last_g_violations

        fitness = self.last_objective + h_penalty + g_penalty

        return fitness if np.isfinite(fitness) else 1e10

    def get_analysis(self):
        """Detailed analysis"""

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
        """Verify strict monotonicity"""
        if self.last_angles is None:
            return False
        for i in range(len(self.last_angles) - 1):
            if self.last_angles[i] >= self.last_angles[i + 1]:
                return False
        return True

    def print_analysis(self):
        """Pretty-print analysis"""

        analysis = self.get_analysis()

        print("\n" + "=" * 75)
        print("RC49 DETAILED ANALYSIS - 11-Level Inverter SHE-PWM")
        print("=" * 75)

        print(f"\n[OBJECTIVE]")
        print(f"  THD (normalized): {analysis['thd_normalized']:.6f}")
        print(f"  THD (percentage): {analysis['thd_percent']:.3f}%")
        print(f"  Expected range: 0.2-0.5% (exceptional quality)")
        print(f"  Practical need: <0.5% already meets all standards")

        print(f"\n[CONSTRAINTS]")
        print(f"  Equality h (target=1.6667): {analysis['h_constraint']:.6f}")
        print(f"  Equality violation: {analysis['h_violation']:.2e}")
        print(f"  Monotonicity violations: {analysis['g_violations']:.2e}")
        print(f"  Angles strictly ordered: {'YES' if analysis['monotonicity_verified'] else 'NO'}")
        print(f"  Feasible: {'YES ✓' if analysis['feasible'] else 'NO ✗'}")

        print(f"\n[HARMONIC SPECTRUM] (Low-order)")
        for harm, value in sorted(analysis['primary_harmonics'].items()):
            print(f"  h={harm:2d}: {value:7.4f}")

        print(f"\n[STATISTICS]")
        print(f"  Total harmonics: {len(analysis['harmonics']) if analysis['harmonics'] else 0}")
        print(f"  Harmonics |S_j| > 0.1: {analysis['harmonics_count_above_threshold']}")

        print(f"\n[MULTILEVEL PROGRESSION]")
        print(f"  RC45 (3-level):   5-10% THD")
        print(f"  RC46 (5-level):   2-5% THD")
        print(f"  RC47 (7-level):   1-2% THD")
        print(f"  RC48 (9-level):   0.5-1% THD")
        print(f"  RC49 (11-level):  0.2-0.5% THD ← EXTREME CASE")

        print(f"\n[PRACTICAL REALITY]")
        print(f"  Grid standard (EN 61000): <5% THD → 3-level sufficient")
        print(f"  Motor protection limit: dV/dt <5kV/μs → 5-level sufficient")
        print(f"  Current achievement: {analysis['thd_percent']:.3f}% (exceeds any real requirement)")
        print(f"  Cost vs benefit: 12-15× more expensive for unmeasurable gain")
        print(f"  Real-world deployment: <0.1% of inverters (research only)")

        print("=" * 75 + "\n")
