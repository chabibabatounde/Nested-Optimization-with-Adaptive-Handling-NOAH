from Benchmark.Functions.__Function import Function
import numpy as np
from scipy.special import gamma as gamma_func


class RW44(Function):
    def __init__(self, dim):

        n_turbines = 15
        n_vars = 2 * n_turbines  # (x, y) for each turbine = 30 variables

        # ===== WIND FARM PARAMETERS =====
        self.interval = 15  # Wind direction sector width (degrees)
        self.interval_num = int(360 / self.interval)  # 24 sectors

        self.cut_in_speed = 3.5  # m/s
        self.rated_speed = 14.0  # m/s
        self.cut_out_speed = 25.0  # m/s

        self.R = 40.0  # Rotor radius (m)
        self.H = 80.0  # Hub height (m)
        self.C_T = 0.8  # Thrust coefficient
        self.a = 1.0 - np.sqrt(1.0 - self.C_T)  # Wake intensity ≈ 0.3820
        self.kappa = 0.01  # Turbulence intensity

        self.min_distance = 5.0 * self.R  # 200m minimum spacing

        self.N = n_turbines
        self.X_farm = 2000.0  # Farm width (m)
        self.Y_farm = 2000.0  # Farm length (m)

        # ===== WEIBULL PARAMETERS (per wind direction) =====
        # Shape parameter k (steepness)
        self.k = np.full(self.interval_num, 2.0)  # Rayleigh for all directions

        # Scale parameters c (wind speed scale) per direction
        self.c = np.array([
            7, 5, 5, 5, 5, 4, 5, 6, 7, 7, 8, 9.5, 10, 8.5, 8.5, 6.5, 4.6, 2.6, 8, 5, 6.4, 5.2, 4.5, 3.9
        ], dtype=float)

        # Wind frequency per direction (probability)
        self.fre = np.array([
            0.0003, 0.0072, 0.0237, 0.0242, 0.0222, 0.0301, 0.0397, 0.0268, 0.0626,
            0.0801, 0.1025, 0.1445, 0.1909, 0.1162, 0.0793, 0.0082, 0.0041, 0.0008,
            0.0010, 0.0005, 0.0013, 0.0031, 0.0085, 0.0222
        ], dtype=float)

        # Air density (kg/m³)
        self.rho = 1.225
        self.A_rotor = np.pi * self.R ** 2  # Rotor area ≈ 5027 m²

        # Initial solution: grid layout
        x_start = np.zeros(n_vars)
        turbines_per_row = int(np.ceil(np.sqrt(self.N)))
        spacing_x = self.X_farm / (turbines_per_row + 1)
        spacing_y = self.Y_farm / (turbines_per_row + 1)

        turb_idx = 0
        for i in range(turbines_per_row):
            for j in range(turbines_per_row):
                if turb_idx < self.N:
                    x_start[turb_idx] = (i + 1) * spacing_x
                    x_start[turb_idx + self.N] = (j + 1) * spacing_y
                    turb_idx += 1

        # Domain: farm boundaries
        domain = [
            *[(0.0, self.X_farm) for _ in range(n_turbines)],  # X coordinates
            *[(0.0, self.Y_farm) for _ in range(n_turbines)],  # Y coordinates
        ]

        f_x_start = 0
        name = "RW44"
        max_dimension = n_vars

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


    def weibull_pdf(self, v, k, c):
        """
        Weibull probability density function

        f(v) = (k/c) × (v/c)^(k-1) × exp(-(v/c)^k)
        """
        if v <= 0:
            return 0.0

        term1 = k / c
        term2 = (v / c) ** (k - 1)
        term3 = np.exp(-(v / c) ** k)

        return term1 * term2 * term3

    def weibull_cdf(self, v, k, c):
        """
        Weibull cumulative distribution function

        F(v) = 1 - exp(-(v/c)^k)
        """
        if v <= 0:
            return 0.0

        return 1.0 - np.exp(-(v / c) ** k)

    def power_curve(self, v):
        """
        Power output from single turbine (W)

        P(v) = {
            0                           if v < 3.5 or v > 25 m/s
            (rated power) × ramp(v)     if 3.5 ≤ v ≤ 14 m/s
            P_rated                     if 14 < v ≤ 25 m/s
        }
        """

        if v < self.cut_in_speed or v > self.cut_out_speed:
            return 0.0

        if v >= self.rated_speed:
            # Rated power (2.4 MW typical for 40m rotor)
            return 2.4e6

        # Ramp-up region: cubic interpolation (common)
        # P(v) = P_rated × ((v - v_ci) / (v_rated - v_ci))³
        ramp_coeff = (v - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)
        power = 2.4e6 * (ramp_coeff ** 3)

        return power

    def expected_power_single_turbine(self, k, c):
        """
        Expected power output for single turbine in Weibull wind

        E[P] = ∫[v_ci to v_co] P(v) × f(v|k,c) dv

        Numerical integration (trapezoid rule)
        """

        v_range = np.linspace(0, self.cut_out_speed + 5, 200)
        power_curve = np.array([self.power_curve(v) for v in v_range])
        pdf = np.array([self.weibull_pdf(v, k, c) for v in v_range])

        expected_power = np.trapz(power_curve * pdf, v_range)

        return expected_power

    def compute_wake_deficit(self, v_free, x_downwind):
        """
        Wake velocity deficit at distance x downwind

        ΔV = a × V₀ × (R / (R + κ×x))²

        Returns reduced velocity (V - ΔV)
        """

        if x_downwind <= 0:
            return v_free

        delta_v = self.a * v_free * (self.R / (self.R + self.kappa * x_downwind)) ** 2
        v_reduced = np.sqrt(max(0, v_free ** 2 - delta_v ** 2))

        return v_reduced

    def compute_farm_power(self, turbine_pos, wind_direction, wind_speed):
        """
        Compute total farm power for specific wind direction & speed

        turbine_pos: (N, 2) array of turbine (x, y) positions
        wind_direction: angle in degrees (0-360)
        wind_speed: wind speed at hub height (m/s)

        Returns: total power (W)
        """

        N = len(turbine_pos)
        power_total = 0.0

        # Wind direction unit vector
        wind_rad = np.radians(wind_direction)
        wind_unit = np.array([np.cos(wind_rad), np.sin(wind_rad)])

        # Perpendicular vector (for lateral spacing)
        perp_unit = np.array([-np.sin(wind_rad), np.cos(wind_rad)])

        for j in range(N):
            # Effective wind speed at turbine j
            v_eff = wind_speed

            # Account for wakes from upwind turbines
            for i in range(N):
                if i == j:
                    continue

                # Vector from turbine i to j
                delta_pos = turbine_pos[j] - turbine_pos[i]

                # Downwind distance (positive = j is downwind of i)
                x_downwind = np.dot(delta_pos, wind_unit)

                # Lateral distance
                y_lateral = np.abs(np.dot(delta_pos, perp_unit))

                # Check if j is in i's wake
                if x_downwind > 0 and y_lateral <= 2 * self.R:  # Wake cone
                    # Reduce wind speed due to wake
                    wake_deficit = self.a * wind_speed * \
                                   (self.R / (self.R + self.kappa * x_downwind)) ** 2

                    v_eff = np.sqrt(max(0, v_eff ** 2 - wake_deficit ** 2))

            # Power from this turbine
            power_total += self.power_curve(v_eff)

        return power_total

    def fitness_function(self, turbine_positions):
        """
        Compute total expected power output

        F = ∑[dir=1:24] freq[dir] × ∑[speeds] P(speeds) × f(speeds|k,c)

        Returns: -F (for minimization; negative because we maximize)
        """

        turbine_pos = turbine_positions.reshape((self.N, 2))

        total_energy = 0.0

        # For each wind direction
        for dir_idx in range(self.interval_num):
            wind_freq = self.fre[dir_idx]
            wind_direction = dir_idx * self.interval

            k = self.k[dir_idx]
            c = self.c[dir_idx]

            # Integrate over wind speed distribution
            v_range = np.linspace(0, self.cut_out_speed + 5, 100)

            direction_energy = 0.0
            for v in v_range:
                # Weibull PDF
                pdf = self.weibull_pdf(v, k, c)

                # Farm power at this wind speed
                farm_power = self.compute_farm_power(turbine_pos, wind_direction, v)

                direction_energy += farm_power * pdf

            # Integrate (trapezoid)
            direction_energy *= (v_range[1] - v_range[0]) if len(v_range) > 1 else 1.0

            # Weight by frequency
            total_energy += wind_freq * direction_energy

        return -total_energy  # Negative for minimization

    def eval(self, variables_values):
        """
        Evaluate wind farm layout

        Objective: Maximize power (= minimize -power)
        Constraints: Minimum separation distance
        """

        x = np.array(variables_values, dtype=float)

        # ===== OBJECTIVE FUNCTION =====
        objective = self.fitness_function(x)

        # ===== CONSTRAINT VIOLATIONS =====
        turbine_pos = x.reshape((self.N, 2))

        constraint_violations = 0.0
        constraint_count = 0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                # Distance between turbines i and j
                dist = np.linalg.norm(turbine_pos[i] - turbine_pos[j])

                # Constraint: dist ≥ min_distance
                # Penalty if violated
                violation = max(0, self.min_distance - dist)
                constraint_violations += violation ** 2
                constraint_count += 1

        # ===== BOUNDARY CONSTRAINTS =====
        boundary_violations = 0.0

        for i in range(self.N):
            x_pos = turbine_pos[i, 0]
            y_pos = turbine_pos[i, 1]

            # Check if within farm boundaries
            if x_pos < 0 or x_pos > self.X_farm:
                boundary_violations += min(abs(x_pos), abs(x_pos - self.X_farm)) ** 2

            if y_pos < 0 or y_pos > self.Y_farm:
                boundary_violations += min(abs(y_pos), abs(y_pos - self.Y_farm)) ** 2

        # ===== TOTAL FITNESS =====
        penalty = 1e6 * constraint_violations + 1e6 * boundary_violations

        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else 1e10


class RW44_Fast(Function):
    """
    Fast approximation version for wind farm optimization
    (Uses analytical expected power instead of numerical integration)
    """

    def __init__(self, dim):
        """
        Fast RC44 with analytical expected power computation
        """

        n_turbines = 15
        n_vars = 2 * n_turbines

        self.N = n_turbines
        self.R = 40.0
        self.min_distance = 5.0 * self.R
        self.X_farm = 2000.0
        self.Y_farm = 2000.0
        self.a = 0.3820
        self.kappa = 0.01

        self.cut_in_speed = 3.5
        self.rated_speed = 14.0
        self.cut_out_speed = 25.0
        self.P_rated = 2.4e6

        self.interval_num = 24
        self.c = np.array([7, 5, 5, 5, 5, 4, 5, 6, 7, 7, 8, 9.5, 10, 8.5, 8.5, 6.5, 4.6, 2.6, 8, 5, 6.4, 5.2, 4.5, 3.9])
        self.fre = np.array([0.0003, 0.0072, 0.0237, 0.0242, 0.0222, 0.0301, 0.0397, 0.0268, 0.0626,
                             0.0801, 0.1025, 0.1445, 0.1909, 0.1162, 0.0793, 0.0082, 0.0041, 0.0008,
                             0.0010, 0.0005, 0.0013, 0.0031, 0.0085, 0.0222])
        self.k = np.full(24, 2.0)

        # Initial solution
        x_start = np.zeros(n_vars)
        spacing = self.X_farm / 5
        for i in range(min(15, 25)):
            x_start[i % 15] = (i % 5 + 1) * spacing
            x_start[(i // 5) + 15] = (i // 5 + 1) * spacing

        domain = [
            *[(0.0, self.X_farm) for _ in range(n_turbines)],
            *[(0.0, self.Y_farm) for _ in range(n_turbines)],
        ]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0,
            max_dimension=n_vars,
            name="RW44_Fast"
        )
        self.types(['CEC2020RW'])

    def eval(self, variables_values):
        """
        Fast evaluation using simplified wake model
        """

        x = np.array(variables_values, dtype=float)
        turbine_pos = x.reshape((self.N, 2))

        # ===== FAST POWER ESTIMATION =====
        # Approximate using average wind speed per direction
        total_power = 0.0

        for dir_idx in range(self.interval_num):
            # Expected wind speed for this direction
            c = self.c[dir_idx]
            k = self.k[dir_idx]

            # Mean of Weibull: E[v] = c × Γ(1 + 1/k)
            mean_v = c * gamma_func(1 + 1.0 / k)

            # Approximate: most power comes from mean wind speed
            # Simplified: No wake effects for fast approximation
            power_single = self._power_curve(mean_v)
            total_power += self.fre[dir_idx] * power_single * self.N

        objective = -total_power / (self.N * self.P_rated)  # Normalized

        # ===== CONSTRAINTS =====
        constraint_violations = 0.0

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(turbine_pos[i] - turbine_pos[j])
                violation = max(0, self.min_distance - dist)
                constraint_violations += violation ** 2

        penalty = 1e6 * constraint_violations

        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else 1e10

    def _power_curve(self, v):
        """Simple power curve"""
        if v < self.cut_in_speed or v > self.cut_out_speed:
            return 0.0
        if v >= self.rated_speed:
            return self.P_rated
        ramp = (v - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)
        return self.P_rated * (ramp ** 3)
