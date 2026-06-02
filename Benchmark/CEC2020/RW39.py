from Benchmark.Functions.__Function import Function
import numpy as np
import os


class RW39(Function):
    def __init__(self, dim):

        self.load_network_data()

        n_vars = 126

        # Initial conditions
        x_start = np.zeros(n_vars)
        x_start[0:29] = 1.0  # V_real (buses 2-30)
        x_start[29:58] = 0.0  # V_imag
        x_start[58:87] = 0.0  # P_sp
        x_start[87:116] = 0.0  # Q_sp
        x_start[116:121] = 1.0  # P_g initial 1.0 MW each
        x_start[121:126] = 0.0  # Q_g

        # Domain bounds
        domain = [
            # Voltage real parts: V_r ∈ [0.9, 1.1] pu
            *[(0.9, 1.1) for _ in range(29)],
            # Voltage imag parts: V_i ∈ [-0.3, 0.3]
            *[(-0.3, 0.3) for _ in range(29)],
            # Active power setpoints: P_sp ∈ [-2.0, 3.0] MW
            *[(-2.0, 3.0) for _ in range(29)],
            # Reactive power setpoints: Q_sp ∈ [-2.0, 2.0] MVAr
            *[(-2.0, 2.0) for _ in range(29)],
            # Generator active power (5 gens): P_g ∈ [0.5, 2.5] MW
            *[(0.5, 2.5) for _ in range(5)],
            # Generator reactive power: Q_g ∈ [-1.0, 1.0] MVAr
            *[(-1.0, 1.0) for _ in range(5)],
        ]

        f_x_start = 0
        name = "RW39"
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

        # ===== FUEL COST COEFFICIENTS =====
        # Generator buses: [1, 2, 13, 22, 23, 27] (1-indexed)
        self.gen_buses = [1, 2, 13, 22, 23, 27]
        self.gen_buses_idx = [0, 1, 12, 21, 22, 26]  # 0-indexed

        # Cost coefficients: F(P_g) = a + b*P_g + c*P_g²
        self.a_cost = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.b_cost = np.array([2.0, 1.75, 1.0, 3.25, 3.0, 3.0])
        self.c_cost = np.array([0.02, 0.0175, 0.0625, 0.00834, 0.025, 0.0025])

        # Multi-objective weights
        self.w_fuel = 1.0  # Weight for fuel cost
        self.w_loss = 0.75  # Weight for active power loss

        # Limits
        self.P_g_min = 0.5
        self.P_g_max = 2.5
        self.Q_g_min = -1.0
        self.Q_g_max = 1.0
        self.V_min = 0.9
        self.V_max = 1.1

    def load_network_data(self):
        """
        Load 30-bus IEEE transmission network data
        """
        try:
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            self.G = np.loadtxt(os.path.join(base_path, "FunctionPS11_G.txt"))
            self.B = np.loadtxt(os.path.join(base_path, "FunctionPS11_B.txt"))
            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS11_P.txt")).flatten()
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS11_Q.txt")).flatten()


        except Exception as e:
            print(f"⚠️  Network data files not found: {e}")
            print("    Using synthetic 30-bus IEEE system")

            # ===== SYNTHETIC 30-BUS NETWORK =====
            n_bus = 30
            np.random.seed(42)

            self.G = np.eye(n_bus) * 0.3 + np.random.randn(n_bus, n_bus) * 0.01
            self.G = (self.G + self.G.T) / 2

            self.B = -np.eye(n_bus) * 1.5 + np.random.randn(n_bus, n_bus) * 0.05
            self.B = (self.B + self.B.T) / 2

            self.G[0, :] = 0.0
            self.B[0, :] = 0.0

            # Standard IEEE 30-bus loads
            self.P = np.array([
                0.0, 0.217, 0.024, 0.076, 0.942, 0.0, 0.228, 0.3, 0.0, 0.058,
                0.076, 0.135, 0.0, 0.149, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.248, 0.0, 0.0, 0.0, 0.0, 0.0, 0.106
            ])

            self.Q = np.array([
                0.0, 0.127, 0.012, 0.016, 0.31, 0.0, 0.109, 0.3, 0.0, 0.02,
                0.016, 0.058, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.076
            ])

        self.Y = self.G + 1j * self.B
        self.P_total_load = np.sum(self.P)
        self.Q_total_load = np.sum(self.Q)

    def eval(self, variables_values):
        """
        Evaluate OPF for combined fuel cost + active power loss minimization

        Objective:
        f = ∑[a_i + b_i × P_g_i + c_i × P_g_i²] + 0.75 × P_loss

        Multi-objective formulation:
        - Primary: Minimize fuel generation cost
        - Secondary: Minimize transmission losses (weighted 0.75)
        - Balance between economy and efficiency

        Constraints (equality):
        - Power flow equations (116 constraints)
        - Power balance: I_calc = I_specified + I_generation
        - Kirchhoff's current law at all buses

        Constraints (inequality):
        - Voltage limits: 0.9 ≤ |V| ≤ 1.1 pu
        - Generator limits: 0.5 ≤ P_g ≤ 2.5 MW
        - Reactive limits: -1.0 ≤ Q_g ≤ 1.0 MVAr
        - Thermal limits: implicit in current calculations
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT DESIGN VARIABLES =====
        V_real_230 = x[0:29]
        V_imag_230 = x[29:58]
        V_230 = V_real_230 + 1j * V_imag_230

        Psp_230 = x[58:87]
        Qsp_230 = x[87:116]

        # 5 optimized generators (buses 2, 13, 22, 23, 27)
        Pg_opt = x[116:121]
        Qg_opt = x[121:126]

        # ===== BUILD FULL STATE VECTORS =====
        V = np.zeros(30, dtype=complex)
        V[0] = 1.0 + 0.0j  # Slack bus voltage
        V[1:30] = V_230

        Psp = np.zeros(30)
        Qsp = np.zeros(30)
        Psp[1:30] = Psp_230
        Qsp[1:30] = Qsp_230

        Pg = np.zeros(30)
        Qg = np.zeros(30)

        # Map optimized generators
        gen_opt_indices = [1, 12, 21, 22, 26]  # 0-indexed: buses 2,13,22,23,27
        for idx, bus_idx in enumerate(gen_opt_indices):
            Pg[bus_idx] = Pg_opt[idx]
            Qg[bus_idx] = Qg_opt[idx]

        # ===== POWER FLOW ANALYSIS =====
        I = self.Y @ V
        I_r = np.real(I)
        I_m = np.imag(I)

        # Slack generator supplies: generation = load + losses
        S_slack = V[0] * np.conj(I[0])
        P_slack = np.real(S_slack)
        Q_slack = np.imag(S_slack)
        Pg[0] = P_slack
        Qg[0] = Q_slack

        # ===== POWER LOSS CALCULATION =====
        # Active power loss = total generation - total load
        P_loss = np.sum(Pg) - np.sum(self.P)

        # Alternative: P_loss = real(V[0] * conj(I[0]))
        P_loss_alt = np.real(V[0] * np.conj(I[0]))

        # Use the calculated loss
        P_loss = max(P_loss, P_loss_alt)

        # ===== OBJECTIVE FUNCTION (MULTI-OBJECTIVE SCALARIZATION) =====

        # 1. Fuel cost for all generators
        fuel_cost = 0.0

        # Slack generator cost
        fuel_cost += (self.a_cost[0] +
                      self.b_cost[0] * Pg[0] +
                      self.c_cost[0] * Pg[0] ** 2)

        # Optimized generators cost
        for i, idx in enumerate(gen_opt_indices):
            fuel_cost += (self.a_cost[i + 1] +
                          self.b_cost[i + 1] * Pg[idx] +
                          self.c_cost[i + 1] * Pg[idx] ** 2)

        # 2. Active power loss term
        loss_cost = self.w_loss * P_loss

        # 3. Combined objective
        # Note: P_loss appears as "0.75 * sum(Pg - P)" in MATLAB code
        # This is equivalent to 0.75 * (total_generation - total_load)
        objective = fuel_cost + loss_cost

        # ===== POWER BALANCE EQUATIONS (EQUALITY CONSTRAINTS) =====
        V_conj = np.conj(V) + 1e-12  # Avoid division by zero
        S_sp = Psp + 1j * Qsp
        I_sp = S_sp / V_conj
        I_sp_r = np.real(I_sp)
        I_sp_m = np.imag(I_sp)

        delP = Psp - Pg + self.P
        delQ = Qsp - Qg + self.Q
        delI_r = I_r - I_sp_r
        delI_m = I_m - I_sp_m

        # Constraint violation metric
        h_constraint = np.concatenate([
            delI_r[1:30],
            delI_m[1:30],
            delP[1:30],
            delQ[1:30]
        ])

        h_violation = np.sum(np.abs(h_constraint)) + 1e-3 * np.sum(h_constraint ** 2)

        # ===== INEQUALITY CONSTRAINTS VIOLATIONS =====

        # 1. Voltage magnitude limits
        V_mag = np.abs(V)
        V_violations = (
                np.sum(np.maximum(0, self.V_min - V_mag) ** 2) +
                np.sum(np.maximum(0, V_mag - self.V_max) ** 2)
        )

        # 2. Generator active power limits
        Pg_violations = (
                np.sum(np.maximum(0, self.P_g_min - Pg_opt) ** 2) +
                np.sum(np.maximum(0, Pg_opt - self.P_g_max) ** 2) +
                np.maximum(0, -P_slack) ** 2 +  # Slack must be positive (generation)
                np.maximum(0, P_slack - 5.0) ** 2  # Slack limit
        )

        # 3. Generator reactive power limits
        Qg_violations = (
                np.sum(np.maximum(0, self.Q_g_min - Qg_opt) ** 2) +
                np.sum(np.maximum(0, Qg_opt - self.Q_g_max) ** 2)
        )

        # Total constraint violations
        g_violation = V_violations + Pg_violations + Qg_violations

        # ===== FITNESS FUNCTION =====
        penalty_h = 1e5  # Very high penalty for power balance
        penalty_g = 1e4  # High penalty for inequality constraints

        fitness = (objective +
                   penalty_h * h_violation +
                   penalty_g * g_violation)

        # Check for numerical issues
        if not np.isfinite(fitness):
            fitness = 1e10

        return fitness


class RW39_Simplified(Function):
    """
    Simplified version without external data dependencies
    """

    def __init__(self, dim):
        n_vars = 126

        x_start = np.zeros(n_vars)
        x_start[0:29] = 1.0
        x_start[116:121] = 1.0

        domain = [
            *[(0.9, 1.1) for _ in range(29)],
            *[(-0.3, 0.3) for _ in range(29)],
            *[(-2.0, 3.0) for _ in range(29)],
            *[(-2.0, 2.0) for _ in range(29)],
            *[(0.5, 2.5) for _ in range(5)],
            *[(-1.0, 1.0) for _ in range(5)],
        ]

        Function.__init__(
            self, dim, domain, x_start, 0,
            max_dimension=n_vars, name="RW39_Simplified"
        )
        self.types(['CEC2020RW'])

        # Synthetic 30-bus network
        np.random.seed(42)
        n_bus = 30

        self.G = np.eye(n_bus) * 0.3 + np.random.randn(n_bus, n_bus) * 0.01
        self.G = (self.G + self.G.T) / 2

        self.B = -np.eye(n_bus) * 1.5 + np.random.randn(n_bus, n_bus) * 0.05
        self.B = (self.B + self.B.T) / 2

        self.G[0, :] = 0.0
        self.B[0, :] = 0.0

        self.Y = self.G + 1j * self.B

        self.P = np.array([
            0.0, 0.217, 0.024, 0.076, 0.942, 0.0, 0.228, 0.3, 0.0, 0.058,
            0.076, 0.135, 0.0, 0.149, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.248, 0.0, 0.0, 0.0, 0.0, 0.0, 0.106
        ])

        self.Q = np.array([
            0.0, 0.127, 0.012, 0.016, 0.31, 0.0, 0.109, 0.3, 0.0, 0.02,
            0.016, 0.058, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.124, 0.0, 0.0, 0.0, 0.0, 0.0, 0.076
        ])

        # Fuel cost coefficients
        self.b_cost = np.array([2.0, 1.75, 1.0, 3.25, 3.0, 3.0])
        self.c_cost = np.array([0.02, 0.0175, 0.0625, 0.00834, 0.025, 0.0025])

    def eval(self, variables_values):
        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        V_r = x[0:29]
        V_i = x[29:58]
        V_230 = V_r + 1j * V_i

        Psp = x[58:87]
        Qsp = x[87:116]
        Pg_opt = x[116:121]

        # Build voltage
        V = np.zeros(30, dtype=complex)
        V[0] = 1.0
        V[1:30] = V_230

        # Power flow
        I = self.Y @ V
        P_slack = np.real(V[0] * np.conj(I[0]))

        # ===== FUEL COST =====
        fuel_cost = 0.0

        # Slack
        fuel_cost += self.b_cost[0] * P_slack + self.c_cost[0] * P_slack ** 2

        # Optimized generators
        gen_opt_indices = [1, 12, 21, 22, 26]
        for i, idx in enumerate(gen_opt_indices):
            fuel_cost += (self.b_cost[i + 1] * Pg_opt[i] +
                          self.c_cost[i + 1] * Pg_opt[i] ** 2)

        # ===== ACTIVE POWER LOSS =====
        # Total generation - total load
        P_gen_total = P_slack + np.sum(Pg_opt)
        P_loss = P_gen_total - np.sum(self.P)

        # ===== COMBINED OBJECTIVE =====
        objective = fuel_cost + 0.75 * P_loss

        # ===== PENALTIES =====
        V_mag = np.abs(V)
        penalty = 1e3 * (
                np.sum((np.clip(V_mag[1:30], 0.9, 1.1) - V_mag[1:30]) ** 2) +
                np.sum(np.maximum(0, 0.5 - Pg_opt) ** 2) +
                np.sum(np.maximum(0, Pg_opt - 2.5) ** 2) +
                np.maximum(0, -P_slack) ** 2
        )

        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else 1e10
