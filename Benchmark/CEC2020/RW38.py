from Benchmark.__Function import Function
import numpy as np
import os


class RW38(Function):
    def __init__(self, dim):
        """
        Optimal Power Flow (OPF)
        Minimization of Fuel Cost (Economic Dispatch)

        Problem: Determine optimal voltage profile and generator dispatch
        to minimize fuel/generation cost while satisfying power balance

        Variables (126 total):
        - V[2:30] (real & imag):     Voltage at buses 2-30 (29 buses × 2) = 58 vars
        - P_sp[2:30]:                Active power setpoints (29 buses) = 29 vars
        - Q_sp[2:30]:                Reactive power setpoints (29 buses) = 29 vars
        - P_g:                       Generator active power at 5 buses = 5 vars
        - Q_g:                       Generator reactive power at 5 buses = 5 vars

        Total: 58 + 29 + 29 + 5 + 5 = 126 variables

        Network: 30-bus IEEE transmission test system
        Generator buses: 1, 2, 13, 22, 23, 27 (6 generators, but we optimize 5)
        Slack bus: Bus 1 (V = 1.0∠0°, P_g1 = load - losses)

        Fuel Cost Function (Quadratic):
        F(P_g) = ∑[a_i + b_i × P_g_i + c_i × P_g_i²]

        Where:
        - a_i: Fixed cost ($/h) = [0, 0, 0, 0, 0, 0]
        - b_i: Linear cost ($/MWh) = [2.0, 1.75, 1.0, 3.25, 3.0, 3.0]
        - c_i: Quadratic cost ($/MW²h) = [0.02, 0.0175, 0.0625, 0.00834, 0.025, 0.0025]

        Generators at buses (1-indexed): 1, 2, 13, 22, 23, 27
        In optimization: dispatch P_g at buses 2, 13, 22, 23, 27 (5 gens)
        Bus 1: Slack generator (balances load and losses)
        """

        self.load_network_data()

        n_vars = 126

        # Initial conditions
        x_start = np.zeros(n_vars)
        x_start[0:29] = 1.0  # V_real (buses 2-30)
        x_start[29:58] = 0.0  # V_imag
        x_start[58:87] = 0.0  # P_sp
        x_start[87:116] = 0.0  # Q_sp
        x_start[116:121] = 1.0  # P_g (initial 1.0 MW each)
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
        name = "RW38"
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
        self.gen_buses = [1, 2, 13, 22, 23, 27]  # 1-indexed
        self.gen_buses_idx = [0, 1, 12, 21, 22, 26]  # 0-indexed

        # Fuel cost coefficients: F(P_g) = a + b*P_g + c*P_g²
        self.a_cost = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Fixed cost
        self.b_cost = np.array([2.0, 1.75, 1.0, 3.25, 3.0, 3.0])  # Linear
        self.c_cost = np.array([0.02, 0.0175, 0.0625, 0.00834, 0.025, 0.0025])  # Quadratic

        # Generator power limits
        self.P_g_min = 0.5  # Minimum output (MW)
        self.P_g_max = 2.5  # Maximum output (MW)
        self.Q_g_min = -1.0  # Minimum reactive (MVAr)
        self.Q_g_max = 1.0  # Maximum reactive (MVAr)

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

            # Standard loads
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

    def eval(self, variables_values):
        """
        Evaluate OPF for fuel cost minimization

        Objective:
        f = ∑[a_i + b_i × P_g_i + c_i × P_g_i²]

        Constraints:
        - Power flow equations (116 equality constraints)
        - Voltage limits: 0.9 ≤ |V| ≤ 1.1 pu
        - Generator limits: P_g_min ≤ P_g ≤ P_g_max
        - Reactive power limits: Q_g_min ≤ Q_g ≤ Q_g_max
        - Power balance at all buses
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT DESIGN VARIABLES =====
        V_real_230 = x[0:29]
        V_imag_230 = x[29:58]
        V_230 = V_real_230 + 1j * V_imag_230

        Psp_230 = x[58:87]
        Qsp_230 = x[87:116]

        # Generator power (optimized buses: 2, 13, 22, 23, 27)
        Pg_opt = x[116:121]  # 5 generators
        Qg_opt = x[121:126]

        # ===== CONSTRUCT FULL VECTORS =====
        V = np.zeros(30, dtype=complex)
        V[0] = 1.0 + 0.0j  # Slack bus
        V[1:30] = V_230

        Psp = np.zeros(30)
        Qsp = np.zeros(30)
        Psp[1:30] = Psp_230
        Qsp[1:30] = Qsp_230

        Pg = np.zeros(30)
        Qg = np.zeros(30)

        # Map optimized generators to bus indices
        # Buses 2, 13, 22, 23, 27 → indices 1, 12, 21, 22, 26
        gen_opt_indices = [1, 12, 21, 22, 26]
        for idx, bus_idx in enumerate(gen_opt_indices):
            Pg[bus_idx] = Pg_opt[idx]
            Qg[bus_idx] = Qg_opt[idx]

        # ===== POWER FLOW ANALYSIS =====
        I = self.Y @ V
        I_r = np.real(I)
        I_m = np.imag(I)

        # Slack bus power injection
        P_slack = np.real(V[0] * np.conj(I[0]))
        Pg[0] = P_slack

        # Specified currents
        V_conj = np.conj(V) + 1e-10
        S_sp = Psp + 1j * Qsp
        I_sp = S_sp / V_conj
        I_sp_r = np.real(I_sp)
        I_sp_m = np.imag(I_sp)

        # ===== POWER BALANCE EQUATIONS =====
        delP = Psp - Pg + self.P
        delQ = Qsp - Qg + self.Q
        delI_r = I_r - I_sp_r
        delI_m = I_m - I_sp_m

        # ===== OBJECTIVE FUNCTION (FUEL COST) =====
        # Only optimize for dispatched generators (5 units)
        # Generator 1 (slack): cost not optimized
        fuel_cost = 0.0
        for i, idx in enumerate(gen_opt_indices):
            # Use cost coefficients for all 6 generators
            # But only sum for optimized 5 (index 1-5 in cost arrays)
            fuel_cost += (self.a_cost[i + 1] +
                          self.b_cost[i + 1] * Pg[idx] +
                          self.c_cost[i + 1] * Pg[idx] ** 2)

        # Add slack generator cost
        fuel_cost += (self.a_cost[0] +
                      self.b_cost[0] * Pg[0] +
                      self.c_cost[0] * Pg[0] ** 2)

        # ===== CONSTRAINT VIOLATIONS =====
        # 1. Power balance (equality constraints)
        h = np.concatenate([
            delI_r[1:30],
            delI_m[1:30],
            delP[1:30],
            delQ[1:30]
        ])

        h_violation = np.sum(np.abs(h)) * 1e-2 + np.sum(h ** 2) * 1e-3

        # 2. Voltage magnitude limits
        V_mag = np.abs(V)
        V_lower_violations = np.maximum(0, 0.9 - V_mag)
        V_upper_violations = np.maximum(0, V_mag - 1.1)

        # 3. Generator active power limits
        Pg_lower_violations = np.maximum(0, self.P_g_min - Pg_opt)
        Pg_upper_violations = np.maximum(0, Pg_opt - self.P_g_max)

        # 4. Generator reactive power limits
        Qg_lower_violations = np.maximum(0, self.Q_g_min - Qg_opt)
        Qg_upper_violations = np.maximum(0, Qg_opt - self.Q_g_max)

        # 5. Slack generator limits
        Pg_slack_max = 5.0  # Higher limit for slack
        P_slack_violation = np.maximum(0, np.abs(Pg[0]) - Pg_slack_max)

        # Compile inequality violations
        g_violation = (np.sum(V_lower_violations ** 2) +
                       np.sum(V_upper_violations ** 2) +
                       np.sum(Pg_lower_violations ** 2) +
                       np.sum(Pg_upper_violations ** 2) +
                       np.sum(Qg_lower_violations ** 2) +
                       np.sum(Qg_upper_violations ** 2) +
                       P_slack_violation ** 2)

        # ===== FITNESS FUNCTION =====
        # Minimize fuel cost while enforcing constraints
        penalty_h = 1e5  # High penalty for power balance
        penalty_g = 1e4  # Moderate penalty for other constraints

        fitness = fuel_cost + penalty_h * h_violation + penalty_g * g_violation

        return fitness if np.isfinite(fitness) else 1e10


class RW38_Simplified(Function):
    """
    Simplified OPF-Fuel Cost without external data files
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
            max_dimension=n_vars, name="RW38_Simplified"
        )
        self.types(['CEC2020RW'])

        # Synthetic network
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

        # ===== FUEL COST COEFFICIENTS =====
        # Buses: [1, 2, 13, 22, 23, 27]
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

        # Fuel cost for 5 optimized generators + slack
        fuel_cost = 0.0
        gen_opt_indices = [1, 12, 21, 22, 26]  # 0-indexed

        # Slack bus
        fuel_cost += self.b_cost[0] * P_slack + self.c_cost[0] * P_slack ** 2

        # Optimized generators
        for i, idx in enumerate(gen_opt_indices):
            fuel_cost += (self.b_cost[i + 1] * Pg_opt[i] +
                          self.c_cost[i + 1] * Pg_opt[i] ** 2)

        # Penalties
        V_mag = np.abs(V)
        penalty = 1e3 * (
                np.sum((np.clip(V_mag[1:30], 0.9, 1.1) - V_mag[1:30]) ** 2) +
                np.sum(np.maximum(0, 0.5 - Pg_opt) ** 2) +
                np.sum(np.maximum(0, Pg_opt - 2.5) ** 2)
        )

        fitness = fuel_cost + penalty

        return fitness if np.isfinite(fitness) else 1e10
