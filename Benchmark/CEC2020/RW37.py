from Benchmark.Functions.__Function import Function
import numpy as np
import os


class RW37(Function):
    def __init__(self, dim):
        """
        Optimal Power Flow (OPF)
        Minimization of Active Power Loss

        Problem: Determine optimal voltage profile and generator dispatch
        to minimize active power losses in the transmission network

        Variables (126 total):
        - V[2:30] (real & imag):     Voltage at buses 2-30 (29 buses × 2) = 58 vars
        - P_sp[2:30]:                Active power setpoints (29 buses) = 29 vars
        - Q_sp[2:30]:                Reactive power setpoints (29 buses) = 29 vars
        - P_g:                       Generator active power at 5 buses = 5 vars
        - Q_g:                       Generator reactive power at 5 buses = 5 vars

        Total: 58 + 29 + 29 + 5 + 5 = 126 variables

        Network: 30-bus IEEE test system (standard transmission network)
        Generator buses: 2, 13, 22, 23, 27 (5 synchronous generators)
        Slack bus: Bus 1 (V = 1.0∠0°)

        Objective: min(P_loss + operating_cost)
        """

        self.load_network_data()

        n_vars = 126

        # Initial conditions
        x_start = np.zeros(n_vars)
        x_start[0:29] = 1.0  # V_real (buses 2-30)
        x_start[29:58] = 0.0  # V_imag
        x_start[58:87] = 0.0  # P_sp
        x_start[87:116] = 0.0  # Q_sp
        x_start[116:121] = 1.0  # P_g (initial 1.0 MW)
        x_start[121:126] = 0.0  # Q_g

        # Domain bounds
        domain = [
            # Voltage real parts: V_r ∈ [0.9, 1.1]
            *[(0.9, 1.1) for _ in range(29)],
            # Voltage imag parts: V_i ∈ [-0.3, 0.3]
            *[(-0.3, 0.3) for _ in range(29)],
            # Active power setpoints: P_sp ∈ [-2.0, 3.0] MW
            *[(-2.0, 3.0) for _ in range(29)],
            # Reactive power setpoints: Q_sp ∈ [-2.0, 2.0] MVAr
            *[(-2.0, 2.0) for _ in range(29)],
            # Generator active power (buses 2,13,22,23,27): P_g ∈ [0, 2.5] MW
            *[(0, 2.5) for _ in range(5)],
            # Generator reactive power: Q_g ∈ [-1.0, 1.0] MVAr
            *[(-1.0, 1.0) for _ in range(5)],
        ]

        f_x_start = 0
        name = "RW37"
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

    def load_network_data(self):
        """
        Load 30-bus IEEE transmission network data

        File format:
        - G: Conductance matrix (30 × 30)
        - B: Susceptance matrix (30 × 30)
        - P: Active load at each bus (30 × 1)
        - Q: Reactive load at each bus (30 × 1)
        """
        try:
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            self.G = np.loadtxt(os.path.join(base_path, "FunctionPS11_G.txt"))
            self.B = np.loadtxt(os.path.join(base_path, "FunctionPS11_B.txt"))
            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS11_P.txt")).flatten()
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS11_Q.txt")).flatten()

            # Validate dimensions
            assert self.G.shape == (30, 30), f"G shape {self.G.shape} != (30, 30)"
            assert self.B.shape == (30, 30), f"B shape {self.B.shape} != (30, 30)"
            assert len(self.P) == 30, f"P length {len(self.P)} != 30"
            assert len(self.Q) == 30, f"Q length {len(self.Q)} != 30"


        except Exception as e:
            print(f"⚠️  Network data files not found: {e}")
            print("    Using synthetic 30-bus IEEE system")

            # ===== SYNTHETIC 30-BUS NETWORK =====
            n_bus = 30
            np.random.seed(42)

            # Conductance matrix (transmission system - sparse, low resistance)
            self.G = np.eye(n_bus) * 0.3 + np.random.randn(n_bus, n_bus) * 0.01
            self.G = (self.G + self.G.T) / 2

            # Susceptance matrix (reactance dominant)
            self.B = -np.eye(n_bus) * 1.5 + np.random.randn(n_bus, n_bus) * 0.05
            self.B = (self.B + self.B.T) / 2

            # Slack bus: minimal coupling
            self.G[0, :] = 0.0
            self.B[0, :] = 0.0

            # ===== LOAD PROFILE =====
            # Typical transmission network loads
            load_profile = np.array([
                0.0,  # Bus 1: Slack (no load)
                0.217,  # Bus 2
                0.024,  # Bus 3
                0.076,  # Bus 4
                0.942,  # Bus 5
                0.0,  # Bus 6
                0.228,  # Bus 7
                0.3,  # Bus 8
                0.0,  # Bus 9
                0.058,  # Bus 10
                0.076,  # Bus 11
                0.135,  # Bus 12
                0.0,  # Bus 13: Gen bus
                0.149,  # Bus 14
                0.15,  # Bus 15
                0.0,  # Bus 16
                0.0,  # Bus 17
                0.0,  # Bus 18
                0.0,  # Bus 19
                0.0,  # Bus 20
                0.0,  # Bus 21
                0.0,  # Bus 22: Gen bus
                0.0,  # Bus 23: Gen bus
                0.248,  # Bus 24
                0.0,  # Bus 25
                0.0,  # Bus 26
                0.0,  # Bus 27: Gen bus
                0.0,  # Bus 28
                0.0,  # Bus 29
                0.106  # Bus 30
            ])

            reactive_profile = np.array([
                0.0, 0.127, 0.012, 0.016, 0.31, 0.0, 0.109, 0.3,
                0.0, 0.02, 0.016, 0.058, 0.0, 0.05, 0.05, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.124, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.076
            ])

            self.P = load_profile
            self.Q = reactive_profile

        self.Y = self.G + 1j * self.B
        self.gen_buses = [1, 12, 21, 22, 26]  # 0-indexed: buses 2, 13, 22, 23, 27

    def eval(self, variables_values):
        """
        Evaluate OPF for active power loss minimization

        Objective:
        f = P_loss + α × ∑P_g + β × ∑(P_g - P_target)²

        Where:
        - P_loss: Real power loss = Re(V1 × I*1)
        - ∑P_g: Generation cost (fuel cost)
        - Generation setpoint mismatch penalty
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT DESIGN VARIABLES =====
        # Voltages at buses 2-30 (29 buses)
        V_real_230 = x[0:29]
        V_imag_230 = x[29:58]
        V_230 = V_real_230 + 1j * V_imag_230

        # Power setpoints
        Psp_230 = x[58:87]
        Qsp_230 = x[87:116]

        # Generator dispatch (at buses 2, 13, 22, 23, 27)
        Pg_dispatch = x[116:121]
        Qg_dispatch = x[121:126]

        # ===== CONSTRUCT FULL VECTORS =====
        V = np.zeros(30, dtype=complex)
        V[0] = 1.0 + 0.0j  # Slack bus
        V[1:30] = V_230

        Psp = np.zeros(30)
        Qsp = np.zeros(30)
        Psp[1:30] = Psp_230
        Qsp[1:30] = Qsp_230

        # Generator allocation on designated buses
        Pg = np.zeros(30)
        Qg = np.zeros(30)
        gen_bus_indices = [1, 12, 21, 22, 26]  # 0-indexed
        for idx, bus_idx in enumerate(gen_bus_indices):
            Pg[bus_idx] = Pg_dispatch[idx]
            Qg[bus_idx] = Qg_dispatch[idx]

        # ===== POWER FLOW ANALYSIS =====
        # Current injection: I = Y × V
        I = self.Y @ V

        # Real and imaginary components
        I_r = np.real(I)
        I_m = np.imag(I)

        # Specified current from power setpoints
        V_conj = np.conj(V) + 1e-10
        S_sp = Psp + 1j * Qsp
        I_sp = S_sp / V_conj
        I_sp_r = np.real(I_sp)
        I_sp_m = np.imag(I_sp)

        # Power balance equations
        delP = Psp - Pg + self.P  # Active power mismatch
        delQ = Qsp - Qg + self.Q  # Reactive power mismatch

        # Current mismatch (for constraint enforcement)
        delI_r = I_r - I_sp_r
        delI_m = I_m - I_sp_m

        # ===== OBJECTIVE FUNCTION =====
        # 1. Active power loss at slack bus
        S_loss = V[0] * np.conj(I[0])
        P_loss = np.real(S_loss)

        # 2. Generation cost (typically quadratic)
        # C(P_g) = α × P_g + β × P_g²
        alpha_cost = 0.1  # $/MW linear term
        beta_cost = 0.01  # $/MW² quadratic term
        gen_cost = np.sum(alpha_cost * Pg_dispatch + beta_cost * Pg_dispatch ** 2)

        # 3. Load following penalty (soft constraint)
        load_balance_penalty = np.sum((Psp_230 - self.P[1:30]) ** 2) * 0.1

        # Total objective
        f = P_loss + gen_cost + load_balance_penalty

        # ===== EQUALITY CONSTRAINTS =====
        # Power flow equations: f(V, θ) = 0
        # 29 buses × 4 equations per bus = 116 constraints
        h = np.concatenate([
            delI_r[1:30],  # Current real mismatch (29 constraints)
            delI_m[1:30],  # Current imag mismatch (29 constraints)
            delP[1:30],  # Active power balance (29 constraints)
            delQ[1:30]  # Reactive power balance (29 constraints)
        ])

        # ===== INEQUALITY CONSTRAINTS =====
        V_mag = np.abs(V)

        # 1. Voltage magnitude limits: 0.9 ≤ |V| ≤ 1.1 pu
        g_V_lower = 0.9 - V_mag  # g < 0: satisfied
        g_V_upper = V_mag - 1.1  # g < 0: satisfied

        # 2. Generator active power limits: 0 ≤ P_g ≤ P_g_max
        P_g_max = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        g_Pg_lower = -Pg_dispatch  # P_g ≥ 0
        g_Pg_upper = Pg_dispatch - P_g_max

        # 3. Generator reactive power limits: Q_g_min ≤ Q_g ≤ Q_g_max
        Q_g_min, Q_g_max = -1.0, 1.0
        g_Qg_lower = Q_g_min - Qg_dispatch
        g_Qg_upper = Qg_dispatch - Q_g_max

        # 4. Voltage stability: reactive power reserve
        Q_available = 2.0  # Max reactive power margin
        g_Q_stability = np.abs(Qg_dispatch) - Q_available

        # 5. Transmission line thermal limits (simplified)
        # |I_line| ≤ I_max (1.5 pu)
        I_max = 1.5
        g_line_thermal = np.abs(I) - I_max

        # Compile inequality constraints
        g_violations = np.concatenate([
            np.maximum(0, g_V_lower),  # Lower voltage limit violation
            np.maximum(0, g_V_upper),  # Upper voltage limit violation
            np.maximum(0, g_Pg_lower),  # Generator min power violation
            np.maximum(0, g_Pg_upper),  # Generator max power violation
            np.maximum(0, g_Qg_lower),  # Generator min reactive violation
            np.maximum(0, g_Qg_upper),  # Generator max reactive violation
            np.maximum(0, g_Q_stability),  # Reactive reserve violation
            np.maximum(0, g_line_thermal)  # Thermal limit violation
        ])

        # ===== CONSTRAINT VIOLATION PENALTIES =====
        # Equality constraints (power balance) - CRITICAL
        h_violation = np.sum(np.abs(h)) * 1e-2 + np.sum(h ** 2) * 1e-3

        # Inequality constraints - less strict
        g_violation = np.sum(g_violations ** 2)

        # ===== FITNESS FUNCTION =====
        penalty_h = 1e5  # High penalty for power balance violation
        penalty_g = 1e3  # Moderate penalty for other constraints

        fitness = f + penalty_h * h_violation + penalty_g * g_violation

        return fitness if np.isfinite(fitness) else 1e10


class RW37_Simplified(Function):
    """
    Simplified OPF without external data dependencies
    """

    def __init__(self, dim):
        n_vars = 126

        x_start = np.zeros(n_vars)
        x_start[0:29] = 1.0
        x_start[116:121] = 1.0

        domain = [
            *[(0.9, 1.1) for _ in range(29)],  # V_real
            *[(-0.3, 0.3) for _ in range(29)],  # V_imag
            *[(-2.0, 3.0) for _ in range(29)],  # P_sp
            *[(-2.0, 2.0) for _ in range(29)],  # Q_sp
            *[(0, 2.5) for _ in range(5)],  # P_g
            *[(-1.0, 1.0) for _ in range(5)],  # Q_g
        ]

        Function.__init__(
            self, dim, domain, x_start, 0,
            max_dimension=n_vars, name="RW37_Simplified"
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

    def eval(self, variables_values):
        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        V_r = x[0:29]
        V_i = x[29:58]
        V_230 = V_r + 1j * V_i

        Psp = x[58:87]
        Qsp = x[87:116]
        Pg = x[116:121]

        # Build voltage vector
        V = np.zeros(30, dtype=complex)
        V[0] = 1.0
        V[1:30] = V_230

        # Power flow
        I = self.Y @ V

        # Losses
        S_loss = V[0] * np.conj(I[0])
        P_loss = np.real(S_loss)

        # Generation cost
        gen_cost = np.sum(0.1 * Pg + 0.01 * Pg ** 2)

        # Objective
        f = P_loss + gen_cost

        # Penalty for constraint violations
        penalty = 1e3 * (
                np.sum((np.abs(V[1:30]) - 1.0) ** 2) +
                np.sum(np.maximum(0, -Pg) ** 2) +
                np.sum(np.maximum(0, Pg - 2.5) ** 2)
        )

        fitness = f + penalty

        return fitness if np.isfinite(fitness) else 1e10
