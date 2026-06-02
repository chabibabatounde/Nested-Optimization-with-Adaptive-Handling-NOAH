from Benchmark.__Function import Function
import numpy as np
import os


class RW36(Function):
    def __init__(self, dim):
        """
        Optimal Sizing of Distributed Generation (DG) and Capacitors
        for Reactive Power Loss Minimization in Distribution Networks

        Problem: Determine DG active/reactive power and capacitor sizing
        to minimize both active AND reactive power losses at main transformer

        Variables (158 total):
        - V[2:38] (real & imag):     Voltage at buses 2-38 (37 buses × 2) = 74 vars
        - P_sp[2:38]:                Active power setpoints (37 buses) = 37 vars
        - Q_sp[2:38]:                Reactive power setpoints (37 buses) = 37 vars
        - P_dg:                      DG active power at buses 34-38 (5 DGs) = 5 vars
        - Q_dg:                      DG reactive power at buses 34-38 (5 DGs) = 5 vars

        Total: 74 + 37 + 37 + 5 + 5 = 158 variables

        Network: 38-bus IEEE distribution system
        Load model: Voltage-dependent (ZIP model)
        Control: DG with reactive power capability (PV bus behavior)
        """

        self.load_network_data()

        n_vars = 158

        # Initial conditions
        x_start = np.zeros(n_vars)
        x_start[0:37] = 0.95  # V_real (buses 2-38)
        x_start[37:74] = 0.0  # V_imag
        x_start[74:111] = 0.0  # P_sp
        x_start[111:148] = 0.0  # Q_sp
        x_start[148:153] = 0.0  # P_dg
        x_start[153:158] = 0.0  # Q_dg

        # Domain bounds
        domain = [
            # Voltage real parts: V_r ∈ [0.8, 1.1]
            *[(0.8, 1.1) for _ in range(37)],
            # Voltage imag parts: V_i ∈ [-0.5, 0.5]
            *[(-0.5, 0.5) for _ in range(37)],
            # Active power setpoints: P_sp ∈ [0, 3.0] MW
            *[(0, 3.0) for _ in range(37)],
            # Reactive power setpoints: Q_sp ∈ [-1.5, 1.5] MVAr
            *[(-1.5, 1.5) for _ in range(37)],
            # DG active power (buses 34-38): P_dg ∈ [0, 1.5] MW
            *[(0, 1.5) for _ in range(5)],
            # DG reactive power (buses 34-38): Q_dg ∈ [-1.0, 1.0] MVAr
            *[(-1.0, 1.0) for _ in range(5)],
        ]

        f_x_start = 0
        name = "RW36"
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
        Load 38-bus distribution network with voltage-dependent loads

        Load model: ZIP format
        P = P0 * (V/Vref)^np   (voltage-dependent active power)
        Q = Q0 * (V/Vref)^nq   (voltage-dependent reactive power)

        File format:
        P = [P0, Pp, Pq, Pz, Vref, np_load] for each bus
        Q = [Q0, Qp, Qq, Qz, Vref, nq_load] for each bus
        """
        try:
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            self.G = np.loadtxt(os.path.join(base_path, "FunctionPS2_G.txt"))
            self.B = np.loadtxt(os.path.join(base_path, "FunctionPS2_B.txt"))
            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS2_P.txt"))
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS2_Q.txt"))

            # Validate dimensions
            assert self.G.shape == (38, 38), f"G shape {self.G.shape} != (38, 38)"
            assert self.P.shape[0] == 38, f"P rows {self.P.shape[0]} != 38"
            assert self.Q.shape[0] == 38, f"Q rows {self.Q.shape[0]} != 38"


        except Exception as e:
            print(f"⚠️  Network data files not found: {e}")
            print("    Using synthetic 38-bus system")

            n_bus = 38
            np.random.seed(42)

            # ===== ADMITTANCE MATRIX =====
            # Conductance: G[i,j] > 0 (resistive elements)
            self.G = np.eye(n_bus) * 0.6 + np.random.randn(n_bus, n_bus) * 0.02
            self.G = (self.G + self.G.T) / 2

            # Susceptance: B[i,j] < 0 (inductive elements)
            self.B = -np.eye(n_bus) * 2.2 + np.random.randn(n_bus, n_bus) * 0.08
            self.B = (self.B + self.B.T) / 2

            # Slack bus: no shunt admittance
            self.G[0, :] = 0.0
            self.B[0, :] = 0.0

            # ===== LOAD DATA WITH ZIP CHARACTERISTICS =====
            # Each load has format: [P0, Pp, Pq, Pz, Vref, np_load]
            self.P = np.zeros((n_bus, 6))
            self.Q = np.zeros((n_bus, 6))

            # Random load profiles
            P0_values = np.random.uniform(0.1, 3.0, n_bus)
            Q0_values = np.random.uniform(-1.0, 1.0, n_bus)

            for i in range(n_bus):
                # Active power: total decomposed into ZIP components
                self.P[i, 0] = P0_values[i]  # Total load
                self.P[i, 1] = 0.6 * P0_values[i]  # Current-dependent (linear)
                self.P[i, 2] = 0.3 * P0_values[i]  # Impedance-dependent (quadratic)
                self.P[i, 3] = 0.1 * P0_values[i]  # Constant power
                self.P[i, 4] = 1.0  # Reference voltage (pu)
                self.P[i, 5] = 1.0  # Voltage exponent

                # Reactive power: similar decomposition
                self.Q[i, 0] = Q0_values[i]
                self.Q[i, 1] = 0.5 * Q0_values[i]
                self.Q[i, 2] = 0.4 * Q0_values[i]
                self.Q[i, 3] = 0.1 * Q0_values[i]
                self.Q[i, 4] = 1.0
                self.Q[i, 5] = 1.5  # Reactive more voltage-sensitive

            # Slack bus: no load
            self.P[0, :] = 0.0
            self.Q[0, :] = 0.0

        self.Y = self.G + 1j * self.B

    def eval(self, variables_values):
        """
        Evaluate combined active and reactive power loss minimization

        Objective function:
        f = 0.5 × (P_loss + ∑P_sp) + 0.5 × (Q_loss + ∑Q_sp)

        Weighted objective minimizes:
        - Active power losses (network resistance)
        - Reactive power losses (network reactance)
        - DG power output (operation cost)
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT DESIGN VARIABLES =====
        # Voltages at buses 2-38
        V_real_238 = x[0:37]
        V_imag_238 = x[37:74]
        V_238 = V_real_238 + 1j * V_imag_238

        # Power setpoints at buses 2-38
        Psp_238 = x[74:111]
        Qsp_238 = x[111:148]

        # DG power at buses 34-38 (5 generators)
        Pdg_vector = np.zeros(38)
        Qdg_vector = np.zeros(38)
        Pdg_vector[[33, 34, 35, 36, 37]] = x[148:153]
        Qdg_vector[[33, 34, 35, 36, 37]] = x[153:158]

        # ===== CONSTRUCT FULL VECTORS =====
        V = np.zeros(38, dtype=complex)
        V[0] = 1.0
        V[1:38] = V_238

        Psp = np.zeros(38)
        Qsp = np.zeros(38)
        Psp[1:38] = Psp_238
        Qsp[1:38] = Qsp_238

        # ===== VOLTAGE-DEPENDENT LOADS =====
        V_mag = np.abs(V) + 1e-10

        # Active load: P_load = P0 × (V/Vref)^np
        P_load = self.P[:, 0] * np.power(V_mag / (self.P[:, 4] + 1e-10),
                                         self.P[:, 5])

        # Reactive load: Q_load = Q0 × (V/Vref)^nq
        Q_load = self.Q[:, 0] * np.power(V_mag / (self.Q[:, 4] + 1e-10),
                                         self.Q[:, 5])

        # ===== POWER FLOW ANALYSIS =====
        # Current injection: I = Y × V
        I = self.Y @ V

        # Real and imaginary parts
        I_r = np.real(I)
        I_m = np.imag(I)

        # Specified current from power setpoints
        V_conj = np.conj(V) + 1e-10
        S_sp = Psp + 1j * Qsp
        I_sp = S_sp / V_conj
        I_sp_r = np.real(I_sp)
        I_sp_m = np.imag(I_sp)

        # Power balance errors
        delP = Psp - Pdg_vector + P_load
        delQ = Qsp - Qdg_vector + Q_load

        # Current mismatch (for constraint enforcement)
        delI_r = I_r - I_sp_r
        delI_m = I_m - I_sp_m

        # ===== OBJECTIVE FUNCTION =====
        # Power losses at slack bus
        S_loss = V[0] * np.conj(I[0])
        P_loss = np.real(S_loss)  # Active power loss
        Q_loss = np.imag(S_loss)  # Reactive power loss

        # DG output (operation cost)
        P_dg_total = np.sum(Pdg_vector)
        Q_dg_total = np.sum(Qdg_vector)

        # Total setpoint powers
        P_sp_total = np.sum(Psp_238)
        Q_sp_total = np.sum(Qsp_238)

        # Combined objective: weighted average
        # 50% active loss minimization + 50% reactive loss minimization
        f_active = P_loss + 0.05 * P_sp_total
        f_reactive = Q_loss + 0.05 * Q_sp_total

        f = 0.5 * f_active + 0.5 * f_reactive

        # ===== EQUALITY CONSTRAINTS =====
        # Power flow equations at buses 2-38 (37 buses, 4 equations each)
        h = np.concatenate([
            delI_r[1:38],  # Current real mismatch (37)
            delI_m[1:38],  # Current imag mismatch (37)
            delP[1:38],  # Active power balance (37)
            delQ[1:38]  # Reactive power balance (37)
        ])

        # ===== INEQUALITY CONSTRAINTS =====
        V_mag = np.abs(V)

        # Voltage magnitude limits: 0.8 ≤ |V| ≤ 1.1 pu
        g_V_min = 0.8 - V_mag
        g_V_max = V_mag - 1.1

        # Current limits at DG buses (2.0 pu base)
        dg_buses = [33, 34, 35, 36, 37]
        I_dg_limits = np.abs(I[dg_buses]) - 2.0

        # DG power non-negativity
        g_Pdg_min = -Pdg_vector[dg_buses]

        # DG reactive power limits (within ±1.0 MVAr)
        g_Qdg_upper = np.abs(Qdg_vector[dg_buses]) - 1.0

        # Reactive power balancing: limit reactive injection
        g_Q_balance = np.abs(Q_loss) - 0.5

        g = np.concatenate([
            np.maximum(0, g_V_min),  # Lower voltage limit violation
            np.maximum(0, g_V_max),  # Upper voltage limit violation
            np.maximum(0, I_dg_limits),  # DG current limits
            np.maximum(0, g_Pdg_min),  # DG active power non-negativity
            np.maximum(0, g_Qdg_upper),  # DG reactive power limits
            [np.maximum(0, g_Q_balance)]  # Reactive loss limit
        ])

        # ===== CONSTRAINT VIOLATION PENALTIES =====
        # Equality constraints (power balance) are critical
        h_violation = np.sum(np.abs(h)) * 1e-2 + np.sum(h ** 2) * 1e-3

        # Inequality constraints
        g_violation = np.sum(np.maximum(0, g) ** 2)

        # ===== FITNESS FUNCTION =====
        penalty_weight_h = 1e4
        penalty_weight_g = 1e3

        fitness = f + penalty_weight_h * h_violation + penalty_weight_g * g_violation

        return fitness if np.isfinite(fitness) else 1e10


class RW36_Simplified(Function):
    """
    Simplified version of RC36 without external data dependencies
    For rapid prototyping and testing
    """

    def __init__(self, dim):
        n_vars = 158

        x_start = np.zeros(n_vars)
        x_start[0:37] = 0.95

        domain = [
            *[(0.8, 1.1) for _ in range(37)],  # V_real
            *[(-0.5, 0.5) for _ in range(37)],  # V_imag
            *[(0, 3.0) for _ in range(37)],  # P_sp
            *[(-1.5, 1.5) for _ in range(37)],  # Q_sp
            *[(0, 1.5) for _ in range(5)],  # P_dg
            *[(-1.0, 1.0) for _ in range(5)],  # Q_dg
        ]

        Function.__init__(
            self, dim, domain, x_start, 0,
            max_dimension=n_vars, name="RW36_Simplified"
        )
        self.types(['CEC2020RW'])

        # Synthetic 38-bus network
        np.random.seed(42)
        n_bus = 38

        self.G = np.eye(n_bus) * 0.6 + np.random.randn(n_bus, n_bus) * 0.02
        self.G = (self.G + self.G.T) / 2

        self.B = -np.eye(n_bus) * 2.2 + np.random.randn(n_bus, n_bus) * 0.08
        self.B = (self.B + self.B.T) / 2

        self.G[0, :] = 0.0
        self.B[0, :] = 0.0

        self.Y = self.G + 1j * self.B

        # Voltage-dependent loads
        self.P = np.zeros((n_bus, 6))
        self.Q = np.zeros((n_bus, 6))

        P0 = np.random.uniform(0.1, 3.0, n_bus)
        Q0 = np.random.uniform(-1.0, 1.0, n_bus)

        for i in range(n_bus):
            self.P[i] = [P0[i], 0.6 * P0[i], 0.3 * P0[i], 0.1 * P0[i], 1.0, 1.0]
            self.Q[i] = [Q0[i], 0.5 * Q0[i], 0.4 * Q0[i], 0.1 * Q0[i], 1.0, 1.5]

        self.P[0, :] = 0.0
        self.Q[0, :] = 0.0

    def eval(self, variables_values):
        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        V_r = x[0:37]
        V_i = x[37:74]
        V_238 = V_r + 1j * V_i

        Psp = x[74:111]
        Qsp = x[111:148]
        Pdg = x[148:153]
        Qdg = x[153:158]

        # Build voltage vector
        V = np.zeros(38, dtype=complex)
        V[0] = 1.0
        V[1:38] = V_238

        # Voltage-dependent loads
        V_mag = np.abs(V) + 1e-10
        P_load = self.P[:, 0] * np.power(V_mag / self.P[:, 4], self.P[:, 5])
        Q_load = self.Q[:, 0] * np.power(V_mag / self.Q[:, 4], self.Q[:, 5])

        # Power flow
        I = self.Y @ V

        # Losses
        S_loss = V[0] * np.conj(I[0])
        P_loss = np.real(S_loss)
        Q_loss = np.imag(S_loss)

        # Objective: weighted combination
        f_active = P_loss + 0.05 * np.sum(Psp)
        f_reactive = Q_loss + 0.05 * np.sum(Qsp)

        f = 0.5 * f_active + 0.5 * f_reactive

        # Penalty for constraint violations
        penalty = 1e3 * (
                np.sum((np.abs(V[1:38]) - 0.95) ** 2) +
                np.sum(np.maximum(0, -Pdg) ** 2) +
                np.sum(np.maximum(0, np.abs(Qdg) - 1.0) ** 2)
        )

        fitness = f + penalty

        return fitness if np.isfinite(fitness) else 1e10
