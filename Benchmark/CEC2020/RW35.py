from Benchmark.__Function import Function
import numpy as np
import os


class RW35(Function):
    def __init__(self, dim):
        """
        Optimal Sizing of Distributed Generation (DG)
        for Active Power Loss Minimization in Distribution Networks

        Problem: Determine DG sizing and reactive power setpoints to minimize
        active power losses at main transformer while maintaining voltage stability

        Variables (153 total):
        - V[2:38] (real & imag):  Voltage at buses 2-38 (37 buses × 2) = 74 vars
        - P_sp[2:38]:             Active power setpoints (37 buses) = 37 vars
        - Q_sp[2:38]:             Reactive power setpoints (37 buses) = 37 vars
        - P_dg:                   DG active power at buses 34-38 (5 buses) = 5 vars

        Total: 74 + 37 + 37 + 5 = 153 variables

        Network: 38-bus IEEE distribution test system
        Load model: Constant impedance with voltage-dependent characteristics

        Load = P0 * (V/Vref)^np_load  (voltage-dependent ZIP model)
        """

        self.load_network_data()

        n_vars = 153

        # Initial conditions
        x_start = np.zeros(n_vars)
        x_start[0:37] = 0.95  # V_real (buses 2-38)
        x_start[37:74] = 0.0  # V_imag
        x_start[74:111] = 0.0  # P_sp
        x_start[111:148] = 0.0  # Q_sp
        x_start[148:153] = 0.0  # P_dg

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
        ]

        f_x_start = 0
        name = "RW35"
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
        Load 38-bus IEEE test system network data
        with voltage-dependent load characteristics

        Load format:
        P = [P0, Pp, Pq, Pz, Vref, np_load]
        Q = [Q0, Qp, Qq, Qz, Vref, nq_load]

        ZIP model: P_load = Pp·(V/Vref)¹ + Pq·(V/Vref)² + Pz·(V/Vref)⁰
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

        except:
            print("⚠️  Network data files not found. Using synthetic 38-bus system.")

            n_bus = 38

            # Create realistic admittance matrix
            np.random.seed(42)

            # Conductance matrix (diagonal dominant)
            self.G = np.eye(n_bus) * 0.6 + np.random.randn(n_bus, n_bus) * 0.02
            self.G = (self.G + self.G.T) / 2

            # Susceptance matrix (diagonal dominant, negative)
            self.B = -np.eye(n_bus) * 2.2 + np.random.randn(n_bus, n_bus) * 0.08
            self.B = (self.B + self.B.T) / 2

            # Slack bus (bus 1): zero injection
            self.G[0, :] = 0.0
            self.B[0, :] = 0.0

            # Load profiles with ZIP characteristics
            # P = [P0, Pp, Pq, Pz, Vref, np]
            # Q = [Q0, Qp, Qq, Qz, Vref, nq]

            P0_values = np.random.uniform(0.1, 3.0, n_bus)
            Q0_values = np.random.uniform(-1.0, 1.0, n_bus)

            self.P = np.zeros((n_bus, 6))
            self.Q = np.zeros((n_bus, 6))

            for i in range(n_bus):
                # Active power ZIP model
                self.P[i, 0] = P0_values[i]  # Total load
                self.P[i, 1] = 0.6 * P0_values[i]  # Current-dependent (linear)
                self.P[i, 2] = 0.3 * P0_values[i]  # Impedance-dependent (quadratic)
                self.P[i, 3] = 0.1 * P0_values[i]  # Constant power
                self.P[i, 4] = 1.0  # Reference voltage
                self.P[i, 5] = 1.0  # Voltage exponent

                # Reactive power ZIP model
                self.Q[i, 0] = Q0_values[i]
                self.Q[i, 1] = 0.5 * Q0_values[i]
                self.Q[i, 2] = 0.4 * Q0_values[i]
                self.Q[i, 3] = 0.1 * Q0_values[i]
                self.Q[i, 4] = 1.0
                self.Q[i, 5] = 1.5

            # Slack bus: zero load
            self.P[0, :] = 0.0
            self.Q[0, :] = 0.0

        self.Y = self.G + 1j * self.B

    def eval(self, variables_values):
        """
        Evaluate DG sizing optimization problem

        Objective: Minimize active power losses at main transformer
                   + sum of active power setpoints (DG cost)

        f = P_loss(bus 1) + ∑P_sp

        Constraints: Power flow equations (equality)
                     Voltage limits (implicit via domain)
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT DESIGN VARIABLES =====
        # Voltage at buses 2-38 (37 buses)
        V_real_238 = x[0:37]
        V_imag_238 = x[37:74]
        V_238 = V_real_238 + 1j * V_imag_238

        # Power setpoints at buses 2-38
        Psp_238 = x[74:111]
        Qsp_238 = x[111:148]

        # DG active power at buses 34-38 (5 generators)
        # Buses 34-38 are indices 33-37 (0-indexed)
        Pdg_vector = np.zeros(38)
        Pdg_vector[[33, 34, 35, 36, 37]] = x[148:153]

        # ===== CONSTRUCT FULL VOLTAGE VECTOR =====
        V = np.zeros(38, dtype=complex)
        V[0] = 1.0  # Slack bus (bus 1)
        V[1:38] = V_238

        # ===== CONSTRUCT POWER SETPOINT VECTORS =====
        Psp = np.zeros(38)
        Qsp = np.zeros(38)
        Psp[1:38] = Psp_238
        Qsp[1:38] = Qsp_238

        # ===== VOLTAGE-DEPENDENT LOAD MODEL =====
        # P_load = P0 * (V/Vref)^np
        # ZIP format: P = P[0] * ((V/V_ref)^P[5] + Pp + Pq + Pz)
        # Simplified: P_load = P[0] * (|V|/P[4])^P[5]

        V_mag = np.abs(V) + 1e-10

        # Load active power (voltage-dependent)
        P_load = self.P[:, 0] * (V_mag / (self.P[:, 4] + 1e-10)) ** self.P[:, 5]

        # Load reactive power (voltage-dependent)
        Q_load = self.Q[:, 0] * (V_mag / (self.Q[:, 4] + 1e-10)) ** self.Q[:, 5]

        # ===== NETWORK ANALYSIS =====
        # Current injection: I = Y × V
        I = self.Y @ V

        # Real and imaginary parts
        I_r = np.real(I)
        I_m = np.imag(I)

        # Specified current from setpoints: I_sp = (P_sp + jQ_sp) / V*
        # Avoid division by zero
        V_conj = np.conj(V) + 1e-10
        S_sp = Psp + 1j * Qsp
        I_sp = S_sp / V_conj
        I_sp_r = np.real(I_sp)
        I_sp_m = np.imag(I_sp)

        # Power balance errors
        # Active power: P_sp - P_dg - P_load = 0
        # Reactive power: Q_sp + Q_load = 0
        delP = Psp - Pdg_vector - P_load
        delQ = Qsp + Q_load

        # Current mismatch
        delI_r = I_r - I_sp_r
        delI_m = I_m - I_sp_m

        # ===== OBJECTIVE FUNCTION =====
        # Active power loss at main transformer (bus 1)
        # P_loss = Re(V1 * conj(I1))
        P_loss = np.real(V[0] * np.conj(I[0]))

        # DG operation cost (sum of active power injections)
        P_dg_total = np.sum(Pdg_vector)

        # Total objective: minimize losses + encourage DG dispersion
        f = P_loss + 0.1 * P_dg_total

        # ===== EQUALITY CONSTRAINTS =====
        # Power flow equations at all buses 2-38 (37 buses)
        # h = [delI_r(2:38), delI_m(2:38), delP(2:38), delQ(2:38)]
        h = np.concatenate([
            delI_r[1:38],  # Current real mismatch (37)
            delI_m[1:38],  # Current imag mismatch (37)
            delP[1:38],  # Active power balance (37)
            delQ[1:38]  # Reactive power balance (37)
        ])

        # ===== INEQUALITY CONSTRAINTS =====
        # Voltage magnitude limits: 0.8 ≤ |V| ≤ 1.1 per unit
        V_mag = np.abs(V)
        g_V_min = 0.8 - V_mag
        g_V_max = V_mag - 1.1

        # Current limits at DG buses
        dg_buses = [33, 34, 35, 36, 37]
        I_dg_limits = np.abs(I[dg_buses]) - 2.0

        # DG power non-negativity (should be satisfied by domain)
        g_Pdg_min = -Pdg_vector[[33, 34, 35, 36, 37]]

        g = np.concatenate([
            np.maximum(0, g_V_min),
            np.maximum(0, g_V_max),
            np.maximum(0, I_dg_limits),
            np.maximum(0, g_Pdg_min)
        ])

        # ===== CONSTRAINT VIOLATION PENALTY =====
        # Equality constraints are critical (power balance)
        h_violation = np.sum(np.abs(h)) * 1e-2 + np.sum(h ** 2) * 1e-3

        # Inequality constraints
        g_violation = np.sum(np.maximum(0, g) ** 2)

        # ===== FITNESS EVALUATION =====
        penalty_weight_h = 1e4
        penalty_weight_g = 1e3

        fitness = f + penalty_weight_h * h_violation + penalty_weight_g * g_violation

        return fitness if np.isfinite(fitness) else 1e10


class RW35_Simplified(Function):
    """
    Simplified version without external data files
    for easier testing and validation
    """

    def __init__(self, dim):
        n_vars = 153

        x_start = np.zeros(n_vars)
        x_start[0:37] = 0.95

        domain = [
            *[(0.8, 1.1) for _ in range(37)],
            *[(-0.5, 0.5) for _ in range(37)],
            *[(0, 3.0) for _ in range(37)],
            *[(-1.5, 1.5) for _ in range(37)],
            *[(0, 1.5) for _ in range(5)],
        ]

        Function.__init__(
            self, dim, domain, x_start, 0,
            max_dimension=n_vars, name="RW35_SimpleLoss"
        )
        self.types(['CEC2020RW'])

        # Synthetic 38-bus network
        np.random.seed(42)
        n_bus = 38

        # Admittance matrix
        self.G = np.eye(n_bus) * 0.6 + np.random.randn(n_bus, n_bus) * 0.02
        self.G = (self.G + self.G.T) / 2

        self.B = -np.eye(n_bus) * 2.2 + np.random.randn(n_bus, n_bus) * 0.08
        self.B = (self.B + self.B.T) / 2

        self.G[0, :] = 0.0
        self.B[0, :] = 0.0

        self.Y = self.G + 1j * self.B

        # Load with voltage-dependent characteristics
        # P = [P0, Pp, Pq, Pz, Vref, np_load]
        self.P = np.zeros((n_bus, 6))
        self.Q = np.zeros((n_bus, 6))

        P0_values = np.random.uniform(0.1, 3.0, n_bus)
        Q0_values = np.random.uniform(-1.0, 1.0, n_bus)

        for i in range(n_bus):
            self.P[i] = [P0_values[i], 0.6 * P0_values[i], 0.3 * P0_values[i],
                         0.1 * P0_values[i], 1.0, 1.0]
            self.Q[i] = [Q0_values[i], 0.5 * Q0_values[i], 0.4 * Q0_values[i],
                         0.1 * Q0_values[i], 1.0, 1.5]

        self.P[0, :] = 0.0
        self.Q[0, :] = 0.0

    def eval(self, variables_values):
        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # Extract variables
        V_r = x[0:37]
        V_i = x[37:74]
        V_238 = V_r + 1j * V_i

        Psp = x[74:111]
        Qsp = x[111:148]
        Pdg = x[148:153]

        # Build voltage vector
        V = np.zeros(38, dtype=complex)
        V[0] = 1.0
        V[1:38] = V_238

        # Voltage-dependent loads
        V_mag = np.abs(V) + 1e-10
        P_load = self.P[:, 0] * (V_mag / self.P[:, 4]) ** self.P[:, 5]

        # Network analysis
        I = self.Y @ V

        # Power loss at slack bus
        P_loss = np.real(V[0] * np.conj(I[0]))

        # DG cost
        P_dg_total = np.sum(Pdg)

        # Objective
        f = P_loss + 0.1 * P_dg_total

        # Penalty for constraint violations
        penalty = 1e3 * (
                np.sum((np.abs(V[1:38]) - 0.95) ** 2) +
                np.sum(np.maximum(0, -Pdg) ** 2) +
                np.sum(np.maximum(0, P_load[1:38] - 5.0) ** 2)
        )

        fitness = f + penalty

        return fitness if np.isfinite(fitness) else 1e10
