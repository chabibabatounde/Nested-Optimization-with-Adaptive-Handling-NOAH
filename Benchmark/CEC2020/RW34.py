from Benchmark.Functions.__Function import Function
import numpy as np
from scipy.io import loadmat
import os


class RW34(Function):
    def __init__(self, dim):
        """
        Optimal Sizing of Single Phase Distributed Generation (DG)
        with Reactive Power Support for Phase Balancing

        Problem: Design DG placement and sizing on 3-phase distribution network
        to minimize current unbalance at main transformer

        Variables (118 total):
        - V[4:30] (real & imag):  Voltage at buses 4-30 (27 buses × 2) = 54 vars
        - P_sp[4:30]:  Active power setpoints (27 buses) = 27 vars
        - Q_sp[4:30]:  Reactive power setpoints (27 buses) = 27 vars
        - P_dg, Q_dg:  DG active/reactive power at selected buses = 10 vars

        Total: 54 + 27 + 27 + 10 = 118 variables
        """

        # ===== LOAD NETWORK DATA =====
        self.load_network_data()

        # Initialize design variables
        n_vars = 118
        x_start = np.zeros(n_vars)

        # Initial conditions
        x_start[0:27] = 0.95  # V_real (buses 4-30)
        x_start[27:54] = 0.0  # V_imag
        x_start[54:81] = 0.0  # P_sp
        x_start[81:108] = 0.0  # Q_sp
        x_start[108:113] = 0.0  # P_dg
        x_start[113:118] = 0.0  # Q_dg

        # Domain bounds for each variable
        domain = [
            # Voltage real parts: V_r ∈ [0.8, 1.1]
            *[(0.8, 1.1) for _ in range(27)],
            # Voltage imag parts: V_i ∈ [-0.5, 0.5]
            *[(-0.5, 0.5) for _ in range(27)],
            # Active power setpoints: P_sp ∈ [0, 5.0] MW
            *[(0, 5.0) for _ in range(27)],
            # Reactive power setpoints: Q_sp ∈ [-2.0, 2.0] MVAr
            *[(-2.0, 2.0) for _ in range(27)],
            # DG active power: P_dg ∈ [0, 2.0] MW (5 buses)
            *[(0, 2.0) for _ in range(5)],
            # DG reactive power: Q_dg ∈ [-1.0, 1.0] MVAr (5 buses)
            *[(-1.0, 1.0) for _ in range(5)],
        ]

        f_x_start = 0
        name = "RW34"
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
        Load 30-bus IEEE test system network data
        Y = G + jB (admittance matrix)
        P, Q : active/reactive power loads
        """
        try:

            # Try to load from files
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            self.G = np.loadtxt(os.path.join(base_path, "FunctionPS1_G.txt"))
            self.B = np.loadtxt(os.path.join(base_path, "FunctionPS1_B.txt"))
            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS1_P.txt"))
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS1_Q.txt"))


        except:
            # Fallback: Initialize 30-bus system with default values
            print("⚠️  Network data files not found. Using default 30-bus IEEE system.")

            n_bus = 30

            # Admittance matrix (sparse representation)
            # Diagonal dominant for stability
            self.G = np.eye(n_bus) * 0.5 + np.random.randn(n_bus, n_bus) * 0.01
            self.B = -np.eye(n_bus) * 2.0 + np.random.randn(n_bus, n_bus) * 0.05

            # Make symmetric
            self.G = (self.G + self.G.T) / 2
            self.B = (self.B + self.B.T) / 2

            # Load profiles (example values)
            self.P = np.random.uniform(0.1, 2.0, n_bus)
            self.Q = np.random.uniform(-1.0, 1.0, n_bus)

            # Zero injection at slack bus (bus 1)
            self.P[0] = 0.0
            self.Q[0] = 0.0

        self.Y = self.G + 1j * self.B

    def eval(self, variables_values):
        """
        Evaluate DG sizing optimization problem

        Objective: Minimize current unbalance at main transformer (3-phase)

        Constraints: Power flow equations at all buses (equality constraints)

        Three-phase voltage reference:
        V_a = 1.0 ∠0°
        V_b = 1.0 ∠240° = exp(j4π/3)
        V_c = 1.0 ∠120° = exp(j2π/3)
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT DESIGN VARIABLES =====
        # Voltage at buses 4-30 (complex)
        V_real_427 = x[0:27]  # Real parts
        V_imag_427 = x[27:54]  # Imaginary parts
        V_427 = V_real_427 + 1j * V_imag_427

        # Power setpoints at buses 4-30
        Psp_427 = x[54:81]  # Active power
        Qsp_427 = x[81:108]  # Reactive power

        # DG power at selected buses (9, 16, 21, 24, 30)
        dg_buses_idx = [8, 15, 20, 23, 29]  # 0-indexed
        Pdg_dg = x[108:113]
        Qdg_dg = x[113:118]

        # ===== CONSTRUCT FULL VOLTAGE VECTOR =====
        V = np.zeros(30, dtype=complex)

        # Slack bus (bus 1): 3-phase reference
        V[0] = 1.0  # Phase A

        # Phase B and C references (120° and 240° shifts)
        V_B_ref = np.exp(1j * 2 * np.pi / 3)  # 120°
        V_C_ref = np.exp(1j * 4 * np.pi / 3)  # 240°

        # Buses 2-3: Second and third phase references (simplified)
        V[1] = V_B_ref
        V[2] = V_C_ref

        # Buses 4-30: Design variables
        V[3:30] = V_427

        # ===== CONSTRUCT POWER INJECTION VECTORS =====
        P_sp = np.zeros(30)
        Q_sp = np.zeros(30)

        P_sp[3:30] = Psp_427
        Q_sp[3:30] = Qsp_427

        # ===== CONSTRUCT DG VECTORS =====
        P_dg = np.zeros(30)
        Q_dg = np.zeros(30)

        for idx, bus in enumerate(dg_buses_idx):
            P_dg[bus] = Pdg_dg[idx]
            Q_dg[bus] = Qdg_dg[idx]

        # ===== NETWORK ANALYSIS =====
        # Current calculation: I = Y × V
        I = self.Y @ V
        I_r = np.real(I)
        I_m = np.imag(I)

        # Specified current from power setpoints: I_sp = (P_sp + jQ_sp) / V*
        S_sp = P_sp + 1j * Q_sp
        I_sp = np.conj(S_sp / (np.abs(V) + 1e-10))
        I_sp_r = np.real(I_sp)
        I_sp_m = np.imag(I_sp)

        # Power balance errors
        delP = P_sp - P_dg - self.P
        delQ = Q_sp - Q_dg - self.Q

        # Current mismatch
        delI_r = I_r - I_sp_r
        delI_m = I_m - I_sp_m

        # ===== OBJECTIVE FUNCTION =====
        # Minimize current unbalance at main transformer (bus 1)
        # Unbalance factors (positive and negative sequences)

        # Three-phase currents at slack bus
        I_a = I[0]
        I_b = I[1]
        I_c = I[2]

        # Positive sequence: I+ = (Ia + a*Ib + a²*Ic) / 3
        a = np.exp(1j * 2 * np.pi / 3)
        I_pos = (I_a + a * I_b + a ** 2 * I_c) / 3.0

        # Negative sequence: I- = (Ia + a²*Ib + a*Ic) / 3
        I_neg = (I_a + a ** 2 * I_b + a * I_c) / 3.0

        # Unbalance metric: ratio of negative to positive sequence magnitude
        I_pos_mag = np.abs(I_pos) + 1e-10
        I_neg_mag = np.abs(I_neg)

        unbalance_factor = I_neg_mag / I_pos_mag

        # Objective: minimize unbalance and total imported current
        f = np.abs(I[0]) + unbalance_factor

        # ===== EQUALITY CONSTRAINTS =====
        # Power flow equations at all buses (24 buses 4-30: 27×2 + 27 real/imag)
        # h = [delI_r(4:30), delI_m(4:30), delP(4:30), delQ(4:30)]
        h = np.concatenate([
            delI_r[3:30],  # Current real mismatch (27)
            delI_m[3:30],  # Current imag mismatch (27)
            delP[3:30],  # Power active balance (27)
            delQ[3:30]  # Power reactive balance (27)
        ])

        # ===== INEQUALITY CONSTRAINTS =====
        # Voltage magnitude limits: 0.8 ≤ |V| ≤ 1.1 per unit
        V_mag = np.abs(V)
        g_V_min = 0.8 - V_mag
        g_V_max = V_mag - 1.1

        # Current limits at DG buses (example: max 2.0 per unit)
        I_dg_limits = np.abs(I[dg_buses_idx]) - 2.0

        g = np.concatenate([
            np.maximum(0, g_V_min),
            np.maximum(0, g_V_max),
            np.maximum(0, I_dg_limits)
        ])

        # ===== CONSTRAINT VIOLATION PENALTY =====
        h_violation = np.sum(h ** 2)
        g_violation = np.sum(np.maximum(0, g) ** 2)

        # ===== FITNESS EVALUATION =====
        fitness = f + 1e3 * h_violation + 1e3 * g_violation

        return fitness if np.isfinite(fitness) else 1e10


class RW34_SimpleDG(Function):
    """
    Simplified version without external data files
    """

    def __init__(self, dim):
        n_vars = 118

        x_start = np.zeros(n_vars)
        x_start[0:27] = 0.95

        domain = [
            *[(0.8, 1.1) for _ in range(27)],
            *[(-0.5, 0.5) for _ in range(27)],
            *[(0, 5.0) for _ in range(27)],
            *[(-2.0, 2.0) for _ in range(27)],
            *[(0, 2.0) for _ in range(5)],
            *[(-1.0, 1.0) for _ in range(5)],
        ]

        Function.__init__(
            self, dim, domain, x_start, 0,
            max_dimension=n_vars, name="RW34_SimpleDG"
        )
        self.types(['CEC2020RW'])

        # Create synthetic 30-bus network
        np.random.seed(42)
        n_bus = 30
        self.G = np.eye(n_bus) * 0.5 + np.random.randn(n_bus, n_bus) * 0.01
        self.B = -np.eye(n_bus) * 2.0 + np.random.randn(n_bus, n_bus) * 0.05
        self.G = (self.G + self.G.T) / 2
        self.B = (self.B + self.B.T) / 2
        self.Y = self.G + 1j * self.B
        self.P = np.random.uniform(0.1, 2.0, n_bus)
        self.Q = np.random.uniform(-1.0, 1.0, n_bus)
        self.P[0] = 0.0
        self.Q[0] = 0.0

    def eval(self, variables_values):
        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # Extract variables
        V_r = x[0:27]
        V_i = x[27:54]
        V_427 = V_r + 1j * V_i

        Psp = x[54:81]
        Qsp = x[81:108]

        Pdg = x[108:113]
        Qdg = x[113:118]

        # Build voltage vector
        V = np.zeros(30, dtype=complex)
        V[0] = 1.0
        V[1] = np.exp(1j * 2 * np.pi / 3)
        V[2] = np.exp(1j * 4 * np.pi / 3)
        V[3:30] = V_427

        # Power flows
        I = self.Y @ V

        # Unbalance at slack bus
        I_a, I_b, I_c = I[0], I[1], I[2]
        a = np.exp(1j * 2 * np.pi / 3)
        I_pos = (I_a + a * I_b + a ** 2 * I_c) / 3.0
        I_neg = (I_a + a ** 2 * I_b + a * I_c) / 3.0

        unbalance = np.abs(I_neg) / (np.abs(I_pos) + 1e-10)

        # Objective
        f = np.abs(I_a) + unbalance

        # Penalty for constraint violations
        penalty = 1e3 * (
                np.sum((np.abs(V) - 0.95) ** 2) +
                np.sum(Pdg[Pdg < 0]) +
                np.sum((np.abs(V) - 1.0) ** 2)
        )

        fitness = f + penalty

        return fitness if np.isfinite(fitness) else 1e10
