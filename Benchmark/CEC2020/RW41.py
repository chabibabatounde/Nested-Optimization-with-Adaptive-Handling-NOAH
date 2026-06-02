from Benchmark.Functions.__Function import Function
import numpy as np
import os


class RW41(Function):
    def __init__(self, dim):

        self.load_network_data()

        n_vars = 74

        # ===== INITIAL SOLUTION =====
        x_start = np.zeros(n_vars)
        x_start[0:37] = 1.0
        x_start[37:74] = 0.0

        # ===== DOMAIN =====
        domain = [
            *[(0.9, 1.1) for _ in range(37)],
            *[(-0.3, 0.3) for _ in range(37)],
        ]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0,
            max_dimension=n_vars,
            name="RW41"
        )

        self.types(['CEC2020RW'])

        # ===== DER =====
        self.der_buses = np.array([33, 34, 35, 36, 37])

        self.P_der_fixed = np.array([0.2] * 5)
        self.Q_der_fixed = np.array([0.18] * 5)

    # =========================
    # LOAD DATA
    # =========================
    def load_network_data(self):
        try:
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            G = np.loadtxt(os.path.join(base_path, "FunctionPS2_G.txt"))
            B = np.loadtxt(os.path.join(base_path, "FunctionPS2_B.txt"))

            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS2_P.txt"))
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS2_Q.txt"))

            # ✅ Ybus correcte
            self.Y = G + 1j * B

        except Exception:
            self._create_synthetic_network()

    # =========================
    # SYNTHETIC NETWORK
    # =========================
    def _create_synthetic_network(self):
        n_bus = 38
        np.random.seed(42)

        # ✅ taille corrigée (7 colonnes)
        self.P = np.zeros((n_bus, 7))
        self.Q = np.zeros((n_bus, 7))

        for i in range(1, 33):
            self.P[i, 0] = np.random.uniform(0.05, 0.15)
            self.P[i, 6] = 1.0

            self.Q[i, 0] = np.random.uniform(0.02, 0.08)
            self.Q[i, 6] = 2.0

        # ===== Ybus synthétique =====
        self.Y = np.zeros((n_bus, n_bus), dtype=complex)

        for i in range(n_bus - 1):
            R = 0.01 + 0.003 * np.sin(i)
            X = 0.04 + 0.015 * np.cos(i)

            Z = R + 1j * X
            y = 1.0 / Z

            self.Y[i, i + 1] -= y
            self.Y[i + 1, i] -= y

            self.Y[i, i] += y
            self.Y[i + 1, i + 1] += y

    # =========================
    # EVALUATION
    # =========================
    def eval(self, variables_values):

        BIG = 1e10

        x = np.array(variables_values, dtype=float).flatten()

        # ✅ sécurité
        if self.Y is None or not isinstance(self.Y, np.ndarray):
            return BIG

        # ===== VARIABLES =====
        V_real = x[0:37]
        V_imag = x[37:74]

        # ===== VOLTAGE VECTOR =====
        V = np.zeros(38, dtype=complex)
        V[0] = 1.0 + 0j
        V[1:] = V_real + 1j * V_imag

        if not np.all(np.isfinite(V)):
            return BIG

        # ===== CURRENT =====
        I = self.Y @ V

        if not np.all(np.isfinite(I)):
            return BIG

        # ✅ ✅ FIX POWER FLOW (IMPORTANT)
        S = V * np.conj(I)
        P_injected = np.real(S)
        Q_injected = np.imag(S)

        # ===== LOAD =====
        V_mag = np.abs(V)

        P_load = self.P[:, 0] * (V_mag ** self.P[:, 6])
        Q_load = self.Q[:, 0] * (V_mag ** self.Q[:, 6])

        # ===== DER =====
        P_der = np.zeros(38)
        Q_der = np.zeros(38)

        for i, bus in enumerate(self.der_buses):
            P_der[bus] = self.P_der_fixed[i]
            Q_der[bus] = self.Q_der_fixed[i]

        # ===== MISMATCH =====
        delta_P = P_injected - (P_der - P_load)
        delta_Q = Q_injected - (Q_der - Q_load)

        if not np.all(np.isfinite(delta_P)):
            return BIG

        # ===== OBJECTIVE =====
        objective = np.sum(delta_P[1:]**2) + np.sum(delta_Q[1:]**2)

        # ===== PENALTIES =====
        V_pen = np.sum(np.maximum(0, 0.9 - V_mag)**2) + \
                np.sum(np.maximum(0, V_mag - 1.1)**2)

        I_pen = np.sum(np.maximum(0, np.abs(I) - 2.0)**2)

        # Slack power
        P_grid = P_injected[0]
        Q_grid = Q_injected[0]

        grid_pen = (
            max(0, abs(P_grid) - 2.0)**2 +
            max(0, abs(Q_grid) - 1.5)**2
        )

        penalty = 1e3 * V_pen + 1e3 * I_pen + 5e2 * grid_pen

        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else BIG
