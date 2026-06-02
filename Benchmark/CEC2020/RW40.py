from Benchmark.__Function import Function
import numpy as np
import os


class RW40(Function):
    def __init__(self, dim):

        self.load_network_data()

        n_vars = 76

        # ===== INITIAL SOLUTION =====
        x_start = np.zeros(n_vars)
        x_start[0:37] = 1.0
        x_start[37:74] = 0.0
        x_start[74] = 0.0
        x_start[75] = 1.0

        # ===== DOMAIN =====
        domain = [
            *[(0.8, 1.2) for _ in range(37)],
            *[(-0.5, 0.5) for _ in range(37)],
            (-0.05, 0.05),
            (0.9, 1.1),
        ]

        Function.__init__(
            self,
            dim,
            domain,
            x_start,
            0,
            max_dimension=n_vars,
            name="RW40"
        )

        self.types(['CEC2020RW'])

        # ===== DER =====
        self.der_buses = [33, 34, 35, 36, 37]

        self.P_der_rated = np.array([
            1.0 / 5.102e-3,
            1.0 / 1.502e-3,
            1.0 / 4.506e-3,
            1.0 / 2.253e-3,
            1.0 / 2.253e-3
        ])

        self.Q_der_rated = np.array([
            1.0 / 0.05,
            1.0 / 0.03,
            1.0 / 0.05,
            1.0 / 0.01,
            1.0 / 0.1
        ])

        self.P_load_exp = 1.0
        self.Q_load_exp = 2.0

    # =========================
    # NETWORK DATA
    # =========================
    def load_network_data(self):
        try:
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS2_P.txt"))
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS2_Q.txt"))
            self.L = np.loadtxt(os.path.join(base_path, "FunctionPS14_linedata.txt"))

        except Exception:
            self._create_synthetic_network()

    def _create_synthetic_network(self):
        n_bus = 38
        np.random.seed(42)

        self.P = np.zeros((n_bus, 1))
        self.Q = np.zeros((n_bus, 1))

        self.P[1:33, 0] = np.random.uniform(0.05, 0.15, 32)
        self.Q[1:33, 0] = np.random.uniform(0.02, 0.08, 32)

        lines = []
        for i in range(1, n_bus):
            r = 0.01 + 0.005 * np.random.random()
            x = 0.05 + 0.02 * np.random.random()
            lines.append([i, i + 1, r, x, 0.0001, 1.5])

        self.L = np.array(lines)

    # =========================
    # YBUS
    # =========================
    def ybus(self, L, w):
        n_bus = 38
        Y = np.zeros((n_bus, n_bus), dtype=complex)

        w_nom = 2 * np.pi * 50
        w_actual = w_nom * (1 + w)

        for line in L:
            i = int(line[0]) - 1
            j = int(line[1]) - 1

            R = line[2]
            X = line[3] * (w_actual / w_nom)
            B = line[4]

            Z = R + 1j * X
            if abs(Z) < 1e-12:
                continue

            y = 1.0 / Z

            Y[i, j] -= y
            Y[j, i] -= y
            Y[i, i] += y + 1j * B
            Y[j, j] += y + 1j * B

        return Y

    # =========================
    # EVAL
    # =========================
    def eval(self, variables_values):

        BIG = 1e10
        EPS = 1e-10

        x = np.array(variables_values, dtype=float).flatten()

        # ===== VARIABLES =====
        V_real = x[0:37]
        V_imag = x[37:74]
        w = x[74]
        V_slack = x[75]

        # ===== BUILD V =====
        V = np.zeros(38, dtype=complex)
        V[0] = V_slack + 0j
        V[1:] = V_real + 1j * V_imag

        if not np.all(np.isfinite(V)):
            return BIG

        # ===== YBUS =====
        Y = self.ybus(self.L, w)

        # ===== CURRENT =====
        I = Y @ V

        if not np.all(np.isfinite(I)):
            return BIG

        # ===== POWER (FIX ✅) =====
        S = V * np.conj(I)
        P_inj = np.real(S)
        Q_inj = np.imag(S)

        # ===== LOAD =====
        V_mag = np.abs(V)

        P_load = self.P[:, 0] * (V_mag ** self.P_load_exp)
        Q_load = self.Q[:, 0] * (V_mag ** self.Q_load_exp)

        # ===== DER =====
        P_der = np.zeros(38)
        Q_der = np.zeros(38)

        for i, bus in enumerate(self.der_buses):
            P_der[bus] = self.P_der_rated[i] * (1 - w)
            Q_der[bus] = self.Q_der_rated[i] * (1 - V_mag[bus])

        # ===== MISMATCH =====
        delta_P = P_inj - (P_der - P_load)
        delta_Q = Q_inj - (Q_der - Q_load)

        if not np.all(np.isfinite(delta_P)):
            return BIG

        # ===== OBJECTIVE =====
        objective = np.sum(delta_P**2) + np.sum(delta_Q**2)

        # ===== PENALTIES =====
        # Voltage
        V_pen = np.sum(np.maximum(0, 0.95 - V_mag)**2) + \
                np.sum(np.maximum(0, V_mag - 1.05)**2)

        # Current
        I_pen = np.sum(np.maximum(0, np.abs(I) - 1.5)**2)

        # Frequency
        w_pen = max(0, abs(w) - 0.05)**2

        penalty = 1e3 * V_pen + 1e3 * I_pen + 1e2 * w_pen

        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else BIG
