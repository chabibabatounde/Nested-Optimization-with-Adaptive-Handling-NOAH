from Benchmark.__Function import Function
import numpy as np


class RW25(Function):
    def __init__(self, dim):
        x_start = np.zeros(4)
        f_x_start = 0
        domain = [
            (0.4, 0.5),   # x1 : R
            (0.25, 0.4),  # x2 : Ro
            (1.0, 10.0),  # x3 : mu
            (10.0, 60.0), # x4 : Q
        ]
        name = "RW25"
        max_dimension = 4

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
        x = np.array(variables_values).reshape(1, -1)[0]

        # ===== VARIABLES =====
        R, Ro, mu, Q = x

        # ===== CONSTANTS =====
        gamma = 0.0307
        C = 0.5
        n = -3.55
        C1 = 10.04
        Ws = 101000.0
        Pmax = 1000.0
        delTmax = 50.0
        hmin = 0.001
        gg = 386.4
        N = 750.0

        BIG = 1e10
        EPS = 1e-10

        # ===== PRESSURE =====
        arg_log = 8.122e6 * mu + 0.8

        if arg_log <= 0:
            return BIG

        try:
            inner_log = np.log10(arg_log)
            if inner_log <= 0:
                return BIG

            P = (np.log10(inner_log) - C1) / n

        except:
            return BIG

        # Clamp P to avoid overflow
        if not np.isfinite(P):
            return BIG

        P = np.clip(P, -50, 50)

        # ===== TEMPERATURE =====
        try:
            if P > 50:
                delT = BIG
            else:
                delT = 2.0 * (10.0 ** P - 560.0)
        except:
            delT = BIG

        # ===== ENERGY =====
        Ef = 9336.0 * Q * gamma * C * delT

        if not np.isfinite(Ef) or Ef <= 0:
            return BIG

        # ===== FILM THICKNESS =====
        omega = (2.0 * np.pi * N / 60.0) ** 2

        geom = (R**4 - Ro**4) / 4.0

        h = (omega * 2.0 * np.pi * mu / (Ef + EPS) * geom - 1e-5)

        if not np.isfinite(h):
            return BIG

        # ===== PRESSURE Po =====
        if R <= Ro:
            return BIG

        log_ratio = np.log((R + EPS) / (Ro + EPS))

        if abs(h) < EPS or not np.isfinite(log_ratio):
            return BIG

        Po = (6.0 * mu * Q / (np.pi * h**3 + EPS)) * log_ratio

        if not np.isfinite(Po):
            return BIG

        # ===== LOAD =====
        denom = log_ratio - 1e-5
        if abs(denom) < EPS:
            return BIG

        W = (np.pi * Po / 2.0) * (R**2 - Ro**2) / denom

        if not np.isfinite(W):
            return BIG

        # ===== OBJECTIVE =====
        f = (Q * Po / 0.7 + Ef) / 12.0

        if not np.isfinite(f):
            return BIG

        # ===== CONSTRAINTS =====
        g = np.zeros(7)

        g[0] = Ws - W
        g[1] = Po - Pmax
        g[2] = delT - delTmax
        g[3] = hmin - h
        g[4] = Ro - R

        # Squeeze number
        squeeze = (gamma / (gg * (Po + EPS))) * (
            Q / (2.0 * np.pi * (R + EPS) * (h + EPS))
        )
        g[5] = squeeze - 0.001

        # Pressure per area
        area = np.pi * (R**2 - Ro**2)
        if area <= EPS:
            return BIG

        pressure_per_area = W / area
        g[6] = pressure_per_area - 5000.0

        # ===== PENALTY =====
        penalty = np.sum(np.maximum(0, g) ** 2)

        fitness = f + 1e5 * penalty

        return fitness
