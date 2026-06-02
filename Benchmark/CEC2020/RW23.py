from Benchmark.__Function import Function
import numpy as np


class RW23(Function):
    def __init__(self, dim):
        x_start = np.zeros(5)
        f_x_start = 0
        domain = [
            (71.0, 125.0),    # x1 : d1 (diameter of pulley 1, mm)
            (71.0, 125.0),    # x2 : d2 (diameter of pulley 2, mm)
            (71.0, 125.0),    # x3 : d3 (diameter of pulley 3, mm)
            (71.0, 125.0),    # x4 : d4 (diameter of pulley 4, mm)
            (12.0, 20.0),     # x5 : w (width of belt, mm)
        ]
        name = "RW23"
        max_dimension = 5

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

        # ===== VARIABLES (convert from mm to m) =====
        d1 = x[0] * 1e-3  # diameter of pulley 1 (m)
        d2 = x[1] * 1e-3  # diameter of pulley 2 (m)
        d3 = x[2] * 1e-3  # diameter of pulley 3 (m)
        d4 = x[3] * 1e-3  # diameter of pulley 4 (m)
        w = x[4] * 1e-3   # width of belt (m)

        # ===== PHYSICAL PARAMETERS =====
        N = 350.0          # motor speed (rpm)
        N1 = 750.0         # target speed at stage 1 (rpm)
        N2 = 450.0         # target speed at stage 2 (rpm)
        N3 = 250.0         # target speed at stage 3 (rpm)
        N4 = 150.0         # target speed at stage 4 (rpm)
        rho = 7200.0       # density of belt material (kg/m³)
        a = 3.0            # center distance (m)
        mu = 0.35          # coefficient of friction
        s = 1.75e6         # tensile strength (Pa)
        t = 8e-3           # belt thickness (m)

        # ===== CENTER DISTANCES =====
        # Center distance for each stage
        C1 = (np.pi * d1 / 2.0 * (1 + N1 / N)
              + ((N1 / N - 1)**2) * d1**2 / (4 * a) + 2 * a)
        C2 = (np.pi * d2 / 2.0 * (1 + N2 / N)
              + ((N2 / N - 1)**2) * d2**2 / (4 * a) + 2 * a)
        C3 = (np.pi * d3 / 2.0 * (1 + N3 / N)
              + ((N3 / N - 1)**2) * d3**2 / (4 * a) + 2 * a)
        C4 = (np.pi * d4 / 2.0 * (1 + N4 / N)
              + ((N4 / N - 1)**2) * d4**2 / (4 * a) + 2 * a)

        # ===== FRICTION RATIO =====
        # Compute safe asin arguments
        def safe_asin(arg):
            return np.arcsin(np.clip(arg, -1.0, 1.0))

        angle1 = (N1 / N - 1) * d1 / (2 * a)
        angle2 = (N2 / N - 1) * d2 / (2 * a)
        angle3 = (N3 / N - 1) * d3 / (2 * a)
        angle4 = (N4 / N - 1) * d4 / (2 * a)

        # Capstan/Eytelwein equation: R = exp(mu * theta)
        # where theta = pi - 2*asin(...)
        R1 = np.exp(mu * (np.pi - 2 * safe_asin(angle1)))
        R2 = np.exp(mu * (np.pi - 2 * safe_asin(angle2)))
        R3 = np.exp(mu * (np.pi - 2 * safe_asin(angle3)))
        R4 = np.exp(mu * (np.pi - 2 * safe_asin(angle4)))

        # ===== TRANSMITTED POWER =====
        # Power transmitted by each stage (W)
        # Formula: P = sigma * t * w * (1 - exp(-mu*theta)) * pi * d * N / 60
        Pmax = 0.75 * 745.6998  # 0.75 hp in watts ≈ 559.2749 W

        exp_term1 = np.exp(-mu * (np.pi - 2 * safe_asin(angle1)))
        exp_term2 = np.exp(-mu * (np.pi - 2 * safe_asin(angle2)))
        exp_term3 = np.exp(-mu * (np.pi - 2 * safe_asin(angle3)))
        exp_term4 = np.exp(-mu * (np.pi - 2 * safe_asin(angle4)))

        P1 = s * t * w * (1 - exp_term1) * np.pi * d1 * N1 / 60.0
        P2 = s * t * w * (1 - exp_term2) * np.pi * d2 * N2 / 60.0
        P3 = s * t * w * (1 - exp_term3) * np.pi * d3 * N3 / 60.0
        P4 = s * t * w * (1 - exp_term4) * np.pi * d4 * N4 / 60.0

        # ===== OBJECTIVE FUNCTION =====
        # Minimize mass of belts
        f = (rho * w * np.pi / 4.0
             * (d1**2 * (1 + (N1 / N)**2)
                + d2**2 * (1 + (N2 / N)**2)
                + d3**2 * (1 + (N3 / N)**2)
                + d4**2 * (1 + (N4 / N)**2)))

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(8)

        # Friction ratio constraints (R >= 2 for safety)
        g[0] = -R1 + 2.0
        g[1] = -R2 + 2.0
        g[2] = -R3 + 2.0
        g[3] = -R4 + 2.0

        # Power transmission constraints (P >= Pmax for each stage)
        g[4] = -P1 + Pmax
        g[5] = -P2 + Pmax
        g[6] = -P3 + Pmax
        g[7] = -P4 + Pmax

        # ===== EQUALITY CONSTRAINTS (h = 0) =====
        # Belt lengths must be equal (C1 = C2 = C3 = C4)
        h = np.zeros(3)
        h[0] = C1 - C2
        h[1] = C1 - C3
        h[2] = C1 - C4

        # ===== PENALTY =====
        tol = 1e-4
        penalty_h = np.sum(np.maximum(0, np.abs(h) - tol) ** 2)
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
