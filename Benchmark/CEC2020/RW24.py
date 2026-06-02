from Benchmark.Functions.__Function import Function
import numpy as np


class RW24(Function):
    def __init__(self, dim):
        x_start = np.zeros(7)
        f_x_start = 0
        domain = [
            (40.0, 60.0),  # x1 : a (link length a, mm)
            (40.0, 60.0),  # x2 : b (link length b, mm)
            (0.0, 20.0),  # x3 : c (link length c, mm)
            (0.0, 20.0),  # x4 : e (offset e, mm)
            (0.0, 20.0),  # x5 : ff (offset ff, mm)
            (40.0, 100.0),  # x6 : l (link length l, mm)
            (-20.0, 20.0),  # x7 : delta (angle offset delta, rad)
        ]
        name = "RW24"
        max_dimension = 7

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

    @staticmethod
    def OBJ11(x, config):
        """
        Compute gripper opening width for configuration (config = 1 or 2).
        This is a helper function for the robot gripper objective.

        Parameters:
        -----------
        x : array of 7 variables [a, b, c, e, ff, l, delta]
        config : int (1 or 2) - gripper configuration

        Returns:
        --------
        opening_width : float (complex if infeasible kinematics)
        """
        a = x[0]
        b = x[1]
        c = x[2]
        e = x[3]
        ff = x[4]
        l = x[5]
        delta = x[6]

        Zmax = 99.9999

        # Compute intermediate angles for configuration
        # These represent the linkage kinematics at min/max positions
        if config == 1:
            # Configuration 1: at maximum extension (l - Zmax)
            denominator = 2 * a * np.sqrt((l - Zmax) ** 2 + e ** 2)
            if np.abs(denominator) < 1e-10:
                return 1e10 + 1j * 1e10

            cos_arg1 = (a ** 2 + (l - Zmax) ** 2 + e ** 2 - b ** 2) / denominator
            cos_arg1 = np.clip(np.real(cos_arg1), -1.0, 1.0)
            alpha = np.arccos(cos_arg1) + np.arctan2(e, l - Zmax)

            denominator2 = 2 * b * np.sqrt((l - Zmax) ** 2 + e ** 2)
            if np.abs(denominator2) < 1e-10:
                return 1e10 + 1j * 1e10

            cos_arg2 = (b ** 2 + (l - Zmax) ** 2 + e ** 2 - a ** 2) / denominator2
            cos_arg2 = np.clip(np.real(cos_arg2), -1.0, 1.0)
            beta = np.arccos(cos_arg2) - np.arctan2(e, l - Zmax)

        else:  # config == 2
            # Configuration 2: at minimum extension (l)
            denominator = 2 * a * np.sqrt(l ** 2 + e ** 2)
            if np.abs(denominator) < 1e-10:
                return 1e10 + 1j * 1e10

            cos_arg1 = (a ** 2 + l ** 2 + e ** 2 - b ** 2) / denominator
            cos_arg1 = np.clip(np.real(cos_arg1), -1.0, 1.0)
            alpha = np.arccos(cos_arg1) + np.arctan2(e, l)

            denominator2 = 2 * b * np.sqrt(l ** 2 + e ** 2)
            if np.abs(denominator2) < 1e-10:
                return 1e10 + 1j * 1e10

            cos_arg2 = (b ** 2 + l ** 2 + e ** 2 - a ** 2) / denominator2
            cos_arg2 = np.clip(np.real(cos_arg2), -1.0, 1.0)
            beta = np.arccos(cos_arg2) - np.arctan2(e, l)

        # Gripper opening width: Y = 2*(e + ff + c*sin(beta + delta))
        opening = 2.0 * (e + ff + c * np.sin(beta + delta))

        return opening

    def eval(self, variables_values):
        x = np.array(variables_values).reshape(1, -1)[0]

        # ===== VARIABLES =====
        a = x[0]  # link length a (mm)
        b = x[1]  # link length b (mm)
        c = x[2]  # link length c (mm)
        e = x[3]  # offset e (mm)
        ff = x[4]  # offset ff (mm)
        l = x[5]  # link length l (mm)
        delta = x[6]  # angle offset delta (rad)

        # ===== PHYSICAL PARAMETERS =====
        Ymin = 50.0  # minimum opening width (mm)
        Ymax = 100.0  # maximum opening width (mm)
        YG = 150.0  # gripper width constraint (mm)
        Zmax = 99.9999  # maximum extension limit (mm)
        P = 100.0  # pressure/force parameter

        # ===== INTERMEDIATE ANGLES (Kinematics) =====
        # Configuration at minimum extension (l)
        denom_a0 = 2 * a * np.sqrt(l ** 2 + e ** 2)
        denom_b0 = 2 * b * np.sqrt(l ** 2 + e ** 2)

        if np.abs(denom_a0) < 1e-10 or np.abs(denom_b0) < 1e-10:
            alpha_0 = 1e10
            beta_0 = 1e10
        else:
            cos_arg_a0 = (a ** 2 + l ** 2 + e ** 2 - b ** 2) / denom_a0
            cos_arg_a0 = np.clip(cos_arg_a0, -1.0, 1.0)
            alpha_0 = np.arccos(cos_arg_a0) + np.arctan2(e, l)

            cos_arg_b0 = (b ** 2 + l ** 2 + e ** 2 - a ** 2) / denom_b0
            cos_arg_b0 = np.clip(cos_arg_b0, -1.0, 1.0)
            beta_0 = np.arccos(cos_arg_b0) - np.arctan2(e, l)

        # Configuration at maximum extension (l - Zmax)
        denom_am = 2 * a * np.sqrt((l - Zmax) ** 2 + e ** 2)
        denom_bm = 2 * b * np.sqrt((l - Zmax) ** 2 + e ** 2)

        if np.abs(denom_am) < 1e-10 or np.abs(denom_bm) < 1e-10:
            alpha_m = 1e10
            beta_m = 1e10
        else:
            cos_arg_am = (a ** 2 + (l - Zmax) ** 2 + e ** 2 - b ** 2) / denom_am
            cos_arg_am = np.clip(cos_arg_am, -1.0, 1.0)
            alpha_m = np.arccos(cos_arg_am) + np.arctan2(e, l - Zmax)

            cos_arg_bm = (b ** 2 + (l - Zmax) ** 2 + e ** 2 - a ** 2) / denom_bm
            cos_arg_bm = np.clip(cos_arg_bm, -1.0, 1.0)
            beta_m = np.arccos(cos_arg_bm) - np.arctan2(e, l - Zmax)

        # ===== OBJECTIVE FUNCTION =====
        # Minimize: -(OBJ11(config=2) + OBJ11(config=1))
        # = Maximize gripper opening at both configurations
        obj_config1 = self.OBJ11(x, 1)
        obj_config2 = self.OBJ11(x, 2)

        # Handle complex numbers (infeasible kinematics)
        if np.iscomplex(obj_config1) or np.imag(obj_config1) != 0:
            obj_config1 = 1e4
        else:
            obj_config1 = np.real(obj_config1)

        if np.iscomplex(obj_config2) or np.imag(obj_config2) != 0:
            obj_config2 = 1e4
        else:
            obj_config2 = np.real(obj_config2)

        f = -(obj_config1 + obj_config2)

        # ===== GRIPPER OPENING WIDTHS =====
        Yxmin = 2.0 * (e + ff + c * np.sin(beta_m + delta))
        Yxmax = 2.0 * (e + ff + c * np.sin(beta_0 + delta))

        # ===== INEQUALITY CONSTRAINTS (g <= 0) =====
        g = np.zeros(7)

        # Opening width constraints
        g[0] = Yxmin - Ymin  # Ymin ≤ Yxmin
        g[1] = -Yxmin  # Yxmin ≥ 0
        g[2] = Ymax - Yxmax  # Yxmax ≤ Ymax
        g[3] = Yxmax - YG  # Yxmax ≤ YG

        # Linkage geometric constraints
        g[4] = l ** 2 + e ** 2 - (a + b) ** 2  # l² + e² ≤ (a + b)²
        g[5] = b ** 2 - (a - e) ** 2 - (l - Zmax) ** 2  # b² ≤ (a-e)² + (l-Zmax)²
        g[6] = Zmax - l  # l ≤ Zmax

        # ===== EQUALITY CONSTRAINTS =====
        h = np.array([0.0])  # No equality constraints

        # ===== Handle complex/infeasible solutions =====
        # If any constraint is complex (infeasible), penalize heavily
        if np.any(np.iscomplex(g)) or np.any(np.imag(g) != 0):
            g = np.real(g)
            f = 1e4

        # ===== PENALTY =====
        tol = 1e-4
        penalty_h = 0.0
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
