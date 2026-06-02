from Benchmark.Functions.__Function import Function
import numpy as np


class RW26(Function):
    def __init__(self, dim):
        x_start = np.zeros(22)
        f_x_start = 0
        domain = [
            (12, 60),          # x1  : Np1 (pinion 1 teeth count)
            (12, 60),          # x2  : Ng1 (gear 1 teeth count)
            (12, 60),          # x3  : Np2 (pinion 2 teeth count)
            (12, 60),          # x4  : Ng2 (gear 2 teeth count)
            (12, 60),          # x5  : Np3 (pinion 3 teeth count)
            (12, 60),          # x6  : Ng3 (gear 3 teeth count)
            (12, 60),          # x7  : Np4 (pinion 4 teeth count)
            (12, 60),          # x8  : Ng4 (gear 4 teeth count)
            (1, 4),            # x9  : index for b1 (pitch in mm)
            (1, 4),            # x10 : index for b2
            (1, 4),            # x11 : index for b3
            (1, 4),            # x12 : index for b4
            (1, 9),            # x13 : index for xp1 (position)
            (1, 9),            # x14 : index for xg1
            (1, 9),            # x15 : index for xg2
            (1, 9),            # x16 : index for xg3
            (1, 9),            # x17 : index for xg4
            (1, 9),            # x18 : index for yp1
            (1, 9),            # x19 : index for yg1
            (1, 9),            # x20 : index for yg2
            (1, 9),            # x21 : index for yg3
            (1, 9),            # x22 : index for yg4
        ]
        name = "RW26"
        max_dimension = 22

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

        # ===== ROUND INTEGERS =====
        x = np.round(x)

        # ===== EXTRACT GEAR TEETH COUNTS =====
        Np1 = x[0]
        Ng1 = x[1]
        Np2 = x[2]
        Ng2 = x[3]
        Np3 = x[4]
        Ng3 = x[5]
        Np4 = x[6]
        Ng4 = x[7]

        # ===== LOOKUP TABLES =====
        # Pitch values (mm)
        Pvalue = np.array([3.175, 5.715, 8.255, 12.7])
        b1 = Pvalue[int(x[8]) - 1] if 1 <= x[8] <= 4 else Pvalue[0]
        b2 = Pvalue[int(x[9]) - 1] if 1 <= x[9] <= 4 else Pvalue[0]
        b3 = Pvalue[int(x[10]) - 1] if 1 <= x[10] <= 4 else Pvalue[0]
        b4 = Pvalue[int(x[11]) - 1] if 1 <= x[11] <= 4 else Pvalue[0]

        # XY position values (mm)
        XYvalue = np.array([12.7, 25.4, 38.1, 50.8, 63.5, 76.2, 88.9, 101.6, 114.3])
        xp1 = XYvalue[int(x[12]) - 1] if 1 <= x[12] <= 9 else XYvalue[0]
        xg1 = XYvalue[int(x[13]) - 1] if 1 <= x[13] <= 9 else XYvalue[0]
        xg2 = XYvalue[int(x[14]) - 1] if 1 <= x[14] <= 9 else XYvalue[0]
        xg3 = XYvalue[int(x[15]) - 1] if 1 <= x[15] <= 9 else XYvalue[0]
        xg4 = XYvalue[int(x[16]) - 1] if 1 <= x[16] <= 9 else XYvalue[0]
        yp1 = XYvalue[int(x[17]) - 1] if 1 <= x[17] <= 9 else XYvalue[0]
        yg1 = XYvalue[int(x[18]) - 1] if 1 <= x[18] <= 9 else XYvalue[0]
        yg2 = XYvalue[int(x[19]) - 1] if 1 <= x[19] <= 9 else XYvalue[0]
        yg3 = XYvalue[int(x[20]) - 1] if 1 <= x[20] <= 9 else XYvalue[0]
        yg4 = XYvalue[int(x[21]) - 1] if 1 <= x[21] <= 9 else XYvalue[0]

        # ===== CALCULATE CENTER DISTANCES =====
        c1 = np.sqrt((xg1 - xp1)**2 + (yg1 - yp1)**2)
        c2 = np.sqrt((xg2 - xp1)**2 + (yg2 - yp1)**2)
        c3 = np.sqrt((xg3 - xp1)**2 + (yg3 - yp1)**2)
        c4 = np.sqrt((xg4 - xp1)**2 + (yg4 - yp1)**2)

        # ===== PHYSICAL PARAMETERS =====
        CRmin = 1.4
        dmin = 25.4        # minimum diameter (mm)
        phi = 20.0 * np.pi / 180.0  # pressure angle (rad)
        W = 55.9           # load (N)
        JR = 0.2           # geometry factor
        Km = 1.6           # load distribution factor
        Ko = 1.5           # overload factor
        Lmax = 127.0       # maximum layout dimension (mm)
        sigma_H = 3290.0   # contact stress limit (MPa)
        sigma_N = 2090.0   # bending stress limit (MPa)
        w1 = 5000.0        # input speed (rpm)
        wmin = 245.0       # minimum output speed (rpm)
        wmax = 255.0       # maximum output speed (rpm)
        Cp = 464.0         # pressure coefficient

        # ===== OBJECTIVE FUNCTION =====
        # Volume/weight of gearbox
        f = (np.pi / 1000.0 *
             (b1 * c1**2 * (Np1**2 + Ng1**2) / (Np1 + Ng1)**2 +
              b2 * c2**2 * (Np2**2 + Ng2**2) / (Np2 + Ng2)**2 +
              b3 * c3**2 * (Np3**2 + Ng3**2) / (Np3 + Ng3)**2 +
              b4 * c4**2 * (Np4**2 + Ng4**2) / (Np4 + Ng4)**2))

        # ===== INEQUALITY CONSTRAINTS =====
        g = np.zeros(86)

        # Bending stress constraints (gear 1-4, stages 1-4)
        g[0] = ((366000.0 / (np.pi * w1) + 2.0 * c1 * Np1 / (Np1 + Ng1)) *
                ((Np1 + Ng1)**2 / (4.0 * b1 * c1**2 * Np1)) -
                sigma_N * JR / (0.0167 * W * Ko * Km))

        g[1] = ((366000.0 * Ng1 / (np.pi * w1 * Np1) + 2.0 * c2 * Np2 / (Np2 + Ng2)) *
                ((Np2 + Ng2)**2 / (4.0 * b2 * c2**2 * Np2)) -
                sigma_N * JR / (0.0167 * W * Ko * Km))

        g[2] = ((366000.0 * Ng1 * Ng2 / (np.pi * w1 * Np1 * Np2) + 2.0 * c3 * Np3 / (Np3 + Ng3)) *
                ((Np3 + Ng3)**2 / (4.0 * b3 * c3**2 * Np3)) -
                sigma_N * JR / (0.0167 * W * Ko * Km))

        g[3] = ((366000.0 * Ng1 * Ng2 * Ng3 / (np.pi * w1 * Np1 * Np2 * Np3) + 2.0 * c4 * Np4 / (Np4 + Ng4)) *
                ((Np4 + Ng4)**2 / (4.0 * b4 * c4**2 * Np4)) -
                sigma_N * JR / (0.0167 * W * Ko * Km))

        # Contact stress constraints (4 stages)
        g[4] = ((366000.0 / (np.pi * w1) + 2.0 * c1 * Np1 / (Np1 + Ng1)) *
                ((Np1 + Ng1)**3 / (4.0 * b1 * c1**2 * Ng1 * Np1**2)) -
                (sigma_H / Cp)**2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km))

        g[5] = ((366000.0 * Ng1 / (np.pi * w1 * Np1) + 2.0 * c2 * Np2 / (Np2 + Ng2)) *
                ((Np2 + Ng2)**3 / (4.0 * b2 * c2**2 * Ng2 * Np2**2)) -
                (sigma_H / Cp)**2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km))

        g[6] = ((366000.0 * Ng1 * Ng2 / (np.pi * w1 * Np1 * Np2) + 2.0 * c3 * Np3 / (Np3 + Ng3)) *
                ((Np3 + Ng3)**3 / (4.0 * b3 * c3**2 * Ng3 * Np3**2)) -
                (sigma_H / Cp)**2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km))

        g[7] = ((366000.0 * Ng1 * Ng2 * Ng3 / (np.pi * w1 * Np1 * Np2 * Np3) + 2.0 * c4 * Np4 / (Np4 + Ng4)) *
                ((Np4 + Ng4)**3 / (4.0 * b4 * c4**2 * Ng4 * Np4**2)) -
                (sigma_H / Cp)**2 * (np.sin(phi) * np.cos(phi)) / (0.0334 * W * Ko * Km))

        # Interference and undercut constraints (4 stages)
        sqrt_term1 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Np1 + 1e-10) + (1.0 / (Np1 + 1e-10))**2)
        sqrt_term2 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Ng1 + 1e-10) + (1.0 / (Ng1 + 1e-10))**2)
        g[8] = (CRmin * np.pi * np.cos(phi) - Np1 * sqrt_term1 - Ng1 * sqrt_term2 +
                np.sin(phi) * (Np1 + Ng1) / 2.0)

        sqrt_term1 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Np2 + 1e-10) + (1.0 / (Np2 + 1e-10))**2)
        sqrt_term2 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Ng2 + 1e-10) + (1.0 / (Ng2 + 1e-10))**2)
        g[9] = (CRmin * np.pi * np.cos(phi) - Np2 * sqrt_term1 - Ng2 * sqrt_term2 +
                np.sin(phi) * (Np2 + Ng2) / 2.0)

        sqrt_term1 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Np3 + 1e-10) + (1.0 / (Np3 + 1e-10))**2)
        sqrt_term2 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Ng3 + 1e-10) + (1.0 / (Ng3 + 1e-10))**2)
        g[10] = (CRmin * np.pi * np.cos(phi) - Np3 * sqrt_term1 - Ng3 * sqrt_term2 +
                 np.sin(phi) * (Np3 + Ng3) / 2.0)

        sqrt_term1 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Np4 + 1e-10) + (1.0 / (Np4 + 1e-10))**2)
        sqrt_term2 = np.sqrt(np.sin(phi)**2 / 4.0 + 1.0 / (Ng4 + 1e-10) + (1.0 / (Ng4 + 1e-10))**2)
        g[11] = (CRmin * np.pi * np.cos(phi) - Np4 * sqrt_term1 - Ng4 * sqrt_term2 +
                 np.sin(phi) * (Np4 + Ng4) / 2.0)

        # Minimum diameter constraints (pinion and gear at each stage)
        g[12] = dmin - 2.0 * c1 * Np1 / (Np1 + Ng1)
        g[13] = dmin - 2.0 * c2 * Np2 / (Np2 + Ng2)
        g[14] = dmin - 2.0 * c3 * Np3 / (Np3 + Ng3)
        g[15] = dmin - 2.0 * c4 * Np4 / (Np4 + Ng4)

        g[16] = dmin - 2.0 * c1 * Ng1 / (Np1 + Ng1)
        g[17] = dmin - 2.0 * c2 * Ng2 / (Np2 + Ng2)
        g[18] = dmin - 2.0 * c3 * Ng3 / (Np3 + Ng3)
        g[19] = dmin - 2.0 * c4 * Ng4 / (Np4 + Ng4)

        # Layout constraints - X position boundaries
        g[20] = xp1 + ((Np1 + 2.0) * c1 / (Np1 + Ng1)) - Lmax
        g[21] = xg2 + ((Np2 + 2.0) * c2 / (Np2 + Ng2)) - Lmax
        g[22] = xg3 + ((Np3 + 2.0) * c3 / (Np3 + Ng3)) - Lmax
        g[23] = xg4 + ((Np4 + 2.0) * c4 / (Np4 + Ng4)) - Lmax

        g[24] = -xp1 + ((Np1 + 2.0) * c1 / (Np1 + Ng1))
        g[25] = -xg2 + ((Np2 + 2.0) * c2 / (Np2 + Ng2))
        g[26] = -xg3 + ((Np3 + 2.0) * c3 / (Np3 + Ng3))
        g[27] = -xg4 + ((Np4 + 2.0) * c4 / (Np4 + Ng4))

        # Layout constraints - Y position boundaries
        g[28] = yp1 + ((Np1 + 2.0) * c1 / (Np1 + Ng1)) - Lmax
        g[29] = yg2 + ((Np2 + 2.0) * c2 / (Np2 + Ng2)) - Lmax
        g[30] = yg3 + ((Np3 + 2.0) * c3 / (Np3 + Ng3)) - Lmax
        g[31] = yg4 + ((Np4 + 2.0) * c4 / (Np4 + Ng4)) - Lmax

        g[32] = -yp1 + ((Np1 + 2.0) * c1 / (Np1 + Ng1))
        g[33] = -yg2 + ((Np2 + 2.0) * c2 / (Np2 + Ng2))
        g[34] = -yg3 + ((Np3 + 2.0) * c3 / (Np3 + Ng3))
        g[35] = -yg4 + ((Np4 + 2.0) * c4 / (Np4 + Ng4))

        # Layout constraints - Gear positions
        g[36] = xg1 + ((Ng1 + 2.0) * c1 / (Np1 + Ng1)) - Lmax
        g[37] = xg2 + ((Ng2 + 2.0) * c2 / (Np2 + Ng2)) - Lmax
        g[38] = xg3 + ((Ng3 + 2.0) * c3 / (Np3 + Ng3)) - Lmax
        g[39] = xg4 + ((Ng4 + 2.0) * c4 / (Np4 + Ng4)) - Lmax

        g[40] = -xg1 + ((Ng1 + 2.0) * c1 / (Np1 + Ng1))
        g[41] = -xg2 + ((Ng2 + 2.0) * c2 / (Np2 + Ng2))
        g[42] = -xg3 + ((Ng3 + 2.0) * c3 / (Np3 + Ng3))
        g[43] = -xg4 + ((Ng4 + 2.0) * c4 / (Np4 + Ng4))

        g[44] = yg1 + ((Ng1 + 2.0) * c1 / (Np1 + Ng1)) - Lmax
        g[45] = yg2 + ((Ng2 + 2.0) * c2 / (Np2 + Ng2)) - Lmax
        g[46] = yg3 + ((Ng3 + 2.0) * c3 / (Np3 + Ng3)) - Lmax
        g[47] = yg4 + ((Ng4 + 2.0) * c4 / (Np4 + Ng4)) - Lmax

        g[48] = -yg1 + ((Ng1 + 2.0) * c1 / (Np1 + Ng1))
        g[49] = -yg2 + ((Ng2 + 2.0) * c2 / (Np2 + Ng2))
        g[50] = -yg3 + ((Ng3 + 2.0) * c3 / (Np3 + Ng3))
        g[51] = -yg4 + ((Ng4 + 2.0) * c4 / (Np4 + Ng4))

        # Pitch constraints (depend on pitch value b)
        # These constraints encode conditional logic for different pitch values
        g[52] = ((0.945 * c1 - Np1 - Ng1) * (b1 - 5.715) * (b1 - 8.255) * (b1 - 12.70) * (-1.0))
        g[53] = ((0.945 * c2 - Np2 - Ng2) * (b2 - 5.715) * (b2 - 8.255) * (b2 - 12.70) * (-1.0))
        g[54] = ((0.945 * c3 - Np3 - Ng3) * (b3 - 5.715) * (b3 - 8.255) * (b3 - 12.70) * (-1.0))
        g[55] = ((0.945 * c4 - Np4 - Ng4) * (b4 - 5.715) * (b4 - 8.255) * (b4 - 12.70) * (-1.0))

        g[56] = ((0.646 * c1 - Np1 - Ng1) * (b1 - 3.175) * (b1 - 8.255) * (b1 - 12.70) * (+1.0))
        g[57] = ((0.646 * c2 - Np2 - Ng2) * (b2 - 3.175) * (b2 - 8.255) * (b2 - 12.70) * (+1.0))
        g[58] = ((0.646 * c3 - Np3 - Ng3) * (b3 - 3.175) * (b3 - 8.255) * (b3 - 12.70) * (+1.0))
        g[59] = ((0.646 * c4 - Np4 - Ng4) * (b4 - 3.175) * (b4 - 8.255) * (b4 - 12.70) * (+1.0))

        g[60] = ((0.504 * c1 - Np1 - Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 12.70) * (-1.0))
        g[61] = ((0.504 * c2 - Np2 - Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 12.70) * (-1.0))
        g[62] = ((0.504 * c3 - Np3 - Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 12.70) * (-1.0))
        g[63] = ((0.504 * c4 - Np4 - Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 12.70) * (-1.0))

        g[64] = ((0.0 * c1 - Np1 - Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 8.255) * (+1.0))
        g[65] = ((0.0 * c2 - Np2 - Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 8.255) * (+1.0))
        g[66] = ((0.0 * c3 - Np3 - Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 8.255) * (+1.0))
        g[67] = ((0.0 * c4 - Np4 - Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 8.255) * (+1.0))

        g[68] = ((-1.812 * c1 + Np1 + Ng1) * (b1 - 5.715) * (b1 - 8.255) * (b1 - 12.70) * (-1.0))
        g[69] = ((-1.812 * c2 + Np2 + Ng2) * (b2 - 5.715) * (b2 - 8.255) * (b2 - 12.70) * (-1.0))
        g[70] = ((-1.812 * c3 + Np3 + Ng3) * (b3 - 5.715) * (b3 - 8.255) * (b3 - 12.70) * (-1.0))
        g[71] = ((-1.812 * c4 + Np4 + Ng4) * (b4 - 5.715) * (b4 - 8.255) * (b4 - 12.70) * (-1.0))

        g[72] = ((-0.945 * c1 + Np1 + Ng1) * (b1 - 3.175) * (b1 - 8.255) * (b1 - 12.70) * (+1.0))
        g[73] = ((-0.945 * c2 + Np2 + Ng2) * (b2 - 3.175) * (b2 - 8.255) * (b2 - 12.70) * (+1.0))
        g[74] = ((-0.945 * c3 + Np3 + Ng3) * (b3 - 3.175) * (b3 - 8.255) * (b3 - 12.70) * (+1.0))
        g[75] = ((-0.945 * c4 + Np4 + Ng4) * (b4 - 3.175) * (b4 - 8.255) * (b4 - 12.70) * (+1.0))

        g[76] = ((-0.646 * c1 + Np1 + Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 12.70) * (-1.0))
        g[77] = ((-0.646 * c2 + Np2 + Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 12.70) * (-1.0))
        g[78] = ((-0.646 * c3 + Np3 + Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 12.70) * (-1.0))
        g[79] = ((-0.646 * c4 + Np4 + Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 12.70) * (-1.0))

        g[80] = ((-0.504 * c1 + Np1 + Ng1) * (b1 - 3.175) * (b1 - 5.715) * (b1 - 8.255) * (+1.0))
        g[81] = ((-0.504 * c2 + Np2 + Ng2) * (b2 - 3.175) * (b2 - 5.715) * (b2 - 8.255) * (+1.0))
        g[82] = ((-0.504 * c3 + Np3 + Ng3) * (b3 - 3.175) * (b3 - 5.715) * (b3 - 8.255) * (+1.0))
        g[83] = ((-0.504 * c4 + Np4 + Ng4) * (b4 - 3.175) * (b4 - 5.715) * (b4 - 8.255) * (+1.0))

        # Output speed constraints (ratio between input and output)
        g[84] = wmin - w1 * (Np1 * Np2 * Np3 * Np4) / (Ng1 * Ng2 * Ng3 * Ng4)
        g[85] = -wmax + w1 * (Np1 * Np2 * Np3 * Np4) / (Ng1 * Ng2 * Ng3 * Ng4)

        # Replace inf with large penalty
        g = np.where(np.isinf(g), 1e6, g)
        g = np.where(np.isnan(g), 1e6, g)

        # ===== EQUALITY CONSTRAINTS =====
        # No equality constraints
        h = np.array([0.0])

        # ===== PENALTY =====
        tol = 1e-4
        penalty_h = 0.0
        penalty_g = np.sum(np.maximum(0, g) ** 2)
        fitness = f + 1e5 * (penalty_h + penalty_g)

        return fitness
