from Benchmark.Functions.__Function import Function
import numpy as np
import os


class RW42(Function):
    def __init__(self, dim):

        self.load_network_data()

        n_vars = 86

        # Initial conditions
        x_start = np.zeros(n_vars)
        x_start[0:37] = 1.0  # V_real (buses 2-38)
        x_start[37:74] = 0.0  # V_imag
        x_start[74] = 0.0  # w (frequency deviation)
        x_start[75] = 1.0  # V_slack
        x_start[76:81] = 0.2  # P_c (DER setpoints = 0.2 MW)
        x_start[81:86] = 0.18  # Q_c (DER setpoints = 0.18 MVAr)

        # Domain bounds
        domain = [
            # Voltage real parts: V_r ∈ [0.9, 1.1] pu
            *[(0.9, 1.1) for _ in range(37)],
            # Voltage imag parts: V_i ∈ [-0.3, 0.3]
            *[(-0.3, 0.3) for _ in range(37)],
            # Frequency deviation: w ∈ [-0.05, 0.05] (±3 Hz @ 60Hz)
            (-0.05, 0.05),
            # Slack voltage: V_slack ∈ [0.95, 1.05] pu
            (0.95, 1.05),
            # P_c setpoints: [0.05, 0.3] MW (5 DERs)
            *[(0.05, 0.3) for _ in range(5)],
            # Q_c setpoints: [0.05, 0.3] MVAr (5 DERs)
            *[(0.05, 0.3) for _ in range(5)],
        ]

        f_x_start = 0
        name = "RW42"
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

        # ===== ISLANDED MICROGRID PARAMETERS =====

        # DER bus indices (0-indexed)
        self.der_buses = np.array([33, 34, 35, 36, 37])  # Buses 34-38

        # Droop coefficients (can be fixed or parameterized)
        # Standard values for frequency/voltage support
        self.m_p = 0.02  # Active power droop [pu/pu]
        self.m_q = 0.03  # Reactive power droop [pu/pu]

        # ===== NETWORK ADMITTANCE MATRIX =====
        self.Y_base = None  # Base admittance (no frequency)

        print("✓ Droop controller optimization (RC42) initialized")
        print(f"  - Variables: {n_vars}")
        print(f"  - Network: 38 buses (islanded)")
        print(f"  - DERs: 5 controllable units")
        print(f"  - Control: Frequency + Voltage Droop")
        print(f"  - Objective: Minimize active power losses")

    def load_network_data(self):
        """
        Load 38-bus islanded microgrid network data
        """
        try:
            base_path = os.getcwd() + '/Benchmark/Publication/CEC2020/input_data/'

            self.P = np.loadtxt(os.path.join(base_path, "FunctionPS2_P.txt"))
            self.Q = np.loadtxt(os.path.join(base_path, "FunctionPS2_Q.txt"))
            self.L = np.loadtxt(os.path.join(base_path, "FunctionPS14_linedata.txt"))

        except Exception as e:
            print(f"⚠️  Network data not found: {e}")
            print("    Using synthetic 38-bus islanded microgrid")

            # Create synthetic network
            self._create_synthetic_network()

    def _create_synthetic_network(self):
        """
        Create synthetic 38-bus islanded distribution network
        """
        n_bus = 38
        np.random.seed(42)

        # ===== SYNTHETIC LOAD DATA =====
        self.P = np.zeros((n_bus, 6))
        self.Q = np.zeros((n_bus, 6))

        # Active loads (radial decay)
        for i in range(1, 33):
            decay = np.exp(-0.05 * i)
            self.P[i, 0] = np.random.uniform(0.08, 0.18) * decay
            self.P[i, 1] = 1.0
            self.P[i, 5] = 1.0  # V_ref
            self.P[i, 6] = 1.0  # V_exponent

        # Reactive loads
        for i in range(1, 33):
            decay = np.exp(-0.05 * i)
            self.Q[i, 0] = np.random.uniform(0.03, 0.12) * decay
            self.Q[i, 1] = 1.0
            self.Q[i, 5] = 1.0
            self.Q[i, 6] = 2.0

        # ===== SYNTHETIC LINE DATA =====
        # Format: [from, to, R, X, B/2]
        self.L = np.zeros((37, 5))

        for i in range(37):
            self.L[i, 0] = i + 1  # From bus
            self.L[i, 1] = i + 2  # To bus
            self.L[i, 2] = 0.01 + 0.003 * np.sin(i / 15)  # R
            self.L[i, 3] = 0.04 + 0.015 * np.cos(i / 15)  # X
            self.L[i, 4] = 0.0001  # B/2

        print("  ✓ Synthetic 38-bus islanded network created")

    def build_ybus(self, frequency_deviation=0.0):
        """
        Build bus admittance matrix (Y-bus)

        For frequency-dependent components:
        - Induction motors: impedance changes with frequency
        - Power electronics: frequency-synchronized control

        Args:
            frequency_deviation: w (per unit deviation from nominal)

        Returns:
            Y: Complex admittance matrix (38 × 38)
        """
        n_bus = 38
        Y = np.zeros((n_bus, n_bus), dtype=complex)

        # Build from line data
        n_lines = self.L.shape[0]

        for k in range(n_lines):
            i = int(self.L[k, 0] - 1)  # From bus (0-indexed)
            j = int(self.L[k, 1] - 1)  # To bus
            R = self.L[k, 2]
            X = self.L[k, 3]
            B = self.L[k, 4]

            # Series impedance
            Z = R + 1j * X
            y = 1.0 / Z

            # Frequency-dependent reactance (approximate)
            # X' = X × (1 + 0.1 × w)  for induction motors
            X_freq = X * (1 + 0.1 * frequency_deviation)
            Z_freq = R + 1j * X_freq
            y_freq = 1.0 / Z_freq

            # Add series admittance (both directions)
            Y[i, j] -= y_freq
            Y[j, i] -= y_freq

            # Add to self-admittance
            Y[i, i] += y_freq
            Y[j, j] += y_freq

            # Shunt admittance
            y_shunt = 1j * B
            Y[i, i] += y_shunt
            Y[j, j] += y_shunt

        return Y

    def eval(self, variables_values):
        """
        Evaluate islanded microgrid with droop control

        Objective: Minimize active power losses
        f = ∑P_sp (net active power at all buses)

        Why minimize ∑P_sp?
        - In islanded microgrid: ∑P_gen = ∑P_load + P_loss
        - ∑P_sp = ∑(P_gen - P_load) = P_loss
        - Minimizing ∑P_sp minimizes total losses

        Constraints (h):
        - Power balance: Current mismatch at each bus
        - ΔI = I_calculated - I_required = 0
        """

        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # ===== EXTRACT VARIABLES =====
        V_real = x[0:37]
        V_imag = x[37:74]
        V_238 = V_real + 1j * V_imag

        w = x[74]  # Frequency deviation
        V_slack = x[75]  # Slack voltage

        P_c = x[76:81]  # Active power setpoints (5 DERs)
        Q_c = x[81:86]  # Reactive power setpoints (5 DERs)

        # ===== BUILD FULL STATE VECTOR =====
        V = np.zeros(38, dtype=complex)
        V[0] = V_slack + 0.0j  # Slack bus
        V[1:38] = V_238

        # ===== BUILD Y-BUS (FREQUENCY-DEPENDENT) =====
        Y = self.build_ybus(frequency_deviation=w)

        # ===== POWER FLOW CALCULATION =====
        I = Y @ V
        I_real = np.real(I)
        I_imag = np.imag(I)

        V_mag = np.abs(V)
        V_ref = 1.0

        # ===== DER POWER WITH DROOP CONTROL =====
        # Active power droop: P = P_c × (1 - w)
        # Reactive power droop: Q = Q_c × (1 - |V|)

        P_der = np.zeros(38)
        Q_der = np.zeros(38)

        for i, bus_idx in enumerate(self.der_buses):
            # Active power: frequency droop
            P_der[bus_idx] = P_c[i] * (1.0 - w)

            # Reactive power: voltage droop
            Q_der[bus_idx] = Q_c[i] * (1.0 - V_mag[bus_idx])

        # ===== LOAD POWER (Voltage-Dependent ZIP) =====
        P_load_coeff = self.P[:, 0]
        Q_load_coeff = self.Q[:, 0]

        P_load = P_load_coeff * (V_mag / V_ref) ** self.P[:, 6]
        Q_load = Q_load_coeff * (V_mag / V_ref) ** self.Q[:, 6]

        # ===== NET POWER AT EACH BUS =====
        P_available = P_der - P_load
        Q_available = Q_der - Q_load

        # ===== POWER INJECTED INTO NETWORK =====
        P_injected = V_real * I_real + V_imag * I_imag
        Q_injected = V_imag * I_real - V_real * I_imag

        # ===== POWER MISMATCH =====
        delta_P = P_injected - P_available
        delta_Q = Q_injected - Q_available

        # ===== OBJECTIVE FUNCTION =====
        # Minimize sum of net active power (= minimize losses)
        # P_sp = P_available = P_der - P_load
        P_sp = P_available

        # Sum of positive power injections ≈ total generation
        # Sum of all P_sp ≈ losses + unmet demand
        objective = np.sum(P_sp)

        # ===== CONSTRAINT PENALTIES =====

        # 1. Power balance violations (current mismatch)
        balance_error = np.sum(delta_P ** 2) + np.sum(delta_Q ** 2)

        # 2. Voltage magnitude constraints (0.9-1.1 pu)
        V_violation = (
                np.sum(np.maximum(0, 0.9 - V_mag) ** 2) +
                np.sum(np.maximum(0, V_mag - 1.1) ** 2)
        )

        # 3. Current magnitude limits
        I_mag = np.abs(I)
        I_max = 2.0  # Per unit
        I_violation = np.sum(np.maximum(0, I_mag - I_max) ** 2)

        # 4. Frequency stability (small deviation)
        freq_violation = np.maximum(0, abs(w) - 0.05) ** 2

        # 5. Slack voltage near nominal
        slack_violation = (V_slack - 1.0) ** 2

        # ===== TOTAL PENALTY =====
        penalty = (1e3 * balance_error +
                   1e3 * V_violation +
                   1e3 * I_violation +
                   5e2 * freq_violation +
                   5e2 * slack_violation)

        # ===== TOTAL FITNESS =====
        # Note: objective is typically negative (sum of power)
        # Add penalty to make minimization problem
        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else 1e10


class RW42_Simplified(Function):
    """
    Simplified version of droop controller optimization
    without complex frequency-dependent Y-bus
    """

    def __init__(self, dim):
        """
        Simplified droop controller optimization (86 variables)
        """
        n_vars = 86

        x_start = np.zeros(n_vars)
        x_start[0:37] = 1.0  # V_real
        x_start[37:74] = 0.0  # V_imag
        x_start[74] = 0.0  # w
        x_start[75] = 1.0  # V_slack
        x_start[76:81] = 0.2  # P_c
        x_start[81:86] = 0.18  # Q_c

        domain = [
            *[(0.9, 1.1) for _ in range(37)],  # V_real
            *[(-0.3, 0.3) for _ in range(37)],  # V_imag
            (-0.05, 0.05),  # w
            (0.95, 1.05),  # V_slack
            *[(0.05, 0.3) for _ in range(5)],  # P_c
            *[(0.05, 0.3) for _ in range(5)],  # Q_c
        ]

        Function.__init__(
            self, dim, domain, x_start, 0,
            max_dimension=n_vars, name="RW42_Simplified"
        )
        self.types(['CEC2020RW'])

        # Synthetic load profile
        np.random.seed(42)
        n_bus = 38

        self.P_load = np.zeros(n_bus)
        self.P_load[1:33] = np.random.uniform(0.08, 0.18, 32)

        self.Q_load = np.zeros(n_bus)
        self.Q_load[1:33] = np.random.uniform(0.03, 0.12, 32)

        # Line impedances
        self.Z_lines = np.array([
            (0.01 + 0.003 * np.sin(i / 15)) + 1j * (0.04 + 0.015 * np.cos(i / 15))
            for i in range(37)
        ])

        self.der_buses = [33, 34, 35, 36, 37]

    def eval(self, variables_values):
        """
        Simplified droop control evaluation
        """
        x = np.array(variables_values, dtype=float).reshape(1, -1)[0]

        # Extract variables
        V_real = x[0:37]
        V_imag = x[37:74]
        V_238 = V_real + 1j * V_imag

        w = x[74]
        V_slack = x[75]
        P_c = x[76:81]
        Q_c = x[81:86]

        # Build voltage vector
        V = np.zeros(38, dtype=complex)
        V[0] = V_slack + 0.0j
        V[1:38] = V_238

        # ===== SIMPLIFIED POWER FLOW =====
        I = np.zeros(38, dtype=complex)

        # Radial network approximation
        for i in range(36, -1, -1):
            # DER power with droop
            if i in self.der_buses:
                idx = self.der_buses.index(i)
                P_der = P_c[idx] * (1.0 - w)
                Q_der = Q_c[idx] * (1.0 - np.abs(V[i]))
            else:
                P_der = 0.0
                Q_der = 0.0

            # Load power
            P_load = self.P_load[i]
            Q_load = self.Q_load[i]

            # Net power
            S_net = (P_der - P_load) + 1j * (Q_der - Q_load)

            # Current from bus
            if np.abs(V[i]) > 1e-6:
                I[i] = np.conj(S_net / V[i])

            # Current accumulation (radial)
            if i > 0:
                I[i] += I[i + 1]

        # Voltage drop across lines
        for i in range(37):
            V[i + 1] = V[i] - self.Z_lines[i] * I[i]

        # ===== OBJECTIVE: MINIMIZE LOSSES =====
        # Losses = sum of P_der - sum of P_load (at steady-state)
        total_P_der = 0.0

        for idx, bus in enumerate(self.der_buses):
            P_der = P_c[idx] * (1.0 - w)
            total_P_der += P_der

        total_P_load = np.sum(self.P_load)

        # Objective: minimize net generation (= minimize losses)
        losses_approx = total_P_der - total_P_load
        objective = losses_approx

        # ===== PENALTIES =====
        V_mag = np.abs(V)
        I_mag = np.abs(I)

        penalty = (
            # Power balance
                1e3 * np.sum(np.abs(I[1:]) ** 2) +
                # Voltage constraints
                1e3 * (np.sum(np.maximum(0, 0.9 - V_mag) ** 2) +
                       np.sum(np.maximum(0, V_mag - 1.1) ** 2)) +
                # Current limits
                1e3 * np.sum(np.maximum(0, I_mag - 2.0) ** 2) +
                # Frequency stability
                5e2 * abs(w) ** 2
        )

        fitness = objective + penalty

        return fitness if np.isfinite(fitness) else 1e10
