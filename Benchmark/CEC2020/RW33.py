from Benchmark.Functions.__Function import Function
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class RW33(Function):
    def __init__(self, dim):
        # Topology optimization parameters
        self.nelx = 3  # Number of elements in x-direction
        self.nely = 10  # Number of elements in y-direction
        self.penal = 3  # Penalization factor (SIMP model)
        self.volfrac = 0.5  # Volume fraction (50% of domain)
        self.rmin = 1.5  # Filter radius for sensitivities

        # Total design variables: nelx × nely = 3 × 10 = 30
        n_vars = self.nelx * self.nely

        x_start = np.ones(n_vars) * self.volfrac
        f_x_start = 0

        domain = [(0.001, 1.0) for _ in range(n_vars)]

        name = "RW33"
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

        # ===== PRECOMPUTE ELEMENT STIFFNESS MATRIX =====
        self.KE = self._compute_element_stiffness()

    def _compute_element_stiffness(self):
        """
        Compute element stiffness matrix for 4-node quadrilateral (Q4) element
        Using plane stress formulation
        """
        # Young's modulus and Poisson's ratio
        E = 1.0
        nu = 0.3

        k = np.array([
            12, 3, -6, -3, -8, -3, -4, 3,
            3, 12, 3, 0, -3, -8, -3, -4,
            -6, 3, 12, -3, -4, -3, -8, 3,
            -3, 0, -3, 12, 3, -4, 3, -8,
            -8, -3, -4, 3, 12, 3, -6, -3,
            -3, -8, -3, -4, 3, 12, 3, 0,
            -4, -3, -8, 3, -6, 3, 12, -3,
            3, -4, 3, -8, -3, 0, -3, 12
        ], dtype=float)

        KE = (E / (1 - nu ** 2) / 24.0) * k.reshape(8, 8)
        return KE

    def _build_global_stiffness(self, X):
        """
        Assemble global stiffness matrix using SIMP material model
        K = Σ(ρ_e^p × K_e)
        """
        nelx = self.nelx
        nely = self.nely
        penal = self.penal

        # Number of DOFs: 2 per node
        n_nodes = (nelx + 1) * (nely + 1)
        n_dofs = 2 * n_nodes

        # Lists for sparse matrix construction (COO format)
        I_list = []
        J_list = []
        V_list = []

        # Element loop
        for ely in range(1, nely + 1):
            for elx in range(1, nelx + 1):
                # Element density (design variable)
                rho = X[ely - 1, elx - 1]

                # SIMP model: E(ρ) = ρ^p × E0
                ke_scaled = (rho ** penal) * self.KE

                # Node numbering (1-indexed in MATLAB, 0-indexed in Python)
                n1 = (nely + 1) * (elx - 1) + ely
                n2 = (nely + 1) * elx + ely

                # DOF indices (0-indexed)
                edof = np.array([
                    2 * n1 - 2, 2 * n1 - 1,
                    2 * n2 - 2, 2 * n2 - 1,
                    2 * n2 + 1 - 2, 2 * n2 + 1 - 1,
                    2 * n1 + 1 - 2, 2 * n1 + 1 - 1
                ], dtype=int)

                # Add element stiffness contributions
                for i in range(8):
                    for j in range(8):
                        I_list.append(edof[i])
                        J_list.append(edof[j])
                        V_list.append(ke_scaled[i, j])

        # Create sparse global stiffness matrix
        K_global = csr_matrix(
            (V_list, (I_list, J_list)),
            shape=(n_dofs, n_dofs)
        )

        return K_global

    def _apply_boundary_conditions(self, K, F, n_dofs):
        """
        Apply boundary conditions:
        - Left edge (x=0): fixed (u=0, v=0)
        - Uniform load on right edge (y-direction)
        """
        nely = self.nely

        # Fixed DOFs: left edge nodes
        fixed_dofs = np.array([2 * i - 1 - 1 for i in range(1, nely + 2)])
        fixed_dofs = np.append(fixed_dofs, [2 * i - 2 for i in range(1, nely + 2)])
        fixed_dofs = np.sort(np.unique(fixed_dofs))

        # Free DOFs
        free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

        # Load: right edge, top node, vertical load
        n_top = (nely + 1) * (self.nelx + 1)
        F[2 * n_top - 1 - 1] = -1.0  # Apply downward load

        return K, F, free_dofs, fixed_dofs

    def _compute_sensitivities(self, X, U):
        """
        Compute design sensitivities (objective gradient w.r.t. design variables)
        dc/dρ = -p × ρ^(p-1) × Ue^T × KE × Ue
        """
        nelx = self.nelx
        nely = self.nely
        penal = self.penal

        dc = np.zeros((nely, nelx))

        for ely in range(1, nely + 1):
            for elx in range(1, nelx + 1):
                # Node numbering
                n1 = (nely + 1) * (elx - 1) + ely
                n2 = (nely + 1) * elx + ely

                # Element DOF displacement vector
                edof = np.array([
                    2 * n1 - 2, 2 * n1 - 1,
                    2 * n2 - 2, 2 * n2 - 1,
                    2 * n2 + 1 - 2, 2 * n2 + 1 - 1,
                    2 * n1 + 1 - 2, 2 * n1 + 1 - 1
                ], dtype=int)

                Ue = U[edof, 0] if U.ndim > 1 else U[edof]

                # Sensitivity: dc/dρ = -p × ρ^(p-1) × Ue^T × KE × Ue
                dc[ely - 1, elx - 1] = (
                        -penal * (X[ely - 1, elx - 1] ** (penal - 1)) *
                        (Ue @ self.KE @ Ue)
                )

        return dc

    def _filter_sensitivities(self, dc, X):
        """
        Density filter to avoid checkerboard patterns
        dc_filtered[i] = Σ(w[j] × dc[j]) / Σ(w[j])
        """
        nely = self.nely
        nelx = self.nelx
        rmin = self.rmin

        dc_filtered = np.zeros_like(dc)

        for ely in range(nely):
            for elx in range(nelx):
                sum_w = 0.0
                sum_wdc = 0.0

                # Neighborhood search
                for ely2 in range(max(0, ely - int(np.ceil(rmin))),
                                  min(nely, ely + int(np.ceil(rmin)) + 1)):
                    for elx2 in range(max(0, elx - int(np.ceil(rmin))),
                                      min(nelx, elx + int(np.ceil(rmin)) + 1)):
                        distance = np.sqrt((ely - ely2) ** 2 + (elx - elx2) ** 2)

                        if distance <= rmin:
                            weight = rmin - distance
                            sum_w += weight * X[ely2, elx2]
                            sum_wdc += weight * X[ely2, elx2] * dc[ely2, elx2]

                dc_filtered[ely, elx] = sum_wdc / sum_w if sum_w > 0 else 0.0

        return dc_filtered

    def eval(self, variables_values):
        """
        Evaluate topology optimization problem

        Objective: Minimize compliance (maximize stiffness)
        C = F^T × U = Σ(ρ_e^p × Ue^T × KE × Ue)

        Subject to:
        - Volume constraint: Σ(ρ_e) ≤ V_frac × V_total
        - 0 < ρ_min ≤ ρ_e ≤ 1
        - K × U = F (equilibrium equation)
        """

        x = np.array(variables_values).reshape(1, -1)[0]

        # Reshape design variables into density matrix
        X = x.reshape((self.nely, self.nelx))
        X = np.maximum(0.001, np.minimum(1.0, X))  # Enforce bounds

        # ===== FE ANALYSIS =====
        nelx = self.nelx
        nely = self.nely
        n_nodes = (nelx + 1) * (nely + 1)
        n_dofs = 2 * n_nodes

        # Build global stiffness matrix
        K_global = self._build_global_stiffness(X)

        # Initialize force vector
        F = np.zeros((n_dofs, 1))

        # Apply boundary conditions
        K_global, F, free_dofs, fixed_dofs = self._apply_boundary_conditions(
            K_global, F, n_dofs
        )

        # Reduce system to free DOFs
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)].tocsr()
        F_reduced = F[free_dofs, :]

        # Solve: K × U = F
        try:
            U_reduced = spsolve(K_reduced, F_reduced.ravel())
            if U_reduced.ndim == 1:
                U_reduced = U_reduced.reshape(-1, 1)
        except:
            return 1e10

        # Reconstruct full displacement vector
        U = np.zeros((n_dofs, 1))
        U[free_dofs, :] = U_reduced.reshape(-1, 1)

        # ===== OBJECTIVE FUNCTION =====
        # Compliance: C = U^T × K × U = F^T × U
        c = float(F.ravel() @ U.ravel())

        f = c

        # ===== SENSITIVITIES =====
        dc = self._compute_sensitivities(X, U)

        # ===== FILTER SENSITIVITIES =====
        dc_filtered = self._filter_sensitivities(dc, X)

        # ===== VOLUME CONSTRAINT =====
        volume_current = np.mean(X)
        g_volume = volume_current - self.volfrac

        # ===== INEQUALITY CONSTRAINTS =====
        g = np.array([g_volume])

        # ===== EQUALITY CONSTRAINTS =====
        h = np.array([0.0])

        # ===== PENALTY METHOD =====
        penalty_g = np.sum(np.maximum(0, g) ** 2)

        if np.isnan(f) or np.isinf(f) or f > 1e10:
            return 1e10

        fitness = f + 1e4 * penalty_g

        return fitness
