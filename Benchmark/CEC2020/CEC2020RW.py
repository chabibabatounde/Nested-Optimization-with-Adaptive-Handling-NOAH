from Benchmark.Functions.__Function import Function
import numpy as np


class CEC2020RW(Function):
    """
    Wrapper générique pour les problèmes CEC2020-RW.

    Chaque problème doit définir :
        - objective(x)
        - inequality_constraints(x) : g_i(x) <= 0
        - equality_constraints(x)   : h_j(x) = 0

    L'évaluation renvoie par défaut :
        f_penalized = f(x) + penalty_factor * violation(x)

    avec :
        violation(x) =
            sum(max(0, g_i(x))^2) + sum(max(0, abs(h_j(x)) - eps)^2)
    """

    def __init__(
        self,
        dim,
        domain,
        x_start,
        f_x_start,
        max_dimension,
        name,
        f_star=None,
        penalty_factor=1e10,
        equality_tolerance=1e-4,
    ):
        super().__init__(
            dim,
            domain,
            x_start,
            f_x_start,
            max_dimension,
            name
        )

        self.f_star = f_star
        self.penalty_factor = penalty_factor
        self.equality_tolerance = equality_tolerance
        self.types(["CEC2020RW", "Constrained", "Real-world"])

    # ------------------------------------------------------------------
    # Méthodes à redéfinir dans chaque problème
    # ------------------------------------------------------------------

    def objective(self, x):
        raise NotImplementedError(
            f"objective(x) doit être implémentée pour {self.name()}"
        )

    def inequality_constraints(self, x):
        """
        Retourne un tableau g(x), avec convention :
            g_i(x) <= 0
        """
        return np.array([])

    def equality_constraints(self, x):
        """
        Retourne un tableau h(x), avec convention :
            h_j(x) = 0
        """
        return np.array([])

    # ------------------------------------------------------------------
    # Gestion des violations
    # ------------------------------------------------------------------

    def constraint_violation(self, x):
        g = np.asarray(self.inequality_constraints(x), dtype=float)
        h = np.asarray(self.equality_constraints(x), dtype=float)

        g_violation = np.sum(np.maximum(0.0, g) ** 2)

        h_violation = np.sum(
            np.maximum(0.0, np.abs(h) - self.equality_tolerance) ** 2
        )

        return g_violation + h_violation

    def is_feasible(self, x):
        return self.constraint_violation(x) <= 0.0

    # ------------------------------------------------------------------
    # Évaluation compatible avec ton framework
    # ------------------------------------------------------------------

    def eval(self, variables_values):
        x = np.asarray(variables_values, dtype=float)

        f = self.objective(x)
        violation = self.constraint_violation(x)

        return f + self.penalty_factor * violation

    def eval_raw(self, variables_values):
        """
        Renvoie uniquement f(x), sans pénalité.
        Utile pour analyser les résultats.
        """
        x = np.asarray(variables_values, dtype=float)
        return self.objective(x)

    def eval_constraints(self, variables_values):
        """
        Renvoie les contraintes séparément.
        """
        x = np.asarray(variables_values, dtype=float)
        return {
            "g": self.inequality_constraints(x),
            "h": self.equality_constraints(x),
            "violation": self.constraint_violation(x),
        }
