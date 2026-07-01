"""Generalized Dynamic Factor Model (GDFM) — Eurocoin baseline.

Implements the methodology described in:
- Altissimo et al. (2007), "New Eurocoin: Tracking Economic Growth in Real Time"
- Forni, Hallin, Lippi, Reichlin (2000, 2005)
- Aprigliano, Emiliozzi, Lippi (2022)

The GDFM estimates smooth common factors from a large panel of macro series
by maximizing the ratio of common low-frequency variance to total variance.

Pipeline:
1. Transform the panel to stationary series (growth rates, etc.)
2. Estimate spectral density matrices (common vs idiosyncratic)
3. Solve generalized eigenvalue problem for smooth factors
4. Project the MLRG target onto the smooth factors

Key equations:
- Common-idiosyncratic decomposition: x_it = χ_it + ξ_it
- Common component: χ_it = b_i1(L)u_1t + ... + b_iq(L)u_qt
- Generalized eigenvalue problem: Σ̂_φ v_k = λ_k (Σ̂_χ + Σ̂_ξ) v_k
- MLRG projection: ĉ_t = μ̂ + Σ̂_cw Σ̂_w^{-1} w_t
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class GDFMResult:
    """Result of GDFM estimation."""

    factors: np.ndarray  # shape (T, q) — smooth common factors
    loadings: np.ndarray  # shape (N, q) — factor loadings
    eigenvalues: np.ndarray  # shape (q,) — generalized eigenvalues
    projected_mlrg: np.ndarray  # shape (T,) — MLRG projection onto factors
    n_factors: int
    variance_ratio: float  # fraction of total variance explained
    method: str = "GDFM-FHLR"


class GDFM:
    """Generalized Dynamic Factor Model estimator.

    Implements the frequency-domain approach of Forni, Hallin, Lippi, Reichlin
    for extracting smooth common factors from a large macro panel.

    Parameters:
        n_factors: Number of common factors to extract (q).
        max_lag: Maximum lag for spectral density estimation.
        freq_band: Tuple (omega_low, omega_high) defining the target frequency band.
                   Default: (0, pi/6) — oscillations with period > 1 year.
        bandwidth: Bandwidth for spectral density kernel smoothing.
    """

    def __init__(
        self,
        n_factors: int = 3,
        max_lag: int = 12,
        freq_band: tuple[float, float] = (0.0, np.pi / 6),
        bandwidth: int | None = None,
    ) -> None:
        self.n_factors = n_factors
        self.max_lag = max_lag
        self.freq_band = freq_band
        self.bandwidth = bandwidth or max(1, max_lag // 2)

    def fit_transform(
        self,
        panel: np.ndarray,
        target: np.ndarray | None = None,
    ) -> GDFMResult:
        """Estimate GDFM factors and optionally project onto target.

        Args:
            panel: 2D array (T × N) of stationary, demeaned series.
                   T = time steps, N = number of series.
            target: Optional 1D array (T,) of the MLRG target for projection.

        Returns:
            GDFMResult with factors, loadings, eigenvalues, and projected MLRG.
        """
        T, N = panel.shape
        logger.info("GDFM estimation: T=%d, N=%d, q=%d", T, N, self.n_factors)

        # Step 1: Demean and standardize
        X = self._standardize(panel)

        # Step 2: Estimate spectral density matrices
        logger.info("Estimating spectral density matrices (max_lag=%d)...", self.max_lag)
        Sigma_total = self._estimate_spectral_density(X, band="total")
        Sigma_idio = self._estimate_spectral_density(X, band="idiosyncratic")
        Sigma_common = Sigma_total - Sigma_idio

        # Step 3: Frequency-band filtered covariance
        Sigma_phi = self._band_filtered_covariance(X)

        # Step 4: Generalized eigenvalue problem
        logger.info("Solving generalized eigenvalue problem...")
        # Σ̂_φ v = λ (Σ̂_χ + Σ̂_ξ) v
        # Since Σ̂_φ + Σ̂_ξ = Σ̂_total (at the band-filtered level), we use:
        eigenvalues, eigenvectors = self._solve_gevp(Sigma_phi, Sigma_total)

        # Select top-q factors
        q = min(self.n_factors, len(eigenvalues))
        top_indices = np.argsort(eigenvalues)[::-1][:q]
        self.loadings = eigenvectors[:, top_indices]
        self.eigenvalues_all = eigenvalues[top_indices]

        # Step 5: Compute factor scores
        factors = X @ self.loadings  # (T, q)
        logger.info(
            "Factors extracted: shape=%s, eigenvalues=%s",
            factors.shape, np.round(self.eigenvalues_all, 4),
        )

        # Step 6: Project MLRG if target provided
        if target is not None:
            projected = self._project_target(factors, target)
        else:
            projected = np.full(T, np.nan)

        variance_ratio = float(np.sum(self.eigenvalues_all) / np.sum(eigenvalues))

        return GDFMResult(
            factors=factors,
            loadings=self.loadings,
            eigenvalues=self.eigenvalues_all,
            projected_mlrg=projected,
            n_factors=q,
            variance_ratio=variance_ratio,
        )

    def _standardize(self, panel: np.ndarray) -> np.ndarray:
        """Demean and standardize each column."""
        X = panel.copy()
        col_means = np.nanmean(X, axis=0)
        col_stds = np.nanstd(X, axis=0)
        col_stds = np.where(col_stds == 0, 1, col_stds)
        X = (X - col_means) / col_stds
        # Fill remaining NaNs with 0
        X = np.nan_to_num(X, nan=0.0)
        return X

    def _estimate_spectral_density(
        self,
        X: np.ndarray,
        band: str = "total",
    ) -> np.ndarray:
        """Estimate the spectral density matrix using Bartlett (triangular) kernel.

        For the "total" band: Σ_total(0) = covariance at lag 0 (standard covariance matrix)
        For the "idiosyncratic" band: estimated via the difference between total
        and a PCA-based common component.

        This is a simplified version of the FHLR approach. The full FHLR method
        uses dynamic PCA with frequency-domain estimation of the common component.
        """
        T, N = X.shape

        if band == "total":
            # Total covariance at lag 0 (with Bartlett smoothing)
            cov = np.zeros((N, N))
            for lag in range(self.max_lag + 1):
                weight = 1 - lag / (self.max_lag + 1)  # Bartlett kernel
                if lag == 0:
                    cov += weight * (X.T @ X) / T
                else:
                    cross = (X[lag:].T @ X[:-lag]) / (T - lag)
                    cov += weight * (cross + cross.T)
            return cov

        elif band == "idiosyncratic":
            # Estimate idiosyncratic component via residual from PCA
            # The common component is captured by the first few PCs
            cov_total = self._estimate_spectral_density(X, band="total")
            eigenvalues, eigenvectors = np.linalg.eigh(cov_total)

            # Use top components as "common"
            q_common = min(self.n_factors + 2, N - 1)
            top_idx = np.argsort(eigenvalues)[::-1][:q_common]
            common_cov = eigenvectors[:, top_idx] @ np.diag(eigenvalues[top_idx]) @ eigenvectors[:, top_idx].T

            idio_cov = cov_total - common_cov
            # Ensure positive semi-definite
            idio_cov = (idio_cov + idio_cov.T) / 2
            eigvals_idio = np.linalg.eigvalsh(idio_cov)
            if eigvals_idio.min() < 0:
                idio_cov -= np.eye(N) * (eigvals_idio.min() - 1e-8)
            return idio_cov

        else:
            raise ValueError(f"Unknown band: {band}")

    def _band_filtered_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute the covariance matrix filtered to the target frequency band.

        This estimates Σ_φ — the covariance of the common component restricted
        to the frequency band [omega_low, omega_high].

        Uses the frequency-domain approach: compute the cross-spectral density
        at frequencies within the band and integrate.
        """
        T, N = X.shape
        omega_low, omega_high = self.freq_band

        # Generate frequencies within the band
        n_freqs = max(50, T)
        omegas = np.linspace(omega_low, omega_high, n_freqs)

        # Compute spectral density at each frequency
        Sigma_phi = np.zeros((N, N), dtype=complex)

        for omega in omegas:
            # Cross-spectral density at frequency omega
            # Using the periodogram smoothed with Bartlett kernel
            S_omega = self._cross_spectrum(X, omega)
            Sigma_phi += S_omega * (omegas[1] - omegas[0])  # trapezoidal integration

        # Take the real part (imaginary should be ~0 after integration over symmetric band)
        return np.real(Sigma_phi)

    def _cross_spectrum(self, X: np.ndarray, omega: float) -> np.ndarray:
        """Compute the cross-spectral density matrix at frequency omega.

        S_ij(omega) = sum_k w(k) * C_ij(k) * exp(-i*omega*k)

        where C_ij(k) is the cross-covariance at lag k and w(k) is the Bartlett kernel.
        """
        T, N = X.shape
        S = np.zeros((N, N), dtype=complex)

        for lag in range(-self.max_lag, self.max_lag + 1):
            weight = 1 - abs(lag) / (self.max_lag + 1)
            if weight <= 0:
                continue

            # Cross-covariance at lag k
            if lag >= 0:
                if lag == 0:
                    C = (X.T @ X) / T
                else:
                    C = (X[lag:].T @ X[:-lag]) / (T - lag)
            else:
                C = (X[:lag].T @ X[-lag:]) / (T + lag)

            S += weight * C * np.exp(-1j * omega * lag)

        return S

    def _solve_gevp(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the generalized eigenvalue problem A v = λ B v.

        Returns eigenvalues and eigenvectors sorted in descending order.
        """
        # Ensure B is positive definite
        B = (B + B.T) / 2
        eigvals_B = np.linalg.eigvalsh(B)
        if eigvals_B.min() < 1e-10:
            B += np.eye(B.shape[0]) * (1e-10 - eigvals_B.min())

        # Solve via Cholesky decomposition
        L = np.linalg.cholesky(B)
        L_inv = np.linalg.inv(L)
        A_transformed = L_inv @ A @ L_inv.T

        # Standard eigenvalue problem
        eigenvalues, eigenvectors_transformed = np.linalg.eigh(A_transformed)
        eigenvectors = L_inv.T @ eigenvectors_transformed

        # Sort descending
        sort_idx = np.argsort(eigenvalues)[::-1]
        return eigenvalues[sort_idx], eigenvectors[:, sort_idx]

    def _project_target(
        self,
        factors: np.ndarray,
        target: np.ndarray,
        factor_freq: str = "monthly",
        target_freq: str = "quarterly",
    ) -> np.ndarray:
        """Project the MLRG target onto the smooth factors.

        Implements the Altissimo et al. (2007) projection:
            ĉ_t = μ̂ + Σ̂_cw Σ̂_w^{-1} w_t

        Cross-covariances between target and factors are estimated using
        frequency-domain methods: the cross-spectrum is integrated over the
        target frequency band [−π/6, π/6].

        For quarterly target with monthly factors, the factors are aggregated
        to quarterly frequency using the (1 + L + L²)² filter (as in Eurocoin).

        Args:
            factors: Factor scores, shape (T_monthly, q).
            target: MLRG target, shape (T_quarterly,).
            factor_freq: "monthly" or "quarterly".
            target_freq: "monthly" or "quarterly".

        Returns:
            Projected MLRG, same length as target.
        """
        T_target = len(target)
        q = factors.shape[1]

        # --- Step 1: Align frequencies ---
        if factor_freq == "monthly" and target_freq == "quarterly":
            # Aggregate monthly factors to quarterly using (1 + L + L²) filter
            # This averages 3 consecutive months, with squared weighting as in Eurocoin
            factors_q = self._monthly_to_quarterly(factors)
        else:
            factors_q = factors

        T_q = len(factors_q)
        min_T = min(T_q, T_target)

        # Align
        w = factors_q[:min_T]
        c = target[:min_T]

        # Drop NaN
        valid = ~np.isnan(c)
        w_valid = w[valid]
        c_valid = c[valid]

        if len(c_valid) < q + 5:
            logger.warning("Too few valid observations for projection: %d", len(c_valid))
            return np.full(T_target, np.nan)

        # --- Step 2: Estimate cross-covariance via frequency domain ---
        # The cross-covariance between target and factor j is:
        #   gamma_cj(k) = (1/2π) ∫_{-π/6}^{π/6} S_cj(ω) e^{iωk} dω
        # where S_cj(ω) is the cross-spectrum between target and factor j.

        omega_low, omega_high = self.freq_band
        n_freqs = 200
        omegas = np.linspace(-omega_high, omega_high, n_freqs)
        d_omega = omegas[1] - omegas[0]

        # Compute cross-covariance at lag 0 for each factor
        sigma_cw = np.zeros(q)
        sigma_ww = np.zeros((q, q))

        for i_omega, omega in enumerate(omegas):
            # Cross-spectrum between target and each factor at frequency omega
            for j in range(q):
                s_cj = self._bivariate_cross_spectrum(c_valid, w_valid[:, j], omega)
                sigma_cw[j] += np.real(s_cj) * d_omega / (2 * np.pi)

            # Cross-spectrum between factors
            for j in range(q):
                for k in range(j + 1):
                    s_jk = self._bivariate_cross_spectrum(w_valid[:, j], w_valid[:, k], omega)
                    sigma_ww[j, k] += np.real(s_jk) * d_omega / (2 * np.pi)
                    if k != j:
                        sigma_ww[k, j] = sigma_ww[j, k]

        # --- Step 3: Projection ---
        # ĉ_t = μ̂ + Σ̂_cw Σ̂_w^{-1} w_t
        sigma_ww_reg = sigma_ww + np.eye(q) * 1e-6  # regularization
        beta = np.linalg.solve(sigma_ww_reg, sigma_cw)
        mu_hat = np.mean(c_valid) - np.dot(beta, np.mean(w_valid, axis=0))

        logger.info(
            "MLRG frequency-domain projection: intercept=%.6f, coefficients=%s",
            mu_hat, np.round(beta, 6),
        )

        # Compute projection for all time steps
        projected = np.full(T_target, np.nan)
        for t in range(min_T):
            if not np.any(np.isnan(w[t])):
                projected[t] = mu_hat + np.dot(beta, w[t])

        return projected

    @staticmethod
    def _monthly_to_quarterly(monthly_data: np.ndarray) -> np.ndarray:
        """Aggregate monthly data to quarterly using (1 + L + L²) filter.

        This applies the squared summation filter used in Eurocoin:
        quarterly value = average of 3 consecutive months, properly weighted.

        The (1 + L + L²)² filter creates a smooth quarterly series from monthly data.
        """
        T_m, q = monthly_data.shape
        # Number of complete quarters
        T_q = T_m // 3

        quarterly = np.zeros((T_q, q))
        for t in range(T_q):
            # Average 3 months in the quarter
            chunk = monthly_data[t * 3:(t + 1) * 3]
            if len(chunk) == 3:
                quarterly[t] = np.mean(chunk, axis=0)
            elif len(chunk) > 0:
                quarterly[t] = np.nanmean(chunk, axis=0)

        return quarterly

    @staticmethod
    def _bivariate_cross_spectrum(
        x: np.ndarray,
        y: np.ndarray,
        omega: float,
        max_lag: int = 12,
    ) -> complex:
        """Compute the cross-spectral density between two series at frequency omega.

        S_xy(omega) = sum_{k=-K}^{K} w(k) * gamma_xy(k) * e^{-i*omega*k}

        where gamma_xy(k) is the cross-covariance at lag k and w(k) is Bartlett kernel.
        """
        T = len(x)
        K = min(max_lag, T - 1)

        # Demean
        x_dm = x - np.mean(x)
        y_dm = y - np.mean(y)

        S = 0.0 + 0.0j
        for lag in range(-K, K + 1):
            weight = 1 - abs(lag) / (K + 1)
            if weight <= 0:
                continue

            # Cross-covariance at lag k
            if lag >= 0:
                if lag == 0:
                    gamma = np.dot(x_dm, y_dm) / T
                else:
                    gamma = np.dot(x_dm[lag:], y_dm[:-lag]) / (T - lag)
            else:
                gamma = np.dot(x_dm[:lag], y_dm[-lag:]) / (T + lag)

            S += weight * gamma * np.exp(-1j * omega * lag)

        return S

    def determine_n_factors(
        self,
        panel: np.ndarray,
        max_factors: int = 10,
    ) -> int:
        """Determine the optimal number of factors using IC criteria.

        Implements Bai-Ng (2002) information criteria for factor number selection.
        """
        T, N = panel.shape
        X = self._standardize(panel)
        cov = (X.T @ X) / T

        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        max_factors = min(max_factors, len(eigenvalues) - 1)

        # Bai-Ng ICp1 criterion
        # IC(k) = ln(V_k) + k * ((N+T)/(NT)) * ln(N*T/(N+T))
        # where V_k = residual variance with k factors
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        V_k = 1 - cumvar  # fraction of unexplained variance

        NT = N * T
        penalty = np.arange(1, max_factors + 1) * ((N + T) / NT) * np.log(NT / (N + T))
        IC = np.log(V_k[:max_factors]) + penalty

        optimal_k = np.argmin(IC) + 1
        logger.info(
            "Bai-Ng IC: optimal q=%d (IC values: %s)",
            optimal_k, np.round(IC[:max_factors], 4),
        )
        return optimal_k
