import numpy as np

class Metrics:
    """Collection of coefficient-based metrics (entropy, energy compaction, etc.)."""

    def __init__(self, coeffs: np.ndarray):
        self.coeffs = coeffs

    # ------------------------
    # Shannon entropy
    # ------------------------
    def entropy(self, bins: int = 64) -> float:
        """Compute Shannon entropy of quantized coefficients."""
        quantized = np.digitize(self.coeffs, bins=np.linspace(np.min(self.coeffs), np.max(self.coeffs), bins))
        hist, _ = np.histogram(quantized, bins=bins, density=True)
        hist = hist[hist > 0]  # avoid log(0)
        return float(-np.sum(hist * np.log2(hist)))

    # ------------------------
    # Energy compaction
    # ------------------------
    def energy_compaction(self, k: int | None = None) -> float:
        """Fraction of energy in top-k coefficients by magnitude."""
        energy_total = float(np.sum(self.coeffs**2))
        if energy_total == 0:
            return 0.0
        coeffs_sorted = np.sort(self.coeffs**2)[::-1]
        if k is None:
            k = coeffs_sorted.shape[0] // 4  # top 25%
        return float(np.sum(coeffs_sorted[:k]) / energy_total)

    # ------------------------
    # Additional metrics placeholder
    # ------------------------
    def l1_norm(self) -> float:
        """L1 norm of the coefficients."""
        return float(np.sum(np.abs(self.coeffs)))

    def l2_norm(self) -> float:
        """L2 norm of the coefficients."""
        return float(np.sqrt(np.sum(self.coeffs**2)))

    def sparsity_ratio(self, threshold: float = 1e-5) -> float:
        """Fraction of coefficients below a threshold (sparsity)."""
        total = self.coeffs.size
        if total == 0:
            return 0.0
        sparse_count = np.sum(np.abs(self.coeffs) < threshold)
        return float(sparse_count / total)

