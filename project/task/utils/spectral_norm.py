"""Spectral norm handler for computing the spectral normalization of weight tensors."""

import torch
import torch.nn.functional as F


class SpectralNormHandler:
    def __init__(self, epsilon: float = 1e-12, num_iterations: int = 1):
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.exponent_range = 1.0
        self.cached_exponents: dict = {}
        self.sigma = 0.0

    def _compute_spectral_norm(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute the spectral normalization for a given weight tensor."""
        weight_detach = weight.detach()
        weight_abs = weight_detach.abs()
        weight_mat = weight_abs.view(weight_abs.size(0), -1)

        u = torch.randn(weight_mat.size(0), 1, device=weight.device)
        v = torch.randn(weight_mat.size(1), 1, device=weight.device)

        for _ in range(self.num_iterations):
            v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0)
            u = F.normalize(torch.matmul(weight_mat, v), dim=0)

        sigma = torch.matmul(u.t(), torch.matmul(weight_mat, v))
        self.sigma = torch.clamp(sigma, min=self.epsilon)

        return weight_abs / self.sigma

    def _get_cache_key(self, tensor: torch.Tensor) -> str:
        """Generate a unique key for the tensor based on its properties."""
        return f"{tensor.shape}_{tensor.device}_{tensor.dtype}"

    def compute_weight_update(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute the updated weight with cached exponents."""
        cache_key = self._get_cache_key(weight)

        # Get or compute the exponent
        if cache_key not in self.cached_exponents:
            # Compute the normalized weight
            weight_normalized = self._compute_spectral_norm(weight)

            # comute the weight_normalized average value
            weight_normalized_avg = torch.mean(weight_normalized)
            # compute the average value of the non zero weight
            # weight_normalized_avg = torch.mean(
            #     weight_normalized[weight_normalized != 0]
            # )

            # Compute the exponent
            # exponent = 1 + (self.exponent_range * weight_normalized.view_as(weight))
            # exponent = torch.clamp(exponent, max=10)  # Prevent overflow
            # exponent = 1 + weight_normalized_avg

            # print(f"Weight normalized average: {weight_normalized_avg}")

            # add a check if exponent is nan
            if torch.isnan(weight_normalized_avg):
                # print(f"Exponent is nan. Number of non zero values: {torch.sum(weight_normalized != 0)}")
                exponent = 1.0
            else:
                weight_normalized_avg = round(weight_normalized_avg.item(), 8)
                exponent = 1.0 + weight_normalized_avg

            print(f"Exponent: {exponent}")

            # Save the exponent in cache
            self.cached_exponents[cache_key] = exponent

        # Use cached exponent
        exponent = self.cached_exponents[cache_key]

        # Compute final weight update
        sign_weight = torch.sign(weight)
        weight_abs = weight.abs()
        weight_updated = sign_weight * torch.pow(weight_abs, exponent)
        # normalize the weight to the original sigma
        # weight_updated = weight_updated * (
        #     weight_updated / torch.pow(self.sigma, 1 + self.exponent_range)
        # )

        return weight_updated

    def clear_cache(self):
        """Clear the cached exponents."""
        self.cached_exponents = {}
