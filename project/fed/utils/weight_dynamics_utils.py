"""Utility functions for tracking and analyzing weight dynamics in FL."""

from typing import NamedTuple
import numpy as np
from flwr.common import NDArrays
from flwr.common.logger import log
import logging


class ClientSimilarityStats(NamedTuple):
    """Container for detailed client similarity statistics.

    Attributes
    ----------
        mean: Average similarity across all client pairs
        min_val: Minimum similarity between any pair of clients
        max_val: Maximum similarity between any pair of clients
        std: Standard deviation of similarities
        client_pairs: List of tuples containing (client_i, client_j, similarity)
                     to track which clients were most/least similar
    """

    mean: float
    min_val: float
    max_val: float
    std: float
    client_pairs: list[tuple[int, int, float]]


class WeightDynamicsTracker:
    """Tracks and analyzes weight movements during federated learning.

    This class maintains historical data about weight changes and provides methods to
    compute various metrics about model evolution during training.
    """

    def __init__(self) -> None:
        """Initialize the weight dynamics tracker."""
        # Store initial weights from start of training
        self.initial_weights: NDArrays | None = None

        # Track history of metrics for trend analysis
        self.round_history: list[dict] = []

        # Track historical client similarities for pattern analysis
        self.similarity_history: list[ClientSimilarityStats] = []

    def update_initial_weights(self, weights: NDArrays) -> None:
        """Store initial weights as reference for tracking full training evolution.

        Args:
            weights: Initial model weights to use as baseline
        """
        log(logging.INFO, "Storing initial weights for tracking")
        self.initial_weights = [np.copy(w) for w in weights]

    def update_round_start_weights(self, weights: NDArrays) -> None:
        """Store model weights at start of current training round.

        Args:
            weights: Model weights at start of current round
        """
        log(logging.INFO, "Storing weights at start of current round")
        self.round_start_weights = [np.copy(w) for w in weights]

    def compute_l2_distance(self, weights_a: NDArrays, weights_b: NDArrays) -> float:
        """Compute L2 distance between two sets of weights.

        Args:
            weights_a: First set of weights
            weights_b: Second set of weights

        Returns
        -------
            Average L2 distance across all layers
        """
        l2_distances = []
        for w_a, w_b in zip(weights_a, weights_b, strict=True):
            if w_a.size > 0:  # Skip empty layers
                l2_dist = np.linalg.norm(w_b - w_a)
                l2_distances.append(l2_dist)

        if not l2_distances:
            return 0.0

        return float(np.mean(l2_distances))

    def compute_cosine_similarity(
        self, weights_a: NDArrays, weights_b: NDArrays
    ) -> float:
        """Compute cosine similarity between two sets of weights.

        Args:
            weights_a: First set of weights
            weights_b: Second set of weights

        Returns
        -------
            Average cosine similarity across all layers
        """
        similarities = []
        for w_a, w_b in zip(weights_a, weights_b, strict=True):
            if w_a.size > 0:  # Skip empty layers
                # Flatten arrays for cosine similarity
                flat_a = w_a.flatten()
                flat_b = w_b.flatten()

                # Compute norms
                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)

                if norm_a > 0 and norm_b > 0:
                    sim = np.dot(flat_a, flat_b) / (norm_a * norm_b)
                    similarities.append(sim)

        if not similarities:
            return 0.0

        return float(np.mean(similarities))

    def compute_pairwise_client_similarities(
        self,
        client_updates: list[NDArrays],
    ) -> ClientSimilarityStats:
        """Compute pairwise cosine similarities between all client updates.

        Args:
            client_updates: List of weight updates from each client

        Returns
        -------
            Statistics about client similarities including mean, min_val, max_val, std
        """
        if len(client_updates) <= 1:
            return ClientSimilarityStats(0.0, 0.0, 0.0, 0.0, [])

        similarities = []
        client_pairs = []

        # Compute pairwise similarities between all clients
        for i in range(len(client_updates)):
            for j in range(i + 1, len(client_updates)):
                sim = self.compute_cosine_similarity(
                    client_updates[i], client_updates[j]
                )
                similarities.append(sim)
                client_pairs.append((i, j, sim))

        if not similarities:
            return ClientSimilarityStats(0.0, 0.0, 0.0, 0.0, [])

        similarities = list(np.array(similarities))

        # Get indices of min_val and max_val similarities for logging
        min_idx = np.argmin(similarities)
        max_idx = np.argmax(similarities)
        min_pair = client_pairs[min_idx]
        max_pair = client_pairs[max_idx]

        # Log most and least similar client pairs
        log(
            logging.INFO,
            f"Client similarity stats - Mean: {np.mean(similarities):.4f}, "
            f"Min: {np.min(similarities):.4f} (clients {min_pair[0]},{min_pair[1]}), "
            f"Max: {np.max(similarities):.4f} (clients {max_pair[0]},{max_pair[1]}), "
            f"Std: {np.std(similarities):.4f}",
        )

        return ClientSimilarityStats(
            mean=float(np.mean(similarities)),
            min_val=float(np.min(similarities)),
            max_val=float(np.max(similarities)),
            std=float(np.std(similarities)),
            client_pairs=client_pairs,
        )

    def compute_round_metrics(
        self,
        current_weights: NDArrays,
        client_updates: list[NDArrays],
    ) -> dict[str, float]:
        """Compute comprehensive metrics for current training round.

        Args:
            round_start_weights: Model weights at start of current round
            current_weights: Current global model weights after aggregation
            client_updates: List of weight updates from each client

        Returns
        -------
            Dictionary containing all computed metrics for the round
        """
        metrics = {}

        # Compute L2 distance from training start (if we have initial weights)
        if self.initial_weights is not None:
            metrics["global_l2_distance"] = self.compute_l2_distance(
                self.initial_weights, current_weights
            )

        # Compute L2 distance for current round
        metrics["round_l2_distance"] = self.compute_l2_distance(
            self.round_start_weights, current_weights
        )

        # Compute cosine similarity between round start and end
        metrics["round_cosine_similarity"] = self.compute_cosine_similarity(
            self.round_start_weights, current_weights
        )

        # Get client similarity metrics if we have updates
        if client_updates:
            similarity_stats = self.compute_pairwise_client_similarities(client_updates)
            self.similarity_history.append(similarity_stats)

            similarity_metrics = {
                "client_similarity_mean": similarity_stats.mean,
                "client_similarity_min": similarity_stats.min_val,
                "client_similarity_max": similarity_stats.max_val,
                "client_similarity_std": similarity_stats.std,
            }
            metrics.update(similarity_metrics)

        # Store metrics for history
        self.round_history.append(metrics)

        return metrics

    def get_historical_trends(self) -> dict[str, list[float]]:
        """Get historical trends of tracked metrics.

        Returns
        -------
            Dictionary mapping metric names to their historical values
        """
        trends: dict[str, list[float]] = {}

        if not self.round_history:
            return trends

        # Extract each metric's history
        metric_names = self.round_history[0].keys()
        for metric in metric_names:
            trends[metric] = [
                round_metrics.get(metric, 0.0) for round_metrics in self.round_history
            ]

        return trends
