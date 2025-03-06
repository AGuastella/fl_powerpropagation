"""FedAvg strategy enhanced with weight dynamics tracking capabilities.

This strategy extends the basic FedAvg implementation to track and analyze various
aspects of weight dynamics during training, including weight movement analysis and
client update similarities.
"""

from logging import WARNING
import logging
from pathlib import Path
from typing import Optional, Union, Callable
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from project.fed.utils.weight_dynamics_utils import WeightDynamicsTracker


class FedAvgDynamics(Strategy):
    """Federated Averaging with weight dynamics tracking capabilities."""

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weight_dynamics_logging: bool = True,
        working_dir: Path,
    ) -> None:
        """Initialize the strategy.

        Args:
            fraction_fit: Fraction of clients used during training
            fraction_evaluate: Fraction of clients used during validation
            min_fit_clients: Minimum number of clients used during training
            min_evaluate_clients: Minimum number of clients used during validation
            min_available_clients: Minimum number of total clients in the system
            evaluate_fn: Optional function used for validation
            on_fit_config_fn: Function used to configure training
            on_evaluate_config_fn: Function used to configure validation
            accept_failures: Whether to accept rounds containing failures
            initial_parameters: Initial global model parameters
            fit_metrics_aggregation_fn: Metrics aggregation function
            evaluate_metrics_aggregation_fn: Metrics aggregation function
            weight_dynamics_logging: Whether to track weight dynamics
            working_dir: Directory for saving results and artifacts
        """
        super().__init__()

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.weight_dynamics_logging = weight_dynamics_logging
        self.working_dir = working_dir

        # Initialize weight dynamics tracker
        self.weight_tracker = WeightDynamicsTracker()

    def __repr__(self) -> str:
        """Compute string representation of the strategy."""
        return f"FedAvgDynamics(accept_failures={self.accept_failures})"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters.

        This method is called once at the beginning of training to set the initial
        global model parameters. It also initializes the weight dynamics tracker with
        these parameters if weight dynamics logging is enabled.
        """
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory

        if initial_parameters is not None and self.weight_dynamics_logging:
            log(
                logging.INFO,
                "Initializing weight dynamics tracker with initial parameters",
            )
            self.weight_tracker.update_initial_weights(
                parameters_to_ndarrays(initial_parameters)
            )

        return initial_parameters

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average.

        This method extends the basic FedAvg aggregation by computing and logging weight
        dynamics metrics during the aggregation process.
        """
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Save pre-aggregation weights for dynamics analysis
        # pre_aggregation_weights = parameters_to_ndarrays(results[0][1].parameters)

        # Extract client updates for weight dynamics tracking
        if self.weight_dynamics_logging:
            client_updates = [weights for weights, _ in weights_results]

        # Perform weighted aggregation of weights
        weights_prime = self.aggregate_weights(weights_results)
        parameters_aggregated = ndarrays_to_parameters(weights_prime)

        # Compute weight dynamics metrics
        metrics_aggregated = {}
        if self.weight_dynamics_logging:
            dynamics_metrics = self.weight_tracker.compute_round_metrics(
                weights_prime,
                client_updates,
            )
            metrics_aggregated.update(dynamics_metrics)

        # Aggregate custom metrics if aggregation fn was provided
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            custom_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics_aggregated.update({k: float(v) for k, v in custom_metrics.items()})

        return parameters_aggregated, {
            k: float(v) for k, v in metrics_aggregated.items()
        }

    def aggregate_weights(self, results: list[tuple[NDArrays, int]]) -> NDArrays:
        """Compute weighted average of weights.

        Args:
            results: List of tuples containing weights and number of examples

        Returns:
            Aggregated weights
        """
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for _, num_examples in results)

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights]
            for weights, num_examples in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            np.sum([w[i] for w in weighted_weights], axis=0) / num_examples_total
            for i in range(len(weighted_weights[0]))
        ]

        return weights_prime

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function.

        This method extends the basic evaluation by including weight dynamics metrics in
        the evaluation results.
        """
        if self.evaluate_fn is None:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})

        if eval_res is None:
            return None

        loss, metrics = eval_res

        # Add weight dynamics metrics to evaluation metrics
        # if self.weight_dynamics_logging:
        #     dynamics_metrics = self.weight_tracker.compute_round_metrics(
        #         parameters_ndarrays,  # Use current parameters as round start
        #         parameters_ndarrays,  # And end weights since we're just evaluating
        #         [],  # No client updates during evaluation
        #     )
        #     metrics.update(dynamics_metrics)

        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        self.weight_tracker.update_round_start_weights(
            parameters_to_ndarrays(parameters)
        )

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Don't configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return sample size and required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return sample size and required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)

        return loss_aggregated, metrics_aggregated
