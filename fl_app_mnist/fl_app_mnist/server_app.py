"""ServerApp: FedAvg over CNN parameters."""

from __future__ import annotations

import numpy as np

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fl_app_mnist.task import Net, get_parameters


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    min_clients = int(context.run_config.get("min-clients", 2))

    # Initialise global model with random weights
    initial_params = ndarrays_to_parameters(get_parameters(Net()))

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_params,
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
