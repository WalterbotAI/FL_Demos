"""ServerApp: FedAvg aggregation for MedMNIST 3D."""

from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fl_medmnist3d.task import Net3D, get_parameters


def server_fn(context: Context) -> ServerAppComponents:
    num_rounds  = int(context.run_config.get("num-server-rounds", 3))
    min_clients = int(context.run_config.get("min-clients", 2))

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(get_parameters(Net3D())),
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
