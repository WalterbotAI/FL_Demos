"""ServerApp: Message API FedAvg aggregation for MedMNIST 3D."""

from __future__ import annotations

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_medmnist3d.task import Net3D


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedAvg over Message API ClientApps."""
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    min_clients = int(context.run_config.get("min-clients", 2))
    local_epochs = int(context.run_config.get("local-epochs", 1))
    lr = float(context.run_config.get("learning-rate", 0.001))

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=min_clients,
        min_evaluate_nodes=min_clients,
        min_available_nodes=min_clients,
    )

    initial_arrays = ArrayRecord(torch_state_dict=Net3D().state_dict())
    train_config = ConfigRecord({"local-epochs": local_epochs, "learning-rate": lr})

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=train_config,
    )
    print(result)
