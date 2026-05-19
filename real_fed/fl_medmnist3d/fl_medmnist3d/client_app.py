"""ClientApp: MedMNIST 3D training on each SuperNode."""

from __future__ import annotations

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fl_medmnist3d.task import (
    Net3D,
    evaluate,
    get_parameters,
    load_data,
    set_parameters,
    train,
)


class MedMNISTClient(NumPyClient):
    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        local_epochs: int,
        lr: float,
    ) -> None:
        self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs = local_epochs
        self.lr           = lr
        self.trainloader, self.valloader = load_data(partition_id, num_partitions)
        self.model = Net3D().to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        metrics = train(self.model, self.trainloader, self.device, self.local_epochs, self.lr)
        return get_parameters(self.model), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, num_examples, metrics = evaluate(self.model, self.valloader, self.device)
        return loss, num_examples, metrics


def client_fn(context: Context):
    num_partitions = int(context.run_config.get("num-partitions", 2))
    local_epochs   = int(context.run_config.get("local-epochs", 1))
    lr             = float(context.run_config.get("learning-rate", 0.001))
    # node_id is a large integer assigned by SuperLink; modulo keeps it bounded.
    partition_id   = int(context.node_id) % num_partitions

    return MedMNISTClient(partition_id, num_partitions, local_epochs, lr).to_client()


app = ClientApp(client_fn=client_fn)
