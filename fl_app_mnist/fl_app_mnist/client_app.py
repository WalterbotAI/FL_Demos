"""ClientApp: CNN trained on a local MNIST partition."""

from __future__ import annotations

import torch

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from fl_app_mnist.task import Net, load_data, get_parameters, set_parameters, train, evaluate


class _MNISTClient(NumPyClient):
    def __init__(self, partition_id: int, num_partitions: int, max_samples: int = 2000) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.train_loader, self.test_loader = load_data(partition_id, num_partitions, max_samples)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        return train(self.model, self.train_loader, epochs=1, device=self.device)

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        return evaluate(self.model, self.test_loader, device=self.device)


def client_fn(context: Context):
    num_partitions = int(context.run_config.get("num-partitions", 2))
    max_samples = int(context.run_config.get("max-samples", 2000))
    partition_id = int(context.node_id) % num_partitions
    return _MNISTClient(partition_id, num_partitions, max_samples).to_client()


app = ClientApp(client_fn=client_fn)
