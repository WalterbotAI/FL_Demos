"""ClientApp: logistic regression trained on each SuperNode."""

from __future__ import annotations

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from fl_app.task import load_data, train, evaluate


class _LogRegClient(NumPyClient):
    def __init__(self, partition_id: int) -> None:
        self.X_train, self.y_train, self.X_test, self.y_test = load_data(partition_id)

    def get_parameters(self, config):
        # Parameters are initialised by the server; this is only called if the
        # server strategy requests them (not the case with FedAvg + initial_params).
        from fl_app.task import get_initial_parameters
        return get_initial_parameters()

    def fit(self, parameters, config):
        updated_params, n, metrics = train(parameters, self.X_train, self.y_train)
        return updated_params, n, metrics

    def evaluate(self, parameters, config):
        loss, n, metrics = evaluate(parameters, self.X_test, self.y_test)
        return loss, n, metrics


def client_fn(context: Context):
    # In local-sim: node_id is 0 … (num_supernodes-1).
    # In distributed: node_id is a large integer assigned by SuperLink;
    #   we use modulo to keep the partition seeds bounded.
    partition_id = int(context.node_id) % 100
    return _LogRegClient(partition_id).to_client()


app = ClientApp(client_fn=client_fn)
