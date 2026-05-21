"""ClientApp: MedMNIST 3D training using Flower's Message API."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_medmnist3d.task import Net3D, evaluate, load_data, train


app = ClientApp()


def _partition_id(context: Context, num_partitions: int) -> int:
    """Return a bounded partition id for local simulation and SuperNode runs."""
    configured = context.node_config.get("partition-id")
    if configured is not None:
        return int(configured)
    return int(context.node_id) % num_partitions


@app.train()
def train_message(msg: Message, context: Context) -> Message:
    """Train locally and return updated arrays plus weighted metrics."""
    config = msg.content.get("config", {})
    num_partitions = int(context.run_config.get("num-partitions", 2))
    local_epochs = int(config.get("local-epochs", context.run_config.get("local-epochs", 1)))
    lr = float(config.get("learning-rate", context.run_config.get("learning-rate", 0.001)))
    partition_id = _partition_id(context, num_partitions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, _ = load_data(partition_id, num_partitions)
    model = Net3D().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    metrics = train(model, trainloader, device, local_epochs, lr)
    metrics["num-examples"] = len(trainloader.dataset)

    content = RecordDict(
        {
            "arrays": ArrayRecord(torch_state_dict=model.state_dict()),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_message(msg: Message, context: Context) -> Message:
    """Evaluate locally and return weighted metrics."""
    num_partitions = int(context.run_config.get("num-partitions", 2))
    partition_id = _partition_id(context, num_partitions)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, valloader = load_data(partition_id, num_partitions)
    model = Net3D().to(device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    loss, num_examples, metrics = evaluate(model, valloader, device)
    metrics["eval_loss"] = loss
    metrics["num-examples"] = num_examples

    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
