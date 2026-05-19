# FL\_dist\_demo — Distributed Federated Learning with Flower

A minimal, runnable demo that launches a Flower federation across multiple machines and orchestrates it from a Jupyter notebook.

\---

## Architecture

```
HOST\_SuperLink  (e.g. 192.168.1.10)
  flower-superlink
    port 9092  ← SuperNodes connect here
    port 9093  ← flwr run / Orchestrator connects here

HOST\_SuperNode — SuperNode 0  (e.g. 192.168.1.11)
  flower-supernode  --superlink 192.168.1.10:9092

HOST\_SuperNode — SuperNode 1  (e.g. 192.168.1.11, different port)
  flower-supernode  --superlink 192.168.1.10:9092

Orchestrator  (Jupyter notebook, can run on HOST\_SuperLink)
  flwr run fl\_app/ distributed   →   sends FAB to SuperLink
                                      SuperLink distributes it to SuperNodes
                                      training + aggregation happen
                                      logs stream back to notebook
```

\---

## Environment setup (every machine, once)

Do this on **every** machine that will participate (HOST\_SuperLink, HOST\_SuperNode, Orchestrator).

### Variables

```bash
ENV\_NAME="flower\_ai"
```

### a) Create the conda environment

```bash
conda create -n $ENV\_NAME python=3.10 -y
```

### b) Activate it

```bash
conda activate $ENV\_NAME
```

You should see `(flower\_ai)` in your prompt. All subsequent commands must be run inside this environment.

### c) Install the required packages

**Flower** (includes `flower-superlink`, `flower-supernode` and all FL dependencies):

```bash
pip install "flwr\[simulation]>=1.15,<2.0"
```

**PyTorch — CPU only** (lighter, sufficient for all exercises in this demo):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**MedMNIST** (pin the exact version to ensure compatibility):

```bash
pip install medmnist==3.0.2
```

These three packages are all you need to run every exercise in this demo.

\---

## Register the Jupyter kernel (HOST\_SuperLink / Orchestrator machine only)

The Jupyter server should be launched **outside** the conda environment (from your base or system environment), while the `flower\_ai` environment is registered as a selectable kernel inside the notebook.

### Step 1 — Register the kernel (inside the activated environment)

```bash
conda activate $ENV\_NAME
pip install ipykernel
python -m ipykernel install --user --name $ENV\_NAME --display-name "Python (flower\_ai)"
```

### Step 2 — Launch Jupyter from outside the environment

Deactivate first, then start Jupyter from your base environment:

```bash
conda deactivate
jupyter notebook
```

> This way the Jupyter server is not tied to `flower\_ai`, but the notebook can still
> use it by selecting the registered kernel.

### Step 3 — Select the kernel in the notebook

Open the notebook (`Orchestrator.ipynb` or `Orchestrator\_MedMNIST3D.ipynb`) and select
**Python (flower\_ai)** from the top-right kernel menu (or *Kernel → Change kernel*).

\---

## Step-by-step

### 1 — HOST\_SuperLink machine

```bash
conda activate $ENV\_NAME
bash start\_superlink.sh
```

Leave this terminal open.

### 2 — HOST\_SuperNode machine

```bash
conda activate $ENV\_NAME
bash start\_supernode.sh <SUPERLINK\_IP>
```

If you run two SuperNodes on the **same** machine, use different ClientApp I/O ports:

```bash
bash start\_supernode.sh <SUPERLINK\_IP>       # SuperNode 0 — default port 9094
bash start\_supernode.sh <SUPERLINK\_IP> 9096  # SuperNode 1 — port 9096
```

### 3 — Orchestrator (Jupyter)

Launch Jupyter from outside `flower\_ai` (see above), open the desired notebook,
select the **Python (flower\_ai)** kernel, set `SUPERLINK\_IP`, and run all cells top to bottom.

Ports **9092** and **9093** on HOST\_SuperLink must be reachable from HOST\_SuperNode and the Orchestrator. Open them in your firewall if needed.

\---

## File layout

```
FL\_dist\_demo/
├── README.md
├── start\_superlink.sh              # Run on HOST\_SuperLink
├── start\_supernode.sh              # Run on HOST\_SuperNode
├── Orchestrator.ipynb              # Logistic regression demo
├── Orchestrator\_MNIST.ipynb        # MNIST demo
├── fl\_app/                         # Flower App — logistic regression
│   ├── pyproject.toml
│   └── fl\_app/
│       ├── task.py
│       ├── client\_app.py
│       └── server\_app.py
└── real\_fed/                       # MedMNIST 3D federated experiment
    ├── Orchestrator\_MedMNIST3D.ipynb
    └── fl\_medmnist3d/
        ├── pyproject.toml
        └── fl\_medmnist3d/
            ├── task.py             # Net3D model + dataset + train/evaluate
            ├── client\_app.py       # ClientApp — runs on each SuperNode
            └── server\_app.py       # ServerApp — FedAvg aggregation
```

\---

## Changing the number of rounds or clients

Edit `pyproject.toml` in the relevant app directory:

```toml
\[tool.flwr.app.config]
num-server-rounds = 5
min-clients = 2
```

Or pass overrides at run time from the notebook:

```python
"--run-config", "num-server-rounds=10",
"--run-config", "min-clients=3",
```

\---

## Troubleshooting

|Symptom|Likely cause|
|-|-|
|`flwr run` hangs indefinitely|Not enough SuperNodes connected; check `MIN\_CLIENTS`|
|`Connection refused` on port 9093|SuperLink not started or firewall blocking it|
|`ModuleNotFoundError` on SuperNode|`torch` or `medmnist` not installed in the conda env on that machine|
|Different Flower versions on machines|Install the same `flwr` version everywhere|
|WSL2: SuperNodes on external machine cannot connect|Set up Windows port forwarding (`netsh portproxy`) or install Tailscale inside WSL2|



