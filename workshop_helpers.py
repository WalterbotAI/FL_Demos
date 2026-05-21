"""Small Jupyter helpers for the Flower workshop notebooks."""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


@dataclass(frozen=True)
class WorkshopContext:
    root: Path
    app_dir: Path
    flwr_bin: Path


def preflight(app_dir: str = "fl_app") -> WorkshopContext:
    """Print the active notebook environment and return paths used by demos."""
    root = Path.cwd()
    context = WorkshopContext(
        root=root,
        app_dir=root / app_dir,
        flwr_bin=Path(sys.executable).parent / "flwr",
    )

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"Notebook cwd      : {context.root}")
    print(f"Flower app        : {context.app_dir}")
    print(f"flwr binary       : {context.flwr_bin}")

    for package in ["flwr", "numpy"]:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"{package:16}: MISSING")
            continue
        module = __import__(package)
        print(
            f"{package:16}: {getattr(module, '__version__', 'installed')} "
            f"@ {Path(spec.origin).parent}"
        )

    if not context.app_dir.exists():
        raise FileNotFoundError("Run this notebook from the FL_Demos repo root.")
    if not context.flwr_bin.exists():
        raise FileNotFoundError("flwr is not installed in the active notebook kernel.")

    return context


def bin_dir(context: WorkshopContext) -> str:
    """Return the active kernel's executable directory with a trailing slash."""
    return str(context.flwr_bin.parent) + "/"


def check_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def print_port_status(host: str, ports: dict[str, int]) -> bool:
    """Print reachability for named TCP ports and return True if all are open."""
    statuses = {}
    for label, port in ports.items():
        statuses[label] = check_port_open(host, port)
        marker = "reachable" if statuses[label] else "not reachable"
        print(f"{label:12} {host}:{port}  -> {marker}")
    return all(statuses.values())


def update_superlink_address(address: str, name: str = "distributed") -> None:
    """Write a SuperLink connection address into ~/.flwr/config.toml."""
    try:
        import tomli_w
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install tomli-w to update Flower config.") from exc

    flwr_config = Path.home() / ".flwr" / "config.toml"
    flwr_config.parent.mkdir(parents=True, exist_ok=True)

    config = {}
    if flwr_config.exists():
        with flwr_config.open("rb") as f:
            config = tomllib.load(f)

    config.setdefault("superlink", {}).setdefault(name, {})["address"] = address
    config["superlink"][name]["insecure"] = True

    with flwr_config.open("wb") as f:
        tomli_w.dump(config, f)
    print(f"Updated ~/.flwr/config.toml: [superlink.{name}] address = {address}")


def flower_connections() -> dict:
    """Return SuperLink connections from the user's Flower config."""
    config_path = Path.home() / ".flwr" / "config.toml"
    if not config_path.exists():
        return {}
    with config_path.open("rb") as f:
        return tomllib.load(f).get("superlink", {})


def choose_local_simulation_connection() -> str:
    """Choose the local simulation connection across common Flower versions."""
    connections = flower_connections()
    for name in ["local-simulation", "local-sim", "local"]:
        if name in connections:
            return name
    raise RuntimeError(
        "No local Flower simulation connection found. Run `flwr federation list` "
        "or recreate the default Flower config."
    )


def print_flower_connections() -> str:
    """Print available Flower connections and return the chosen local simulation."""
    connections = flower_connections()
    print("Flower connections:")
    for name in connections:
        if name != "default":
            print(f"- {name}")

    local_sim = choose_local_simulation_connection()
    print(f"\nUsing local simulation connection: {local_sim}")
    return local_sim


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _is_warning_line(line: str) -> bool:
    plain = ANSI_RE.sub("", line)
    return (
        "WARNING" in plain
        or "FutureWarning" in plain
        or "warnings.warn(" in plain
        or "Ray deduplicates logs" in plain
    )


def run_stream(
    cmd: list[str] | str,
    cwd: Path | None = None,
    *,
    suppress_warnings: bool = True,
) -> int:
    """Run a command and stream stdout/stderr into the notebook output."""
    shell_cmd = cmd.replace("\\\n", " ") if isinstance(cmd, str) else cmd
    argv = shlex.split(shell_cmd) if isinstance(shell_cmd, str) else shell_cmd
    if isinstance(cmd, str) and "\n" in cmd:
        print("$ " + cmd.replace("\n", "\n  "), flush=True)
    else:
        print("$ " + " ".join(argv), flush=True)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore",
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
    }
    with subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            if suppress_warnings and _is_warning_line(line):
                continue
            print(line, end="", flush=True)
        return proc.wait()


def start_background(
    cmd: list[str] | str,
    *,
    label: str,
    cwd: Path | None = None,
) -> subprocess.Popen:
    """Start a background process and print its command like a terminal run."""
    shell_cmd = cmd.replace("\\\n", " ") if isinstance(cmd, str) else cmd
    argv = shlex.split(shell_cmd) if isinstance(shell_cmd, str) else shell_cmd
    if isinstance(cmd, str) and "\n" in cmd:
        print("$ " + cmd.replace("\n", "\n  "), flush=True)
    else:
        print("$ " + " ".join(argv), flush=True)
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Started {label} (pid {proc.pid})")
    return proc


def stop_background(procs: list[subprocess.Popen]) -> None:
    """Terminate background processes started from a notebook cell."""
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    procs.clear()
    print("All background processes stopped.")


def run_local_simulation(
    context: WorkshopContext,
    *,
    num_rounds: int = 1,
    min_clients: int = 1,
    num_supernodes: int = 1,
    connection: str | None = None,
) -> None:
    """Run a small Flower simulation from a notebook cell."""
    local_sim = connection or choose_local_simulation_connection()
    rc = run_stream(
        [
            str(context.flwr_bin),
            "run",
            str(context.app_dir),
            local_sim,
            "--run-config",
            f"num-server-rounds={num_rounds}",
            "--run-config",
            f"min-clients={min_clients}",
            "--federation-config",
            f"num-supernodes={num_supernodes}",
            "--stream",
        ]
    )

    if rc != 0:
        raise RuntimeError(f"Local simulation failed with exit code {rc}")
    print("\nLocal simulation completed successfully.")
