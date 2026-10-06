"""Sequential seed1 launcher with explicit child exit/signal reporting."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def exit_status(code):
    if code == 0:
        return "COMPLETE"
    if code < 0 or code in (129, 130, 137, 143):
        return "TERMINATED_BY_SIGNAL"
    return "FAILED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback forbidden")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    active = [None]
    requested = [None]

    def forward(number, frame):
        requested[0] = number
        child = active[0]
        if child is not None and child.poll() is None:
            child.send_signal(number)

    for name in ("SIGTERM", "SIGINT", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, forward)
    events = []
    event_path = logs / f"happo_tacm_seed1_{args.timestamp}.status.json"
    for method, entry, config in (
        ("happo", "algorithm/train_happo.py", "configs/happo_v310_ablation.yaml"),
        ("tacm", "algorithm/train_tacm_rgaa.py", "configs/happo_tacm_rgaa_v310.yaml"),
    ):
        if requested[0] is not None:
            raise SystemExit(128 + requested[0])
        name = f"{method}_v310_main_seed1_2m_{args.timestamp}"
        if (ROOT / "outputs" / name).exists():
            raise FileExistsError(name)
        command = [sys.executable, "-u", entry, "--steps", "2000000", "--profile", "main",
                   "--seed", "1", "--device", "cuda", "--num-envs", "16", "--config", config,
                   "--env-config", "configs/env_v310.yaml", "--output-name", name,
                   "--checkpoint-interval", "500000", "--eval-interval", "0", "--log-interval", "50000",
                   "--final-eval-episodes", "200", "--eval-action-mode", "stochastic", "--eval-action-seed", "2000"]
        print(f"[START] {name}", flush=True)
        with (logs / f"{name}.log").open("x", encoding="utf-8") as stream:
            active[0] = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=os.name == "posix")
            code = active[0].wait()
        events.append({"run": name, "child_exit_code": code, "status": exit_status(code),
                       "ended_at": datetime.now().astimezone().isoformat()})
        event_path.write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
        print(f"[{exit_status(code)}] {name} child_exit_code={code}", flush=True)
        active[0] = None
        if code or requested[0] is not None:
            raise SystemExit((128 - code if code < 0 else code) or 128 + requested[0])


if __name__ == "__main__":
    main()
