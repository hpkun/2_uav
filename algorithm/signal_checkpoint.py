"""Signal requests are serviced only at explicit safe training boundaries.

SIGKILL cannot be caught. No RNG is consumed by this module.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


class SignalCheckpoint:
    def __init__(self):
        self.request = None
        self.previous = {}

    def handle(self, number, frame):
        # Do not perform I/O, serialization or optimizer work in a handler.
        if self.request is None:
            self.request = {"signal_number": number,
                            "signal_name": signal.Signals(number).name,
                            "requested_at": now()}

    def install(self):
        for name in ("SIGTERM", "SIGINT", "SIGHUP"):
            number = getattr(signal, name, None)
            if number is not None:
                self.previous[number] = signal.signal(number, self.handle)

    def restore(self):
        for number, handler in self.previous.items():
            signal.signal(number, handler)

    def save_at_boundary(self, trainer, run_dir, log):
        if self.request is None:
            return
        run_dir = Path(run_dir)
        checkpoint = run_dir / f"checkpoint_emergency_{trainer.env_steps}.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        try:
            trainer.save_checkpoint(temporary)
            with temporary.open("r+b") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, checkpoint)
        finally:
            if temporary.exists():
                temporary.unlink()
        metadata = {**self.request, "saved_at": now(),
                    "sampled_steps": trainer.env_steps, "run_name": run_dir.name,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "termination_reason": "external_signal"}
        target = run_dir / "termination.json"
        temp = target.with_suffix(".json.tmp")
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
        log(f"[TERMINATED_BY_SIGNAL] {metadata['signal_name']} | steps {trainer.env_steps} | {checkpoint.name}")
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(128 + int(self.request["signal_number"]))
