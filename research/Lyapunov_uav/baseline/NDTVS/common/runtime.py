"""Random state, device requirement and walltime/signal budget."""
from __future__ import annotations

import random
import signal
import time
import numpy as np
import torch


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2].cpu())
    if state[3] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state[3]])


class Budget:
    def __init__(self, seconds, reserve):
        self.started, self.seconds, self.reserve = time.monotonic(), seconds, reserve
        self.reason = ""
        self.old = {}
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            self.old[sig] = signal.signal(sig, self.stop)

    def stop(self, signum, _frame):
        self.reason = signal.Signals(signum).name

    def expired(self, estimate=0):
        if self.reason:
            return True
        if time.monotonic() - self.started + estimate + self.reserve >= self.seconds:
            self.reason = "walltime budget"
            return True
        return False

    def close(self):
        for sig, handler in self.old.items():
            signal.signal(sig, handler)


def require_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
