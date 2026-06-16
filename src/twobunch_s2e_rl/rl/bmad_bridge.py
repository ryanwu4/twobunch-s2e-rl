"""slac-rl-side client for the Bmad worker subprocess.

Spawns `rl._bmad_worker` under the bmad-qpad-dev python (the env split: FACET2_S2E only imports
there, torch only here), handshakes, and sends knob settings -> gets back the tracked PENT
drive/witness clouds + survival fractions. The worker's noisy Tao stdout is ignored (we scan
for a sentinel); its stderr is teed to <scratch>/worker.err for debugging.

A dedicated reader THREAD drains the worker's stdout into a queue. This is deliberate: mixing
select() on the pipe fd with buffered readline() is broken -- readline() can pull a chunk that
buffers the sentinel plus following lines into Python's buffer, after which select() sees the fd
empty and blocks forever while the sentinel sits unread. The thread keeps calling readline (which
drains both the OS pipe and Python's buffer) and the main thread polls the queue with a timeout.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np

from ..datagen.paths import repo_root
from ..datagen.sweep_params import PARAM_KEYS
from ._bmad_worker import READY, RESULT

BMAD_PYTHON = "/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python"
_EOF = object()


class BmadBridge:
    def __init__(self, *, baseline_config: str, drive_full_nc: float, witness_full_nc: float,
                 num_macro: int = 20000, csr: bool = True, wakes: bool = True,
                 P: int = 2048, scratch: str | None = None, bmad_python: str = BMAD_PYTHON,
                 ready_timeout: float = 900.0):
        self.scratch = scratch or str(repo_root() / "data" / "rl_bmad_scratch")
        os.makedirs(self.scratch, exist_ok=True)
        self._errlog = open(Path(self.scratch) / "worker.err", "w")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root() / "src")
        cmd = [bmad_python, "-u", "-m", "twobunch_s2e_rl.rl._bmad_worker",
               "--baseline-config", baseline_config,
               "--num-macro", str(int(num_macro)), "--csr", str(int(csr)),
               "--wakes", str(int(wakes)), "--scratch", self.scratch,
               "--drive-full-nc", str(float(drive_full_nc)),
               "--witness-full-nc", str(float(witness_full_nc)), "--P", str(int(P))]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._errlog, text=True, bufsize=1, env=env)
        self._q: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self._scan_for(READY, ready_timeout, "worker init")

    def _pump(self):
        try:
            for line in self.proc.stdout:        # blocking; drains pipe + Python buffer fully
                self._q.put(line)
        finally:
            self._q.put(_EOF)

    def _scan_for(self, prefix: str, timeout: float, what: str) -> str:
        """Pull worker stdout lines from the queue, ignoring Tao noise, until one starts with
        `prefix`. Raises on timeout (no line for `timeout` s) or worker exit (EOF)."""
        while True:
            try:
                line = self._q.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(f"Bmad worker silent >{timeout:.0f}s during {what} "
                                   f"(see {self.scratch}/worker.err)")
            if line is _EOF:
                raise RuntimeError(f"Bmad worker exited during {what} "
                                   f"(see {self.scratch}/worker.err)")
            line = line.strip()
            if line.startswith(prefix):
                return line

    def track(self, knobs_phys, *, P: int | None = None, timeout: float = 3600.0) -> dict:
        """Track one physical 8-knob setting. Returns dict with drive/witness (n,6) np arrays,
        T_drive, T_witness, n_drive, n_witness, ok, error."""
        req = {"knobs": {k: float(knobs_phys[i]) for i, k in enumerate(PARAM_KEYS)}}
        if P is not None:
            req["P"] = int(P)
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        path = self._scan_for(RESULT, timeout, "track")[len(RESULT):]
        data = np.load(path, allow_pickle=False)
        res = {k: data[k] for k in data.files}
        data.close()
        try:
            os.remove(path)
        except OSError:
            pass
        res["ok"] = bool(res["ok"])
        res["error"] = str(res["error"])
        return res

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write("STOP\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
        except Exception:                        # noqa: BLE001
            self.proc.kill()
        finally:
            self._errlog.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
