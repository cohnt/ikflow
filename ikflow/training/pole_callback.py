"""Pole-fraction validation metric.

Measures the fraction of the conditioning domain that the flow maps to runaway
configurations (|q|_inf > 1000 rad). This is the acceptance metric for the iiwa14
retraining: the shipped `lemon-haze-7` checkpoint measures 3.34% here against the
Panda checkpoint's 0.065%, and that factor is the whole of the iiwa's benchmark
deficit in the downstream optimization-IK project.

The sampler is an exact replica of the recorded diagnostic (learned-ik project,
2026-09): position uniform in [0.4, 0, 0.5] +- 0.25 m, orientation from
RollPitchYaw(uniform(-pi, pi, 3)) — deliberately NOT Haar-uniform on SO(3) — and
latent with uniform direction and RADIUS-uniform magnitude in [0, 4.3]. The
per-sample draw order matches the reference so a given (n, seed) reproduces it
bit-for-bit. Do not "fix" the samplers: the recorded baselines were measured with
exactly these draws, and comparability is the point.
"""

import copy
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

POSITION_BASE = np.array([0.4, 0.0, 0.5])
POSITION_SLACK = 0.25
LATENT_RADIUS = 4.3
POLE_THRESHOLD = 1000.0


def rpy_to_wxyz(rpy: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) from Drake-convention roll-pitch-yaw: R = Rz(y)·Ry(p)·Rx(r).

    Implemented in numpy so the training environment never needs Drake; unit-tested
    against pydrake.math.RollPitchYaw in the learned-ik repo's test suite.
    """
    r2, p2, y2 = rpy[0] / 2.0, rpy[1] / 2.0, rpy[2] / 2.0
    cr, sr = np.cos(r2), np.sin(r2)
    cp, sp = np.cos(p2), np.sin(p2)
    cy, sy = np.cos(y2), np.sin(y2)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    # Drake's ToQuaternion() canonicalizes to w >= 0. q and -q are the same rotation but
    # DIFFERENT network conditioning vectors, and the recorded baselines were measured
    # with canonical quaternions — without this the pole fraction reads ~2x high.
    if q[0] < 0:
        q = -q
    return q


def sample_conditioning_and_latents(n: int, width: int, seed: int = 0):
    """The reference sampler. Returns (c, z): c is (n, 8) = [xyz, wxyz, softflow 0], z is (n, width).

    The python-loop draw order (position, rpy, latent direction, latent radius — per
    sample) reproduces the recorded diagnostic's RNG stream exactly.
    """
    rng = np.random.default_rng(seed)
    c = np.empty((n, 8))
    z = np.empty((n, width))
    for i in range(n):
        c[i, :3] = POSITION_BASE + rng.uniform(-POSITION_SLACK, POSITION_SLACK, 3)
        c[i, 3:7] = rpy_to_wxyz(rng.uniform(-np.pi, np.pi, 3))
        c[i, 7] = 0.0  # softflow noise-magnitude column, zero at test time
        direction = rng.normal(0, 1, width)
        z[i] = direction / np.linalg.norm(direction) * rng.uniform(0, LATENT_RADIUS)
    return c, z


def pole_metrics(
    nn_model: torch.nn.Module,
    width: int,
    ndof: int,
    n: int = 4000,
    seed: int = 0,
    chunk: int = 1000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Evaluate the flow in float64 on the reference samples; return pole statistics.

    Clones the model to float64 (the recorded baselines are float64; float32 has a
    ~1e-7 value noise floor) without touching the live training weights.
    """
    model = copy.deepcopy(nn_model).double().eval()
    if device is not None:
        model = model.to(device)
    dev = next(model.parameters()).device
    c_np, z_np = sample_conditioning_and_latents(n, width, seed=seed)
    qinf_chunks = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            c = torch.tensor(c_np[i : i + chunk], dtype=torch.float64, device=dev)
            z = torch.tensor(z_np[i : i + chunk], dtype=torch.float64, device=dev)
            output, _ = model(z, c=c, rev=True)
            qinf_chunks.append(output[:, :ndof].abs().amax(dim=1).cpu())
    qinf = torch.cat(qinf_chunks).numpy()
    return {
        "pole/frac_gt_1000": float((qinf > POLE_THRESHOLD).mean()),
        "pole/frac_gt_3": float((qinf > 3.0).mean()),
        "pole/p50": float(np.percentile(qinf, 50)),
        "pole/p99": float(np.percentile(qinf, 99)),
        "pole/max": float(qinf.max()),
        "pole/n": float(n),
    }


class PoleFractionCallback(Callback):
    """Rank-0 callback: pole metrics at every validation epoch end, plus a status.json
    heartbeat the cluster tooling polls (one exact file path, no directory walks)."""

    def __init__(self, run_dir: str, n: int = 4000, seed: int = 0, chunk: int = 1000):
        self.run_dir = run_dir
        self.n = n
        self.seed = seed
        self.chunk = chunk

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        metrics = pole_metrics(
            pl_module.nn_model,
            width=pl_module.dim_tot,
            ndof=pl_module.ndof,
            n=self.n,
            seed=self.seed,
            chunk=self.chunk,
        )
        pl_module.safe_log_metrics(metrics)
        self._write_status(trainer, pl_module, metrics)

    def _write_status(self, trainer, pl_module, pole: Dict[str, float]):
        batch_size = getattr(trainer.datamodule, "_batch_size", None)
        samples_per_step = trainer.world_size * batch_size if batch_size else None
        try:
            lr = pl_module.get_lr()
        except Exception:
            lr = None
        status = {
            "global_step": trainer.global_step,
            "world_size": trainer.world_size,
            "batch_size_per_rank": batch_size,
            "samples_per_step": samples_per_step,
            "samples_seen": trainer.global_step * samples_per_step if samples_per_step else None,
            "learning_rate": lr,
            "val_l2_error": _metric(trainer, "val/l2_error"),
            "val_angular_error": _metric(trainer, "val/angular_error"),
            "tr_loss": _metric(trainer, "tr/loss_ml"),
            "pole": pole,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        os.makedirs(self.run_dir, exist_ok=True)
        tmp = os.path.join(self.run_dir, ".status.json.tmp")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=1)
        os.replace(tmp, os.path.join(self.run_dir, "status.json"))


def _metric(trainer, name: str):
    value = trainer.callback_metrics.get(name)
    if value is None:
        return None
    return float(value)
