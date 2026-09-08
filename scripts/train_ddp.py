"""DDP-capable IKFlow training entry point.

A fork-local sibling of train.py (which is kept untouched for upstream diffability).
Differences:
  - Multi-GPU / multi-node DDP via torchrun (one `torchrun --nproc_per_node=<gpus>`
    launch per node, c10d rendezvous). Single-process runs work too (no torchrun).
  - Local checkpoint resume: --ckpt_path=auto resumes from <run_dir>/checkpoints/last.ckpt.
  - CSVLogger always (ground truth for analysis); WandbLogger optional, offline-friendly.
    No wandb checkpoint artifacts — checkpoints stay on disk.
  - PoleFractionCallback: the retraining campaign's acceptance metric, logged at every
    validation and written to <run_dir>/status.json as a heartbeat.

Example (laptop smoke, single GPU, no torchrun):
    python scripts/train_ddp.py --robot_name=iiwa14 --run_dir=/tmp/smoke1 \
        --num_nodes=1 --gpus_per_node=1 --batch_size=128 --max_steps=600 \
        --eval_every=200 --val_set_size=20 --checkpoint_every=200 --log_every=50 \
        --disable_wandb

Example (one cluster node, both GPUs):
    torchrun --nnodes=1 --nproc_per_node=2 scripts/train_ddp.py --robot_name=iiwa14 \
        --run_dir=$RUN_DIR --num_nodes=1 --gpus_per_node=2 --ckpt_path=auto ...
"""

import argparse
import os
import time

# ---- Device preamble: MUST run before any jrl import. -----------------------------
# torchrun sets LOCAL_RANK/RANK/WORLD_SIZE; single-process runs default to 0/0/1.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.environ.get("RANK", 0))

import torch  # noqa: E402
import jrl.config  # noqa: E402,F401  (side effect: picks a device, sets torch default device)

# jrl.config calls torch.set_default_device(DEVICE) at import — an inference-side
# convenience that poisons every default-device tensor creation under DDP (samplers,
# loggers, CPU-side dataloading). Neutralize it and pin this process to its own GPU.
torch.set_default_device("cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)

from jrl.robots import get_robot  # noqa: E402
from pytorch_lightning.loggers import CSVLogger, WandbLogger  # noqa: E402
from pytorch_lightning.callbacks import ModelCheckpoint  # noqa: E402
from pytorch_lightning.trainer import Trainer  # noqa: E402
from pytorch_lightning.plugins.environments import (  # noqa: E402
    LightningEnvironment,
    TorchElasticEnvironment,
)
from pytorch_lightning import seed_everything  # noqa: E402
import wandb  # noqa: E402

from ikflow.config import DATASET_TAG_NON_SELF_COLLIDING  # noqa: E402
from ikflow.model import IkflowModelParameters  # noqa: E402

# Our checkpoints embed a pickled IkflowModelParameters (Lightning hyper_parameters);
# torch >= 2.6 defaults torch.load to weights_only=True, which rejects it on resume.
torch.serialization.add_safe_globals([IkflowModelParameters])
from ikflow.ikflow_solver import IKFlowSolver  # noqa: E402
from ikflow.training.lt_model import IkfLitModel  # noqa: E402
from ikflow.training.lt_data import IkfLitDataset  # noqa: E402
from ikflow.training.pole_callback import PoleFractionCallback  # noqa: E402
from ikflow.utils import boolean_string, non_private_dict, get_wandb_project  # noqa: E402

DEFAULT_MAX_EPOCHS = 5000
SEED = 0
seed_everything(SEED, workers=True)


def parse_args():
    parser = argparse.ArgumentParser(prog="IKFlow DDP training script")
    parser.add_argument("--robot_name", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True, help="Checkpoints, metrics and status.json live here")

    # Distribution
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=1)

    # Resume
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="auto",
        help="'auto' resumes from <run_dir>/checkpoints/last.ckpt when present; 'none' forces a fresh start; "
        "otherwise an explicit .ckpt path",
    )

    # Model parameters (defaults = the lemon-haze-7 / iiwa14 architecture)
    parser.add_argument("--coupling_layer", type=str, default="glow")
    parser.add_argument("--rnvp_clamp", type=float, default=2.5)
    parser.add_argument("--softflow_noise_scale", type=float, default=0.001)
    parser.add_argument("--softflow_enabled", type=str, default=True)
    parser.add_argument("--nb_nodes", type=int, default=12)
    parser.add_argument("--dim_latent_space", type=int, default=8)
    parser.add_argument("--coeff_fn_config", type=int, default=3)
    parser.add_argument("--coeff_fn_internal_size", type=int, default=1024)
    parser.add_argument("--y_noise_scale", type=float, default=1e-7)
    parser.add_argument("--zeros_noise_scale", type=float, default=1e-3)
    parser.add_argument("--sigmoid_on_output", type=str, default=False)

    # Training parameters
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--batch_size", type=int, default=256, help="PER-RANK batch size")
    parser.add_argument("--gamma", type=float, default=0.9794578299341784)
    parser.add_argument("--learning_rate", type=float, default=1.06e-4)
    parser.add_argument("--step_lr_every", type=int, default=4883)
    parser.add_argument("--gradient_clip_val", type=float, default=1)
    parser.add_argument("--lambd", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=1.8e-05)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optimizer-step cap; -1 = unbounded (smokes/calibration)")

    # Logging options
    parser.add_argument("--eval_every", type=int, default=20000)
    parser.add_argument("--val_set_size", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=20000)
    parser.add_argument("--pole_eval_n", type=int, default=4000)
    parser.add_argument("--dataset_tags", nargs="+", type=str, default=[DATASET_TAG_NON_SELF_COLLIDING])
    parser.add_argument("--run_description", type=str)
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if GLOBAL_RANK == 0:
        print("\nArgparse arguments:")
        for k, v in vars(args).items():
            print(f"  {k}={v}")
        print()

    assert DATASET_TAG_NON_SELF_COLLIDING in args.dataset_tags
    assert args.optimizer in ["ranger", "adadelta", "adamw"]
    # Ranger uses manual optimization, which is incompatible with the DDP gradient hooks
    # this script exists for.
    assert args.optimizer != "ranger" or args.num_nodes * args.gpus_per_node == 1

    # jrl's Robot.__init__ truncate-rewrites its cached *_link_filepaths_absolute.urdf
    # on EVERY construction, and all ranks share one $HOME/.cache/jrl on Lustre — a rank
    # that loadRobot()s while another rank's write is in flight reads half-written XML
    # (killed job 5549615). Stagger construction by rank, and retry in case a straggler
    # still collides.
    time.sleep(2.0 * GLOBAL_RANK)
    for _attempt in range(3):
        try:
            robot = get_robot(args.robot_name)
            break
        except (AssertionError, ValueError):
            if _attempt == 2:
                raise
            time.sleep(5.0)
    base_hparams = IkflowModelParameters()
    base_hparams.run_description = args.run_description
    base_hparams.coupling_layer = args.coupling_layer
    base_hparams.nb_nodes = args.nb_nodes
    base_hparams.dim_latent_space = args.dim_latent_space
    base_hparams.coeff_fn_config = args.coeff_fn_config
    base_hparams.coeff_fn_internal_size = args.coeff_fn_internal_size
    base_hparams.rnvp_clamp = args.rnvp_clamp
    base_hparams.softflow_noise_scale = args.softflow_noise_scale
    base_hparams.y_noise_scale = args.y_noise_scale
    base_hparams.zeros_noise_scale = args.zeros_noise_scale
    base_hparams.softflow_enabled = boolean_string(args.softflow_enabled)
    base_hparams.sigmoid_on_output = boolean_string(args.sigmoid_on_output)
    if GLOBAL_RANK == 0:
        print(base_hparams)

    torch.autograd.set_detect_anomaly(False)
    data_module = IkfLitDataset(robot.name, args.batch_size, args.val_set_size, args.dataset_tags)

    world_size = args.num_nodes * args.gpus_per_node
    samples_per_step = world_size * args.batch_size

    # Loggers: CSV always; wandb (offline-friendly) unless disabled. wandb.init only on
    # global rank 0 — every torchrun rank executes this script.
    ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    loggers = [CSVLogger(save_dir=args.run_dir, name="metrics")]
    if not args.disable_wandb:
        if GLOBAL_RANK == 0:
            wandb_entity, wandb_project = get_wandb_project()
            cfg = {"robot": args.robot_name, "world_size": world_size, "samples_per_step": samples_per_step}
            cfg.update(non_private_dict(args.__dict__))
            data_module.add_dataset_hashes_to_cfg(cfg)
            wandb.init(entity=wandb_entity, project=wandb_project, notes=args.run_description, config=cfg)
        # NOTE: no log_model — checkpoints are large and live on disk, not in wandb.
        loggers.append(WandbLogger(save_dir=args.run_dir))

    ik_solver = IKFlowSolver(base_hparams, robot)
    model = IkfLitModel(
        ik_solver=ik_solver,
        base_hparams=base_hparams,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        gradient_clip=args.gradient_clip_val,
        lambd=args.lambd,
        gamma=args.gamma,
        step_lr_every=args.step_lr_every,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        sigmoid_on_output=boolean_string(args.sigmoid_on_output),
    )

    # save_top_k=-1 keeps EVERY checkpoint. This was 2, which silently rotated the run's
    # history away: the iiwa14_ddp_r1 run developed pole mass somewhere around step 360000
    # and by the time anyone looked, every checkpoint before 540000 was gone, so when the
    # pole mass appeared could not be recovered without retraining from scratch. Checkpoints
    # are ~611 MB at this architecture and a full run writes ~30 of them, so keeping all of
    # them costs ~19 GB against a filesystem with petabytes free -- nothing, against the
    # cost of re-running a multi-day job to answer a question the checkpoints already held.
    # monitor/mode are irrelevant when nothing is being ranked, so they are dropped.
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_train_steps=args.checkpoint_every,
        save_on_train_epoch_end=False,
        save_top_k=-1,
        save_last=True,
        filename="ikflow-checkpoint-{step}",
    )
    pole_callback = PoleFractionCallback(run_dir=args.run_dir, n=args.pole_eval_n)

    # Rank resolution: inside a Slurm batch job Lightning's auto-detection would pick
    # SLURMEnvironment and read SLURM_PROCID — which is identical for every torchrun
    # child on a node — so rank resolution MUST be pinned to torchrun's env when
    # torchrun is the launcher.
    env = TorchElasticEnvironment() if "TORCHELASTIC_RUN_ID" in os.environ else LightningEnvironment()

    trainer = Trainer(
        logger=loggers,
        callbacks=[checkpoint_callback, pole_callback],
        # int val_check_interval counts training batches; check_val_every_n_epoch=None
        # lets it cross epoch boundaries (at global batch 2048 an "epoch" of the 25M-row
        # dataset is ~12k steps, less than eval_every).
        val_check_interval=args.eval_every,
        check_val_every_n_epoch=None,
        accelerator="gpu",
        devices=args.gpus_per_node,
        num_nodes=args.num_nodes,
        strategy="ddp",
        plugins=[env],
        use_distributed_sampler=True,
        log_every_n_steps=args.log_every,
        max_epochs=DEFAULT_MAX_EPOCHS,
        max_steps=args.max_steps,
        enable_progress_bar=False if (os.getenv("IS_SLURM") is not None) or args.disable_progress_bar else True,
    )
    assert trainer.world_size == world_size, (
        f"trainer.world_size={trainer.world_size} but --num_nodes x --gpus_per_node={world_size}; "
        "rank resolution is broken (SLURMEnvironment hijack?)"
    )

    resume_path = args.ckpt_path
    if resume_path == "auto":
        last = os.path.join(ckpt_dir, "last.ckpt")
        resume_path = last if os.path.isfile(last) else None
    elif resume_path == "none":
        resume_path = None
    if GLOBAL_RANK == 0:
        print(f"world_size={trainer.world_size}, samples_per_step={samples_per_step}, resume={resume_path}")

    _t0 = time.monotonic()
    trainer.fit(model, data_module, ckpt_path=resume_path)
    _elapsed = time.monotonic() - _t0
    if GLOBAL_RANK == 0:
        # Throughput line for the calibration stage. global_step counts optimizer
        # steps from the START of this fit only when not resuming; on resume the
        # steps/s figure is diluted by the restored offset, so calibration runs
        # start fresh. Elapsed includes startup (~30-40 s), amortized by >=1000
        # steps in calibration configs.
        print(
            f"THROUGHPUT steps={trainer.global_step} elapsed_s={_elapsed:.1f} "
            f"steps_per_s={trainer.global_step / _elapsed:.3f} "
            f"samples_per_s={trainer.global_step * samples_per_step / _elapsed:.0f} "
            f"world_size={trainer.world_size} batch_per_rank={args.batch_size}"
        )
