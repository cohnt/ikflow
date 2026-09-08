# IKFlow
Normalizing flows for Inverse Kinematics. Open source implementation to the paper ["IKFlow: Generating Diverse Inverse Kinematics Solutions"](https://ieeexplore.ieee.org/abstract/document/9793576)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2111.08933-red)](https://arxiv.org/abs/2111.08933)


Runtime curve for getting *exact* IK solutions for the Franka Panda (maximum positional/rotational error: 1mm, .572 deg) (generated with `
python scripts/benchmark_generate_exact_solutions.py --model_name=panda__full__lp191_5.25m`):

![alt text](../media/exact_ik_runtime__model:panda__full__lp191_5.25m.png?raw=true)


## This fork

This is a fork of [jstmn/ikflow](https://github.com/jstmn/ikflow) maintained for the
`learned-ik` project (solving IK with an IKFlow network inside a Drake optimization program).
It exists to retrain the iiwa14 checkpoint on multiple GPUs; upstream's training path is
single-GPU only.

**Everything upstream still works unchanged.** `scripts/train.py` is deliberately untouched so
the fork stays easy to diff against upstream, and inference is unaffected.

What this fork adds:

- **`scripts/train_ddp.py`** — multi-node, multi-GPU training via PyTorch DDP under Lightning.
  Launched with `torchrun`; an explicit `TorchElasticEnvironment` plugin is used so Lightning
  does not auto-select `SLURMEnvironment` and mis-rank the torchrun children inside a Slurm job.
- **Every checkpoint is kept** (`save_top_k=-1`). Upstream's rotation discards all but
  the most recent few, which makes a finished run unable to answer any question about
  *when* something changed during training. At ~611 MB each and ~30 per run that is
  ~19 GB, far cheaper than re-running a multi-day job.
- **Local checkpoint resume.** Upstream can only resume from a wandb artifact.
  `--ckpt_path=auto` picks up `last.ckpt` from the run directory, restoring optimizer state,
  the LR schedule and `global_step`.
- **`ikflow/training/pole_callback.py`** — a validation-time diagnostic that samples the
  conditioning/latent domain and reports the fraction of draws whose output joint
  configuration blows up (`pole/frac_gt_1000`, `frac_gt_3`, p50/p99/max). This is the metric
  the retrain is aimed at. It also writes a `status.json` heartbeat, which is how long cluster
  runs are monitored.
- **`samples_seen` logging**, so runs at different world sizes can be compared on the samples
  axis rather than on optimizer-step counts.
- **`--seed` for `scripts/build_dataset.py`** (upstream dataset generation is unseeded).

Bug fixes carried here:

- `ikflow/training/lt_data.py` imported a lowercase `device` that does not exist (`ImportError`).
- `StepLR(..., verbose=...)` raises a `TypeError` on torch >= 2.x.
- Resume failed under torch >= 2.6 due to the `weights_only` default (fixed with `safe_globals`).
- `safe_log_metrics` logged only to `self.logger` (i.e. `loggers[0]`), so with more than one
  logger attached, all but the first silently received nothing.
- `jrl` truncate-rewrites its cached `*_link_filepaths_absolute.urdf` on every `Robot` init;
  with several ranks sharing a filesystem, one rank could read half-written XML. `get_robot`
  is now staggered per rank and retried.

DDP-specific changes to the training internals: the dataset is kept on CPU and handed to a
plain `DataLoader` so Lightning can attach a `DistributedSampler`; `jrl.config`'s
module-scope `torch.set_default_device` is neutralized; the model is placed by Lightning
rather than by an explicit `.to(DEVICE)`; and the per-rank softflow seed is decorrelated.

Example (8 ranks over 4 nodes, 2 GPUs each):

```
torchrun --nnodes=4 --nproc_per_node=2 --node_rank=$NODE_RANK \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
    scripts/train_ddp.py \
    --robot_name=iiwa14 --run_dir=/path/to/run \
    --num_nodes=4 --gpus_per_node=2 \
    --batch_size=512 --learning_rate=1.5e-4 --step_lr_every=2441 \
    --ckpt_path=auto
```

Note `--batch_size` is **per rank**; the global batch is `batch_size * world_size`. When
scaling the global batch, scale `--step_lr_every` down by the same ratio so the LR decay per
*sample* is unchanged.

## Setup

The following section outlines the setup procedures required to run the visualizer that this project uses. The only supported OS is Ubuntu. Visualization may work on Mac and Windows, I haven't tried it though. For Ubuntu, there are different system wide dependencies for `Ubuntu > 21` and `Ubuntu < 21`. For example, `qt5-default` is not in the apt repository for Ubuntu 21.0+ so can't be installed. See https://askubuntu.com/questions/1335184/qt5-default-not-in-ubuntu-21-04.

<ins>Ubuntu >= 21.04</ins>
```
sudo apt-get install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools libosmesa6 build-essential qtcreator
export PYOPENGL_PLATFORM=osmesa # this needs to be run every time you run a visualization script in a new terminal - annoying, I know
```
<ins>Ubuntu <= 20.x.y</ins>

(This includes 20.04 LTS, 18.04 LTS, ...)
```
sudo apt-get install -y qt5-default build-essential qtcreator
```

Lastly, install with uv:
``` bash
git clone https://github.com/jstmn/ikflow.git && cd ikflow
uv sync
uv pip install -e .
```


## Getting started

**> Example 1: Use IKFlow to generate approximate IK solutions for the Franka Panda**

Evaluate a pretrained IKFlow model for the Franka Panda arm. Note that the value for `model_name` - in this case `panda__full__lp191_5.25m` should match an entry in `model_descriptions.yaml` 
```
uv run python scripts/evaluate.py --testset_size=500 --model_name=panda__full__lp191_5.25m
```

**> Example 2: Use IKFlow to generate exact IK solutions for the Franka Panda**

Additional examples are provided in examples/example.py. This file includes examples of collision checking and pose error calculation, among other utilities.

```
ik_solver, _ = get_ik_solver("panda__full__lp191_5.25m")
target_poses = torch.tensor(
    [
        [0.25, 0, 0.5, 1, 0, 0, 0],
        [0.35, 0, 0.5, 1, 0, 0, 0],
        [0.45, 0, 0.5, 1, 0, 0, 0],
    ],
    device=device,
)
solutions, _ = ik_solver.generate_exact_ik_solutions(target_poses)
```


**> Example 3: Visualize the solutions returned by the `fetch_arm__large__mh186_9.25m` model**

Run the following:
```
uv run python scripts/visualize.py --model_name=fetch_arm__large__mh186_9.25m --demo_name=oscillate_target
```
![ikflow solutions for oscillating target pose](../media/ikflow__fetcharm__oscillating-target.gif?raw=true)

Run an interactive notebook: `jupyter notebook notebooks/robot_visualizations.ipynb`


## Notes
This project uses the `w,x,y,z` format for quaternions. That is all.


## Training new models

The training code uses [Pytorch Lightning](https://www.pytorchlightning.ai/) to setup and perform the training and [Weights and Biases](https://wandb.ai/) ('wandb') to track training runs and experiments. WandB isn't required for training but it's what this project is designed around. Changing the code to use Tensorboard should be straightforward (so feel free to put in a pull request for this if you want it :)).

First, create a dataset for the robot:
```
uv run python scripts/build_dataset.py --robot_name=panda --training_set_size=25000000 --only_non_self_colliding
```

Then start a training run:
```
# Login to wandb account - Only needs to be run once
uv run wandb login

# Set wandb project name and entity
export WANDB_PROJECT=ikflow 
export WANDB_ENTITY=<your wandb entity name>

uv run python scripts/train.py \
    --robot_name=panda \
    --nb_nodes=12 \
    --batch_size=128 \
    --learning_rate=0.0005
```

## Common errors

1. GLUT font retrieval function when running a visualizer. Run `export PYOPENGL_PLATFORM=osmesa` and then try again. See https://bytemeta.vip/repo/MPI-IS/mesh/issues/66

```
Traceback (most recent call last):
  File "visualize.py", line 4, in <module>
    from ikflow.visualizations import _3dDemo
  File "/home/jstm/Projects/ikflow/utils/visualizations.py", line 10, in <module>
    from klampt import vis
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/__init__.py", line 3, in <module>
    from .glprogram import *
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/glprogram.py", line 11, in <module>
    from .glviewport import GLViewport
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/glviewport.py", line 8, in <module>
    from . import gldraw
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/klampt/vis/gldraw.py", line 10, in <module>
    from OpenGL import GLUT
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/OpenGL/GLUT/__init__.py", line 5, in <module>
    from OpenGL.GLUT.fonts import *
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/OpenGL/GLUT/fonts.py", line 20, in <module>
    p = platform.getGLUTFontPointer( name )
  File "/home/jstm/Projects/ikflow/venv/lib/python3.8/site-packages/OpenGL/platform/baseplatform.py", line 350, in getGLUTFontPointer
    raise NotImplementedError( 
NotImplementedError: Platform does not define a GLUT font retrieval function
```

2. If you get this error: `tkinter.TclError: no display name and no $DISPLAY environment variable`, add the lines below to the top of `ik_solvers.py` (anywhere before `import matplotlib.pyplot as plt` should work).
``` python
import matplotlib
matplotlib.use("Agg")
```


## Citation
```
@ARTICLE{9793576,
  author={Ames, Barrett and Morgan, Jeremy and Konidaris, George},
  journal={IEEE Robotics and Automation Letters}, 
  title={IKFlow: Generating Diverse Inverse Kinematics Solutions}, 
  year={2022},
  volume={7},
  number={3},
  pages={7177-7184},
  doi={10.1109/LRA.2022.3181374}
}
```