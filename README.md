# Smoothed Dirichlet Diffusion (SDD)

- This repository implements and extends diffusion-model training and sampling. The core contribution is a smoothed-Dirichlet prior–based preconditioning and loss (SDD) that integrates seamlessly with standard `UNet`-family architectures and distributed training.
- Built on EDM/DDPM-style training, we add `SDDPrecond`, `SDDLoss`, and an SDD sampler `SDD_sampler`. The pipeline supports images and NPY/NPZ volumetric data, suitable for general vision and medical imaging research.

## Key Features
- SDD training loss: `SDDLoss` uses analytic expectations and variational terms from the Beta/Dirichlet family. Variants include `KLUB`, `HKD`, and `ELBO`. Tunable hyperparameters: `eta`, `sigmoid_*`, `Shift/Scale`, `lossType`.
- SDD preconditioning: `SDDPrecond` drives prediction using `logit_alpha` as the noise label, compatible with `SongUNet`/`DhariwalUNet`.
- SDD sampler: `SDD_sampler` constructs time steps and reverse updates via differentiable Beta–Dirichlet transforms, supports class-conditional generation and distributed batching, and auto-loads training hyperparameters from `training_options.json`.
- Data loading: Besides standard image folders/Zip, NPY/NPZ volumes are supported with automatic slicing/channel adaptation.
- Evaluation tools: FID computation and reference statistics scripts available.

## Repository Layout
- Top-level scripts:
  - `train.py` training entry with distributed parallelism; supports `ddpmpp/ncsnpp/adm/sdd` architectures and `vp/ve/edm/betadiff` preconditioning. See `docs/train-help.txt` for options.
  - `generate.py` sampling entry that reads `network-snapshot-*.pkl` and `training_options.json`; supports SDD/EDM-style sampling.
  - `fid.py` FID computation and reference statistics.
  - `dataset_tool.py` dataset conversion/cropping for Dir/Zip/LMDB/CIFAR/MNIST.
- Training modules:
  - Training loop with EMA, snapshots, distributed stats, and logging.
  - Networks and preconditioners: `SongUNet`/`DhariwalUNet`, `VP/VE/iDDPM/EDM/SDDPrecond`.
  - Losses: `VPLoss/VELoss/EDMLoss/SDDLoss`.
  - Datasets: `ImageFolderDataset/NPYDataset`.
- Other components: `torch_utils/` (distributed/tools), `models/` (ViT fragments), `docs/` (CLI help), `fid-refs/` (reference stats).

## Environment & Dependencies
- Recommended Python 3.10+ with CUDA 12.x PyTorch. Install via:
  - `pip install -r requirements.txt`
- Note: `requirements.txt` pins `torch/torchvision/triton` and several CUDA components. If using a different CUDA or OS, adjust versions to official wheels and edit/remove the pinned lines before installation.

## Data Preparation
- Image folder or Zip: source must contain recognized image extensions (PNG/JPG, etc.). Class conditioning can be expressed via subfolders or `dataset.json`.
- Conversion/cropping tool (example):
  - Convert a folder to Zip and center-crop to 32×32:
    - `python dataset_tool.py --source=datasets/cifar10-32x32 --dest=datasets/cifar10-32x32.zip --transform=center-crop --resolution=32x32`
 - NPY/NPZ support: 2D/3D/4D arrays are recognized; 3D data are sliced along an axis. Control scaling and slicing via `--npy_resolution` and `--npy_slice_axis`.

## Training
- Single/multi-GPU via `torchrun` (recommended):
  - CIFAR-10 (unconditional, DDPM++ + SDD/betadiff):
    - `torchrun --standalone --nproc_per_node=8 train.py --outdir=SDD-train-runs --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp --precond=betadiff --batch=512 --duration=200 --ema=0.5 --dropout=0.13 --augment=0.12 --xflip=0 --Shift=0.6 --Scale=0.39 --eta=10000.0 --sigmoid_start=10.0 --sigmoid_end=-13.0 --sigmoid_power=1.0 --lossType=HKD`
- Notes:
  - Preconditioning and loss are controlled by `--precond` and SDD hyperparameters. Training hyperparameters are saved to `outdir/<run>/training_options.json` and auto-read by the sampler.
  - `--data` can be Dir/Zip/NPY/NPZ. Class conditioning requires labels via `dataset.json` or folder structure.
  - Distributed batching is governed by `--batch`, `--batch-gpu`, and `--nproc_per_node`. See `training/training_loop.py:55-69, 114-123`.

## Sampling
- Generate from snapshots:
  - `python generate.py --outdir=out --seeds=0-63 --batch=64 --steps=200 --network=SDD-train-runs/00000-.../network-snapshot-200000.pkl`
- Sampler selection: if ablation options (`--solver/--discretization/--schedule`) are given, EDM-style `ablation_sampler` is used (`generate.py:171-270`); otherwise the SDD sampler `SDD_sampler` is used (`generate.py:45-165`).
 - Sampler selection: if ablation options (`--solver/--discretization/--schedule`) are given, EDM-style `ablation_sampler` is used; otherwise the SDD sampler `SDD_sampler` is used.
- Output: two sets of images are written to `out/` and `out_1/`; the script also writes a montage PNG next to the snapshot for quick visualization.

## Evaluation (FID)
- Generate evaluation images:
  - `torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs --network=.../network-snapshot-*.pkl`
- Compute FID:
  - `torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp --ref=fid-refs/cifar10-32x32.npz`
- Build dataset reference stats:
  - `python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz`

## Method Overview (Research-Oriented)
- Noise schedule: SDD parameterizes time in logit space `logit_alpha(t)` with `sigmoid_start/sigmoid_end/sigmoid_power` controlling step shape and endpoints. Training samples current and previous steps to form marginal/conditional terms.
- Target construction: analytic expectation/variance of `x_t` are used to standardize inputs; the network predicts `x_0` in logit space, which is then mapped back to data space.
- Loss family: `compute_loss` aggregates KL and HPD terms to form `KLUB/HKD/ELBO`, including regularization and reverse-direction variants.
- Reverse updates: the sampler forms gradients via Beta–Dirichlet `log_gamma` and aggregates with `logsumexp`.

## Reproduction Tips
- Normalization: training normalizes images to `[-1, 1]`. SDD loss operates over `[Shift, Shift+Scale]`; keep data ranges consistent.
- Hyperparameters: `eta` and `sigmoid_*` strongly affect stability and quality. Start from defaults and tune gradually; `--steps` around 200 works well.
- Multi-GPU: `torchrun --standalone` initializes communication. On Windows/single-node, prefer this mode; ensure `batch` is divisible by `nproc_per_node * batch-gpu`.

## Citation & License
- Paper: this repository aligns with the IJCV topic “Smoothed Dirichlet Diffusion”. Please cite using your official BibTeX.
- License: no explicit LICENSE is provided yet; please add one before public release.

---
- For additional reproduction scripts and visualization, see commands in `docs/*.txt` and plots in `Plot_SDD.ipynb`.
