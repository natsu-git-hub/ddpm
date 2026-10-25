# DDPM on MNIST

A from-scratch PyTorch implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020), trained on MNIST.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Training

Run from the repo root:

```bash
uv run python main.py
```

MNIST is downloaded to `data/` on the first run. Training uses CUDA if available, then Apple MPS, then CPU.

Outputs, all written to the repo root:

- `models/model_{epoch}.pth`: checkpoints (model, optimizer, epoch, loss history)
- `samples/{epoch}.png`: a 4×4 grid of generated digits
- `loss.png`: per-step training loss, updated every epoch

To resume, run the same command again. Training continues from the latest checkpoint in `models/`. Delete `models/` to start from scratch.

## Configuration

Hyperparameters are constants at the top of `src/train.py`:

| Name | Default | Meaning |
|---|---|---|
| `MAX_EPOCHS` | 30 | Total epochs |
| `BATCH_SIZE` | 64 | Batch size |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `TIME_STEPS` | 1000 | Diffusion steps T |
| `SAVE_INTERVAL` | 5 | Epochs between checkpoints |
| `INFERENCES_INTERVAL` | 5 | Epochs between sample grids |
| `NUM_OF_SAMPLES` | 16 | Images per sample grid |

The noise schedule follows the paper: β rises linearly from 1e-4 to 0.02.

## Code layout

- `main.py`: entry point
- `src/train.py`: training loop (Algorithm 1), checkpointing and resume
- `src/utils.py`: noise schedule, loss, sampling (Algorithm 2), image and loss-plot saving
- `src/model.py`: UNet with sinusoidal time embeddings
- `src/dataset.py`: MNIST wrapper that scales pixels to [-1, 1]
