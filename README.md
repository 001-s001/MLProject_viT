# ViT Reproduction for Course Project

This repository contains the Vision Transformer (ViT) reproduction part of a machine learning course project. It is a clean PyTorch reproduction of the core ideas from **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, not a direct copy of the official JAX/Flax implementation.

The project focuses on small-scale, reproducible experiments on CIFAR-10 and CIFAR-100. It implements and evaluates a compact `SimpleViT` model with image patching, patch embedding, class token, positional embedding, Transformer encoder blocks, and an MLP classification head.

## Task

- Data type: RGB image classification datasets.
- Datasets: CIFAR-10 and CIFAR-100.
- Training task: supervised image classification.
- Metrics: training loss, validation loss, validation accuracy, and best validation accuracy.

## Repository Structure

```text
.
├─ configs/        # Experiment configurations
├─ reports/        # ViT paper notes and presentation outline
├─ runs/           # Metrics, figures, failure cases, and best checkpoints
├─ src/            # Model, dataset, training, evaluation, and analysis code
├─ tests/          # Basic unit tests
├─ requirements.txt
└─ README.md
```

## Experiments

| Model | Dataset | Patch Size | Pretrained | Best Val Acc |
| --- | --- | ---: | --- | ---: |
| SimpleViT | CIFAR-10 | 4 | No | 77.24% |
| SimpleViT | CIFAR-10 | 8 | No | 71.34% |
| SimpleViT | CIFAR-100 | 4 | No | 52.48% |

The `patch_size=4` model performs better than `patch_size=8` on CIFAR-10 because the smaller patch preserves more local details in 32x32 images. CIFAR-100 is harder because it contains more fine-grained classes.

These results are lower than the original ViT paper's pretrained results. This is expected because this project trains a compact ViT from scratch on small datasets, without JFT-300M or ImageNet-21k pretraining.

## Checkpoints and Results

Each experiment folder under `runs/` contains:

- `metrics.csv`: epoch-level training and validation metrics.
- `config.yaml`: copied experiment configuration.
- `classes.txt`: class names.
- `best.pt`: checkpoint with the best validation accuracy.

The main CIFAR-10 patch-size-4 run also includes:

- `accuracy_curve.png`
- `loss_curve.png`
- `confusion_matrix.png`
- `failure_cases.csv`

Only `best.pt` is uploaded. `last.pt` is excluded because it is the final-epoch checkpoint, not necessarily the best model.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a smoke test:

```bash
python -m src.train --config configs/cifar10_vit_patch4.yaml --fast-dev-run
```

Run the main experiments:

```bash
python -m src.train --config configs/cifar10_vit_patch4.yaml
python -m src.train --config configs/cifar10_vit_patch8.yaml
python -m src.train --config configs/cifar100_vit_patch4.yaml
```

Generate analysis artifacts for a completed checkpoint:

```bash
python -m src.analysis.plot_metrics --metrics runs/cifar10_vit_patch4/metrics.csv
python -m src.analysis.confusion_matrix --checkpoint runs/cifar10_vit_patch4/best.pt
python -m src.analysis.failure_cases --checkpoint runs/cifar10_vit_patch4/best.pt --top-k 30
```

## References

- Paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Official implementation: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)
