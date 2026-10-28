# EfficientAT for ICBHI Respiratory Sound Classification

This repository adapts [EfficientAT](https://github.com/fschmid56/EfficientAT), an efficient audio CNN project, for binary classification of respiratory sounds from the ICBHI 2017 Respiratory Sound Database. It fine-tunes pretrained MobileNet or Dynamic MobileNet (DyMN) models to classify respiratory cycles as **Normal** or **Abnormal**.

The ICBHI training script creates patient-disjoint training and validation subsets from the official training split, selects checkpoints using validation ICBHI score, then evaluates both the best and last checkpoint on the official test split. Training and metrics are logged to Weights & Biases (W&B).

This is a research adaptation of EfficientAT. The upstream repository and its license are credited in [Third_party_notices.md](Third_party_notices.md); the original [LICENSE](LICENSE) is retained.

## Repository contents

- `ex_icbhi.py` — ICBHI model fine-tuning and evaluation entry point.
- `datasets/icbhi.py` — ICBHI cycle-level dataset loader and patient-grouped split logic.
- `ICBHI_dataset/splitting ICBHI Audios into audio_cycles.ipynb` — notebook for preparing respiratory-cycle audio and metadata.
- `models/`, `helpers/` — EfficientAT model architectures, preprocessing, and training utilities.
- `pyproject.toml`, `poetry.lock` — Poetry environment and locked dependencies.
- `Third_party_notices.md` — upstream software, dataset, and pretrained-weight acknowledgments.

## Requirements

- Python 3.12 or 3.13 are recommended. The Poetry constraint accepts Python 3.12 and newer, excluding version 3.14.1, and below Python 4.
- Poetry.
- A CUDA-capable GPU is recommended; CPU mode is available but training will be slower.
- A W&B account, because the training script logs runs to W&B.

### Install Poetry

Follow [Poetry's official installation guide](https://python-poetry.org/docs/#installation). On Windows PowerShell, the official installer command is:

```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Open a new terminal after installation. From the repository root, select a supported Python interpreter and install the project dependencies from the lock file:

```powershell
poetry env use 3.12
poetry install
```

The project dependencies include PyTorch, TorchAudio, TorchVision, librosa, scikit-learn, pandas, and W&B. PyTorch GPU support depends on the installed PyTorch build and the machine's CUDA drivers. Check the detected device with:

```powershell
poetry run python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) if you need a specific CUDA-compatible PyTorch build.

## Get and prepare the ICBHI dataset

Download the Respiratory Sound Database by following the [official ICBHI Challenge website](https://bhichallenge.med.auth.gr/). Obtain the dataset directly from the official source and follow its access and research-use terms. The audio dataset is separate from this code repository and is not covered by the repository's MIT license.

The training script expects cycle-level WAV files and a CSV under the dataset directory:

```text
ICBHI_dataset/
├── meta.csv
├── metadata_df.csv
├── audios_cycles/
│   ├── <cycle-recording>.wav
│   └── ...
└── splitting ICBHI Audios into audio_cycles.ipynb
```

Run the included notebook to prepare cycle audio and metadata from the downloaded recordings and annotations. The loader expects `meta.csv` to include these columns:

| Column | Description |
| --- | --- |
| `fold` | `train` or `test`, identifying the official partition |
| `filename` | WAV filename located in `audios_cycles/` |
| `target` | Integer class ID: `0` for Normal and `1` for Abnormal |

Each CSV row represents one respiratory cycle. The loader reads cycles from `audios_cycles/`, resamples them to 32 kHz by default, and pads or truncates them to the configured clip length (8 seconds by default). It groups records by patient ID parsed from the filename to make the train/validation split. Rows marked `test` remain the official held-out test set.


## Configure Weights & Biases

Authenticate with W&B before training:

```powershell
poetry run wandb login
```

Enter your W&B API key when prompted. Runs are logged under the `ICBHI-EfficientAT` project. The training script saves best and last checkpoints in the W&B run directory, then reports test results for both.

## Train and evaluate

Run commands from the repository root. For example, fine-tune an AudioSet-pretrained `mn10_as` model:

```powershell
poetry run python ex_icbhi.py `
  --dataset_dir "ICBHI_dataset" `
  --pretrained `
  --model_name mn10_as `
  --n_epochs 20 `
  --batch_size 16 `
  --num_workers 2
```

The script also supports `dymn` models (for example, `--model_name dymn10_as`). Use `--cpu` to force CPU execution. Run `poetry run python ex_icbhi.py --help` to see all options, including clip length, validation fraction, random seed, learning rate, and preprocessing settings.

The default patient-level validation fraction is 0.2 and the default random seed is 1337. The official test split is held out from training and validation. For a reproducible experiment, record the command, dataset metadata, W&B run, model name, and pretrained setting.

## Upstream project and citations

This work builds on the EfficientAT codebase and pretrained audio models. Please cite the relevant EfficientAT publications and the ICBHI Respiratory Sound Database when using this repository in research. See the upstream [EfficientAT repository](https://github.com/fschmid56/EfficientAT) and the official [ICBHI Challenge website](https://bhichallenge.med.auth.gr/) for project and dataset references.

## License and third-party terms

The upstream EfficientAT `LICENSE` file is retained in this repository. Third-party software, pretrained model weights, and the ICBHI dataset may have separate licenses or usage conditions. See [Third_party_notices.md](Third_party_notices.md) and check the terms attached to each source before redistribution.

