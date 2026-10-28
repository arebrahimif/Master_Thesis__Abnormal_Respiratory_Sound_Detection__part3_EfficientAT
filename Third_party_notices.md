# Third-Party Notices

This project adapts the EfficientAT repository for ICBHI respiratory-sound classification. The notices below identify upstream software and external resources used by the project. This file does not replace the license or terms supplied by any upstream project, package, model provider, or dataset owner.

## EfficientAT

- Project: [EfficientAT — Efficient Pre-Trained CNNs for Audio Pattern Recognition](https://github.com/fschmid56/EfficientAT)
- Copyright: Copyright (c) 2022 Florian Schmid
- License: MIT License. The upstream `LICENSE` file is retained in this repository; its copyright and permission notice must remain with copies or substantial portions of the upstream software.
- Use here: this repository retains and adapts EfficientAT model code, preprocessing, and training utilities, and adds ICBHI dataset loading and fine-tuning/evaluation code.

The upstream MIT license text is in [`LICENSE`](LICENSE). The upstream project also acknowledges inspiration from PaSST, PANNs, and the PyTorch MobileNetV3 implementation; refer to the [EfficientAT README](https://github.com/fschmid56/EfficientAT#readme) for those references and their original sources.

## Python dependencies

Third-party Python packages are declared in `pyproject.toml` and pinned in `poetry.lock` (the repository also contains `requirements.txt`). These packages are installed from their respective publishers and remain subject to their own licenses and notices. Important direct dependencies include PyTorch, TorchAudio, TorchVision, librosa, NumPy, pandas, scikit-learn, tqdm, and Weights & Biases. Consult each package's distribution metadata for its current license and any required notices; this project does not relicense those packages.

## Pretrained model weights

The training script can load EfficientAT pretrained models and may download weights from sources configured by the upstream code. Weight files are separate artifacts and may have terms that differ from the source-code MIT license. Check the terms and attribution for each checkpoint at its original distribution source before copying or redistributing model weights. This notice does not grant rights to third-party weights.

## ICBHI Respiratory Sound Database

- Source: [Official ICBHI Challenge website](https://bhichallenge.med.auth.gr/)
- The respiratory audio and associated annotations are a dataset, not software covered by the EfficientAT MIT license. Obtain them from the official source and follow the dataset's access, research-use, citation, and redistribution conditions.
- This project expects locally prepared cycle-level audio and metadata. Do not assume that the code license permits uploading or redistributing the dataset or derived audio.

## Project additions

The ICBHI fine-tuning entry point and dataset loader are project-specific adaptations in `ex_icbhi.py` and `datasets/icbhi.py`. Their presence does not change the license or attribution requirements for upstream EfficientAT code, third-party packages, pretrained weights, or the ICBHI dataset.
