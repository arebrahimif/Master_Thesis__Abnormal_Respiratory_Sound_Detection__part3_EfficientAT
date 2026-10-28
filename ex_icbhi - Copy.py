import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn.functional as F

from datasets.icbhi import ICBHIDataset
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup


def train(args):
    # Train Models for Acoustic Scene Classification

    # logging is done using wandb
    wandb.init(
            project="ICBHI-EfficientAT",
            notes="Fine-tune EfficientAT MobileNet on ICBHI respiratory sound classification.",
            tags=["ICBHI", "Respiratory Sound Classification", "Fine-Tuning"],
            config=args,
            name=args.experiment_name
    )

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         freqm=args.freqm,
                         timem=args.timem,
                         fmin=args.fmin,
                         fmax=args.fmax,
                         fmin_aug_range=args.fmin_aug_range,
                         fmax_aug_range=args.fmax_aug_range
                         )
    mel.to(device)

    # load prediction model
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=args.pretrain_final_temp,
                         num_classes=2)
    else:
        model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=2)
    model.to(device)

    # dataloader: datasets
    train_dataset = ICBHIDataset(
            dataset_dir=args.dataset_dir,
            split="train",
            resample_rate=args.resample_rate,
            clip_length=args.clip_length,
            val_fraction=args.val_fraction,
            seed=args.seed,
    )
    
    val_dataset = ICBHIDataset(
            dataset_dir=args.dataset_dir,
            split="val",
            resample_rate=args.resample_rate,
            clip_length=args.clip_length,
            val_fraction=args.val_fraction,
            seed=args.seed,
    )

    test_dataset = ICBHIDataset(
            dataset_dir=args.dataset_dir,
            split="test",
            resample_rate=args.resample_rate,
            clip_length=args.clip_length,
            val_fraction=args.val_fraction,
            seed=args.seed,
)
    
    # dataloaders
    dl = DataLoader(
            dataset=train_dataset,
            # worker_init_fn=worker_init_fn,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
    )
    
    eval_dl = DataLoader(
            dataset=val_dataset,
            # worker_init_fn=worker_init_fn,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
    )

    test_dl = DataLoader(
            dataset=test_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
    )

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    best_hs_score = -float("inf")
    best_checkpoint = os.path.join(
        wandb.run.dir,
        f"{model_name}_icbhi_best.pt"
    )
    last_checkpoint = os.path.join(
        wandb.run.dir,
        f"{model_name}_icbhi_last.pt"
    )


    for epoch in range(args.n_epochs):
    
        mel.train()
        model.train()
    
        train_losses = []
    
        pbar = tqdm(dl)
        pbar.set_description(
            f"Epoch {epoch + 1}/{args.n_epochs}"
        )
    
        for batch in pbar:
    
            x, f, y = batch
            bs = x.size(0)
    
            x = x.to(device)
            y = y.to(device)
    
            # Waveform -> Mel spectrogram
            x = _mel_forward(x, mel)
    
            # MixUp only when the batch contains multiple samples
            if args.mixup_alpha and bs > 1:
    
                rn_indices, lam = mixup(
                    bs,
                    args.mixup_alpha
                )
    
                lam = lam.to(x.device)
    
                x = (
                    x * lam.reshape(bs, 1, 1, 1)
                    + x[rn_indices] *
                    (1.0 - lam.reshape(bs, 1, 1, 1))
                )
    
                y_hat, _ = model(x)
    
                loss_a = F.cross_entropy(
                    y_hat,
                    y,
                    reduction="none"
                )
    
                loss_b = F.cross_entropy(
                    y_hat,
                    y[rn_indices],
                    reduction="none"
                )
    
                samples_loss = (
                    loss_a * lam.reshape(bs)
                    + loss_b * (1.0 - lam.reshape(bs))
                )
    
            else:
    
                y_hat, _ = model(x)
    
                samples_loss = F.cross_entropy(
                    y_hat,
                    y,
                    reduction="none"
                )
    
            loss = samples_loss.mean()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_losses.append(loss.detach().cpu().item())
    
        scheduler.step()
    
        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
    
        val_loss, val_accuracy, val_sensitivity, \
        val_specificity, val_hs_score, val_cm, \
        _, _ = _evaluate(
            model, mel, eval_dl, device
        )
    
        train_loss = np.mean(train_losses)
    
        current_lr = optimizer.param_groups[0]["lr"]
    
        wandb.log({
            "train/loss": train_loss,
    
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
            "val/sensitivity": val_sensitivity,
            "val/specificity": val_specificity,
            "val/HS_score": val_hs_score,
    
            "learning_rate": current_lr,
            "epoch": epoch + 1,
        })
    
        print(
            f"\nEpoch {epoch + 1}/{args.n_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Sens: {val_sensitivity:.4f} | "
            f"Val Spec: {val_specificity:.4f} | "
            f"Val HS: {val_hs_score:.4f}"
        )
    
        # --------------------------------------------------
        # Save best model according to validation HS-score
        # --------------------------------------------------
    
        if val_hs_score > best_hs_score:
            best_hs_score = val_hs_score
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "mel_state_dict": mel.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_sensitivity": val_sensitivity,
                    "val_specificity": val_specificity,
                    "val_hs_score": val_hs_score,
                },
                best_checkpoint
            )
    
        print(f"✓ New best model saved: epoch {best_epoch}"
              f"HS-score = {best_hs_score:.4f}")

        # ----------------------------
        # Save the Last model 
        # ----------------------------
        
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "mel_state_dict": mel.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "val_hs_score": val_hs_score,
            },
            last_checkpoint
        )
        
        print( f"✓ Last model saved: epoch {epoch + 1}")

    # ======================================================
    # Load best validation model
    # ======================================================
    
    if best_checkpoint is None:
        raise RuntimeError("No best checkpoint was created.")
    
    checkpoint = torch.load(
        best_checkpoint,
        map_location=device,
        weights_only=False
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    mel.load_state_dict(checkpoint["mel_state_dict"])

    
    # ======================================================
    # Final TEST: Best Model
    # ======================================================
    
    test_loss, test_accuracy, \
        test_sensitivity, test_specificity, \
        test_hs_score, cm, \
        test_targets, test_predictions = _evaluate(
            model,
            mel,
            test_dl,
            device
        )
    
    print("\n==============================")
    print("FINAL TEST RESULTS of the Best")
    print("==============================")
    
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_accuracy:.4f}")
    print(f"Test Sensitivity: {test_sensitivity:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")
    print(f"Test HS-score:    {test_hs_score:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)


    # ======================================================
    # W&B TEST RESULTS: Best model
    # ======================================================
    
    wandb.log({
    
        "test/loss-best": test_loss,
        "test/accuracy-best": test_accuracy,
        "test/sensitivity-best": test_sensitivity,
        "test/specificity-best": test_specificity,
        "test/HS_score-best": test_hs_score,
    
        "test/confusion_matrix-best": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_targets,
            preds=test_predictions,
            class_names=["Normal", "Abnormal"]
        ),
    })

    # ======================================================
    # Load last validation model
    # ======================================================
    
    if last_checkpoint is None:
        raise RuntimeError("No last checkpoint was created.")
    checkpoint = torch.load(
        last_checkpoint,
        map_location=device,
        weights_only=False
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    mel.load_state_dict(checkpoint["mel_state_dict"])


    # ======================================================
    # Final TEST: Last Model
    # ======================================================
    
    test_loss, test_accuracy, \
        test_sensitivity, test_specificity, \
        test_hs_score, cm, \
        test_targets, test_predictions = _evaluate(
                                                    model,
                                                    mel,
                                                    test_dl,
                                                    device
                                                    )

    print("\n===============================")
    print("FINAL TEST RESULTS of the Last")
    print("===============================")
    
    print(f"Test Loss:        {test_loss:.4f}")
    print(f"Test Accuracy:    {test_accuracy:.4f}")
    print(f"Test Sensitivity: {test_sensitivity:.4f}")
    print(f"Test Specificity: {test_specificity:.4f}")
    print(f"Test HS-score:    {test_hs_score:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)


    
    # ======================================================
    # W&B TEST RESULTS: Last model
    # ======================================================
    
    wandb.log({
    
        "test/loss-last": test_loss,
        "test/accuracy-last": test_accuracy,
        "test/sensitivity-last": test_sensitivity,
        "test/specificity-last": test_specificity,
        "test/HS_score-last": test_hs_score,
    
        "test/confusion_matrix-last": wandb.plot.confusion_matrix(
            probs=None,
            y_true=test_targets,
            preds=test_predictions,
            class_names=["Normal", "Abnormal"]
        ),
    })


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _evaluate(model, mel, eval_loader, device):

    model.eval()
    mel.eval()

    targets = []
    predictions = []
    losses = []

    with torch.no_grad():

        pbar = tqdm(eval_loader)
        pbar.set_description("Evaluating")

        for batch in pbar:

            x, f, y = batch

            x = x.to(device)
            y = y.to(device)

            x = _mel_forward(x, mel)

            y_hat, _ = model(x)

            loss = F.cross_entropy(y_hat, y)

            preds = torch.argmax(y_hat, dim=1)

            targets.append(y.cpu())
            predictions.append(preds.cpu())
            losses.append(loss.cpu())

    targets = torch.cat(targets).numpy()
    predictions = torch.cat(predictions).numpy()

    loss = torch.stack(losses).mean().item()

    # ------------------------------------------
    # Confusion matrix
    # ------------------------------------------

    cm = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # ------------------------------------------
    # Metrics
    # ------------------------------------------

    eps = 1e-8

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn + eps)
    )

    sensitivity = (
        tp /
        (tp + fn + eps)
    )

    specificity = (
        tn /
        (tn + fp + eps)
    )

    hs_score = (
        2 *
        sensitivity *
        specificity /
        (sensitivity + specificity + eps)
    )

    return (
        loss,
        accuracy,
        sensitivity,
        specificity,
        hs_score,
        cm,
        targets,
        predictions
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="icbhi_mn10_as")
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=12)

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # lr schedule
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--warm_up_len', type=int, default=10)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=65)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    # dataset
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--clip_length', type=float, default=8.0)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=1337)

    args = parser.parse_args()
    train(args)
