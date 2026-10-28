from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def patient_id_from_filename(filename):
    return Path(filename).stem.split("_")[0]


def pad_or_truncate(waveform, target_length):
    if len(waveform) < target_length:
        return np.pad(waveform, (0, target_length - len(waveform)))
    return waveform[:target_length]


def make_train_val_patient_split(patient_ids, val_fraction=0.2, seed=1337):
    patient_ids = sorted(list(set(patient_ids)))

    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * val_fraction))

    val_patients = set(patient_ids[:n_val])
    train_patients = set(patient_ids[n_val:])

    return train_patients, val_patients


class ICBHIDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        split,
        resample_rate=32000,
        clip_length=8.0,
        val_fraction=0.2,
        seed=1337,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "audios_cycles"
        self.split = split
        self.resample_rate = resample_rate
        self.clip_samples = int(resample_rate * clip_length)

        meta_csv = self.dataset_dir / "meta.csv"

        df = pd.read_csv(meta_csv)
        df["fold"] = df["fold"].str.lower()

        official_train_df = df[df["fold"] == "train"].copy()
        official_test_df = df[df["fold"] == "test"].copy()

        official_train_df["patient_id"] = official_train_df["filename"].apply(
            patient_id_from_filename
        )

        train_patients, val_patients = make_train_val_patient_split(
            official_train_df["patient_id"].tolist(),
            val_fraction=val_fraction,
            seed=seed,
        )

        if split.lower() == "train":
            df = official_train_df[
                official_train_df["patient_id"].isin(train_patients)
            ]

        elif split.lower() == "val":
            df = official_train_df[
                official_train_df["patient_id"].isin(val_patients)
            ]

        elif split.lower() == "test":
            df = official_test_df

        else:
            raise ValueError("split must be one of: train, val, test")

        self.items = []

        for _, row in df.iterrows():
            wav_path = self.audio_dir / row["filename"]
            label = int(row["target"])

            if not wav_path.exists():
                continue

            self.items.append((wav_path, label))

        if len(self.items) == 0:
            raise RuntimeError(f"No ICBHI files found for split: {split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        wav_path, label = self.items[index]

        waveform, _ = librosa.load(
            wav_path,
            sr=self.resample_rate,
            mono=True,
        )

        waveform = waveform.astype(np.float32)
        waveform = pad_or_truncate(waveform, self.clip_samples)

        return waveform.reshape(1, -1), wav_path.name, label