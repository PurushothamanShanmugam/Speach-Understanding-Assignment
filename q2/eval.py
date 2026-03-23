import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class SpeakerDataset(Dataset):
    def __init__(self, csv_path, audio_base_dir, max_length=16000):
        self.df = pd.read_csv(csv_path)
        self.audio_base_dir = audio_base_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_base_dir, row["file_name"])
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]

        x = torch.tensor(audio, dtype=torch.float32)
        speaker = torch.tensor(int(row["speaker_id"]), dtype=torch.long)
        env = torch.tensor(int(row["environment_id"]), dtype=torch.long)
        return x, speaker, env


class BaselineSpeakerModel(nn.Module):
    def __init__(self, num_speakers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16000, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_speakers)
        )

    def forward(self, x):
        return self.net(x)


class ReducedDisentangledModel(nn.Module):
    def __init__(self, num_speakers, num_envs=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(16000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.speaker_head = nn.Linear(256, num_speakers)
        self.env_head = nn.Linear(256, num_envs)

    def forward(self, x):
        z = self.encoder(x)
        speaker_logits = self.speaker_head(z)
        env_logits = self.env_head(z)
        return speaker_logits, env_logits


def evaluate_baseline(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, speaker, _ in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(speaker.numpy())

    return y_true, y_pred


def evaluate_disentangled(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, speaker, _ in loader:
            x = x.to(device)
            speaker_logits, _ = model(x)
            pred = torch.argmax(speaker_logits, dim=1).cpu().numpy()

            y_pred.extend(pred)
            y_true.extend(speaker.numpy())

    return y_true, y_pred


def main():
    with open("q2/configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["results_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeakerDataset(
        config["dataset_csv"],
        config["audio_base_dir"],
        max_length=config["max_length"]
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)

    baseline = BaselineSpeakerModel(config["num_speakers"]).to(device)
    disentangled = ReducedDisentangledModel(config["num_speakers"]).to(device)

    baseline.load_state_dict(torch.load(config["baseline_save_path"], map_location=device))
    disentangled.load_state_dict(torch.load(config["model_save_path"], map_location=device))

    b_true, b_pred = evaluate_baseline(baseline, loader, device)
    d_true, d_pred = evaluate_disentangled(disentangled, loader, device)

    b_acc = accuracy_score(b_true, b_pred)
    d_acc = accuracy_score(d_true, d_pred)

    results_df = pd.DataFrame({
        "model": ["baseline", "reduced_disentangled"],
        "accuracy": [b_acc, d_acc]
    })
    results_df.to_csv(os.path.join(config["results_dir"], "eval_results.csv"), index=False)

    print(results_df)

    cm = confusion_matrix(d_true, d_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Reduced Disentangled Model - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], "confusion_matrix.png"))
    plt.close()

    print("Evaluation complete.")


if __name__ == "__main__":
    main()