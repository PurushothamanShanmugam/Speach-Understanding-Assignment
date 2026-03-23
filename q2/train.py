import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


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


def train_baseline(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, speaker, _ in loader:
        x, speaker = x.to(device), speaker.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, speaker)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_disentangled(model, loader, criterion, optimizer, device, lambda_env=0.3):
    model.train()
    total_loss = 0.0
    for x, speaker, env in loader:
        x = x.to(device)
        speaker = speaker.to(device)
        env = env.to(device)

        optimizer.zero_grad()
        speaker_logits, env_logits = model(x)

        speaker_loss = criterion(speaker_logits, speaker)
        env_loss = criterion(env_logits, env)

        loss = speaker_loss - lambda_env * env_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    with open("q2/configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(config["model_save_path"]), exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpeakerDataset(
        config["dataset_csv"],
        config["audio_base_dir"],
        max_length=config["max_length"]
    )

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    baseline = BaselineSpeakerModel(config["num_speakers"]).to(device)
    disentangled = ReducedDisentangledModel(config["num_speakers"]).to(device)

    criterion = nn.CrossEntropyLoss()

    baseline_opt = optim.Adam(baseline.parameters(), lr=config["lr"])
    disentangled_opt = optim.Adam(disentangled.parameters(), lr=config["lr"])

    baseline_history = []
    disentangled_history = []

    for epoch in range(config["epochs"]):
        b_loss = train_baseline(baseline, loader, criterion, baseline_opt, device)
        d_loss = train_disentangled(disentangled, loader, criterion, disentangled_opt, device)

        baseline_history.append(b_loss)
        disentangled_history.append(d_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} | Baseline Loss: {b_loss:.4f} | Reduced Disentangled Loss: {d_loss:.4f}")

    torch.save(baseline.state_dict(), config["baseline_save_path"])
    torch.save(disentangled.state_dict(), config["model_save_path"])

    pd.DataFrame({
        "epoch": list(range(1, config["epochs"] + 1)),
        "baseline_loss": baseline_history,
        "disentangled_loss": disentangled_history
    }).to_csv(os.path.join(config["results_dir"], "training_history.csv"), index=False)

    print("Training complete. Models and history saved.")


if __name__ == "__main__":
    main()