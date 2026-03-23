import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FairSpeechDataset(Dataset):
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
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        group = torch.tensor(int(row["group_id"]), dtype=torch.long)
        return x, label, group


class SimpleSpeechClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16000, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def fairness_loss_fn(logits, labels, groups):
    ce = nn.CrossEntropyLoss(reduction="none")
    per_sample_loss = ce(logits, labels)

    unique_groups = torch.unique(groups)
    group_losses = []

    for g in unique_groups:
        mask = (groups == g)
        if torch.sum(mask) > 0:
            group_losses.append(torch.mean(per_sample_loss[mask]))

    if len(group_losses) < 2:
        return torch.tensor(0.0, device=logits.device)

    fairness_penalty = torch.max(torch.stack(group_losses)) - torch.min(torch.stack(group_losses))
    return fairness_penalty


def train():
    dataset = FairSpeechDataset("data/q3_train.csv", "data/q3_audio")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSpeechClassifier(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()

    os.makedirs("outputs/q3/fair_training", exist_ok=True)

    for epoch in range(5):
        model.train()
        total_loss = 0.0

        for x, labels, groups in loader:
            x = x.to(device)
            labels = labels.to(device)
            groups = groups.to(device)

            optimizer.zero_grad()
            logits = model(x)

            main_loss = ce_loss(logits, labels)
            fair_loss = fairness_loss_fn(logits, labels, groups)

            loss = main_loss + 0.5 * fair_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/5, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "outputs/q3/fair_training/fair_model.pt")
    print("Fairness-aware training complete.")


if __name__ == "__main__":
    train()