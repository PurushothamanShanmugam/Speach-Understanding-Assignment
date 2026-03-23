import torch
import torch.nn as nn


class PrivacyPreservingModule(nn.Module):
    """
    Beginner-friendly transformation module.
    This is a simple neural transformation block, not a full voice conversion system.
    """
    def __init__(self, input_dim=16000, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = PrivacyPreservingModule()
    dummy_input = torch.randn(2, 16000)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)