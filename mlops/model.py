import torch
import torch.nn as nn

class NewModel(nn.Module):
    def __init__(
        self, input_features=4, hidden_layer1=25, hidden_layer2=30, hidden_layer3=50, output_features=3
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_features, hidden_layer1),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.LogSigmoid(),
            nn.Linear(hidden_layer2, hidden_layer3),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer3, output_features)
        )

    def forward(self, x):
        x = self.model(x)
        return x