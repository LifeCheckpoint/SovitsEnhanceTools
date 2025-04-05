import torch
import torch.nn as nn

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Conv1d(512, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(256, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 32, 3, padding=1),
            nn.Conv1d(32, 24, 1)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(24, 32, 1),
            nn.ReLU(),
            
            nn.ConvTranspose1d(32, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.ConvTranspose1d(256, 512, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Conv1d(512, 256, 3, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)