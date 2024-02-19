import numpy as np
import torch
import torch.nn as nn
from src.convolution import ConvBlock
from __init__ import volta_logger


class TSAutoEncoder(nn.Module):
    def __init__(self,
                 nb_layers,
                 ks_min,
                 ks_max,
                 ks_scaling = None,
                 logger = volta_logger):
        super(TSAutoEncoder, self).__init__()

        self.ks_min = ks_min
        self.ks_max = ks_max
        self.nb_layers = nb_layers

        if ks_scaling is None:
            self.ks_scaling = np.round(np.exp((1. / nb_layers) * np.log(ks_max / ks_min)), 2)
            logger.info('Kernel scaling factor has been set to %f', self.ks_scaling)
        else:
            self.ks_scaling = ks_scaling

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
