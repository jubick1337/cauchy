import torch

import torch.nn as nn


class WakeWordDetector(nn.Module):
    def __init__(self, seq_length: int, hidden_size: int, num_layers: int):
        super(WakeWordDetector, self).__init__()
        self._gru = nn.GRU(seq_length, hidden_size, num_layers=num_layers, dropout=0.25)
        self._classifier = nn.Linear(num_layers * hidden_size, out_features=1)

    def forward(self, x: torch.Tensor):
        # x => batch_size, seq_len, features
        x = x.transpose(0, 1)
        _, x = self._gru(x)
        x = x.transpose(0, 1)
        x = x.flatten(1)
        x = self._classifier(x)
        return x.squeeze_(-1)
