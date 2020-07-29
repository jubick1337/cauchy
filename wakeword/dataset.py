from pathlib import Path

import pandas as pd
import torch
import torchaudio


class WakeWordDataSet(torch.utils.data.Dataset):
    def __init__(self, csv_file: Path, sample_rate: int):
        self._data = pd.read_csv(csv_file, header=None)
        self._sample_rate = sample_rate
        self._mfcc = torchaudio.transforms.MFCC(sample_rate)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self._data.iloc[idx]
        waveform, sample_rate = torchaudio.load(data[0])
        label = data[1]
        if sample_rate > self._sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self._sample_rate)(waveform)
        mfcc = self._mfcc(waveform)
        return mfcc, label
