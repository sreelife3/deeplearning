# -*- coding: utf-8 -*-
"""Four architectures on matched hyperparameters.

Matched is the whole point. Same hidden width, same depth, same dropout. If the
CNN wins because it was given more capacity, the study has measured nothing.
"""
import torch
import torch.nn as nn

SEQ_LEN, N_CHANNELS, N_CLASSES = 128, 9, 6
HIDDEN, LAYERS, DROPOUT = 64, 2, 0.4


class _Recurrent(nn.Module):
    """RNN, LSTM and GRU differ by one attribute, so they share everything else
    and no accidental asymmetry can creep in between them."""

    CELL = None

    def __init__(self, input_size=N_CHANNELS, hidden=HIDDEN, layers=LAYERS,
                 classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.rnn = self.CELL(
            input_size=input_size, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x):
        # x: (batch, time, channels)
        out, _ = self.rnn(x)
        return self.fc(self.drop(out[:, -1, :]))     # last time step


class RNN(_Recurrent):
    CELL = nn.RNN


class LSTM(_Recurrent):
    CELL = nn.LSTM


class GRU(_Recurrent):
    CELL = nn.GRU


class CNN1D(nn.Module):
    """Convolution over TIME, not over channels.

    Conv1d wants (batch, channels, time); the data arrives as (batch, time,
    channels). Get the transpose wrong and the model still trains, still
    converges to something plausible, and is quietly convolving across the nine
    sensors instead of across the 128 readings. Hence the assert.
    """

    def __init__(self, input_size=N_CHANNELS, hidden=HIDDEN,
                 classes=N_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, classes)

    def forward(self, x):
        assert x.shape[1] == SEQ_LEN and x.shape[2] == N_CHANNELS, (
            "expected (batch, %d, %d), got %s -- the transpose is the classic "
            "silent bug here" % (SEQ_LEN, N_CHANNELS, tuple(x.shape)))
        h = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.fc(self.drop(h))


REGISTRY = {"rnn": RNN, "lstm": LSTM, "gru": GRU, "cnn1d": CNN1D}


def build(name):
    if name not in REGISTRY:
        raise KeyError("unknown model %r, expected one of %s"
                       % (name, list(REGISTRY)))
    return REGISTRY[name]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
