"""Attention Bi-LSTM baseline of the BdSLW60 dataset paper (ported)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Attention-based Bi-LSTM baseline of the BdSLW60 dataset paper
    (Rubaiyeat et al., MTAP 2025; arXiv 2402.08635, Fig. 9), ported from the
    authors' Keras code (github.com/hasanssl/BdSLW60_Code,
    pipeline_LSTM_225_FLIPPED_ST5_B64_D5MAX.ipynb):

        x -> BiLSTM(64) -> Dropout(0.3) ------------------------+ (value/key)
        x -> BiLSTM(64) ---------------------+ (query)          |
                                             +-> Attention(use_scale) <-+
        concat[BiLSTM_2, attn] -> Flatten -> Dense(64, SiLU) -> Dropout(0.3) -> Dense(C)

    Like the original, the sequence is zero-padded/cropped to a fixed length
    (their 164 frames; here the shared-recipe window, 120) because the Flatten
    head needs a fixed T, and the raw input is multiplied by ``input_scale``
    (their ``npy*100``). No normalisation layer, as in the original.
    """

    def __init__(self, num_class=60, num_point=27, num_person=1, graph=None, graph_args=dict(),
                 in_channels=3, window_size=120, hidden=64, dense=64, dropout=0.3, input_scale=100.0):
        super(Model, self).__init__()
        self.window_size = window_size
        self.input_scale = input_scale
        input_dim = num_point * in_channels * num_person
        self.lstm_v = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.lstm_q = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.drop_v = nn.Dropout(dropout)
        # Keras Attention(use_scale=True): learnable scalar on the dot-product scores, init 1.
        self.att_scale = nn.Parameter(torch.ones(1))
        self.fc = nn.Linear(window_size * 4 * hidden, dense)
        self.drop_fc = nn.Dropout(dropout)
        self.out = nn.Linear(dense, num_class)

    def forward(self, x, keep_prob=None):  # keep_prob: main.py calling convention (unused)
        N, C, T, V, M = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(N, T, C * V * M) * self.input_scale
        if T < self.window_size:
            x = F.pad(x, (0, 0, 0, self.window_size - T))
        elif T > self.window_size:
            x = x[:, :self.window_size]

        v = self.drop_v(self.lstm_v(x)[0])          # (N, T, 2h)
        q = self.lstm_q(x)[0]                       # (N, T, 2h)
        w = torch.softmax(self.att_scale * torch.bmm(q, v.transpose(1, 2)), dim=-1)
        att = torch.bmm(w, v)                       # (N, T, 2h)

        h = torch.cat([q, att], dim=-1).flatten(1)  # (N, T*4h)
        h = self.drop_fc(F.silu(self.fc(h)))
        return self.out(h)
