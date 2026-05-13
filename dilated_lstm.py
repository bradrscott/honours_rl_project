# ── Dilated LSTM ──────────────────────────────────────────────
# Source: lweitkamp/feudalnets-pytorch (MIT)
# https://github.com/lweitkamp/feudalnets-pytorch/blob/master/dilated_lstm.py
#
# Used by the Manager in the Feudal Network.
# At each timestep only one slice of the hidden state is updated,
# and the output is the mean across all r slices — this gives the
# Manager a longer effective memory horizon than a standard LSTM.

import torch
import torch.nn as nn


class DilatedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, radius=10):
        super().__init__()
        self.radius      = radius
        self.hidden_size = hidden_size
        self.rnn         = nn.LSTMCell(input_size, hidden_size)
        self.index       = torch.arange(0, radius * hidden_size, radius)
        self.dilation    = 0

    def forward(self, state, hidden):
        """Only the current dilation slice is updated; output is
        pooled across all r previous outputs."""
        d_idx    = self.dilation_idx
        hx, cx   = hidden

        hx[:, d_idx], cx[:, d_idx] = self.rnn(
            state, (hx[:, d_idx], cx[:, d_idx])
        )

        detached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(
            detached_hx.shape[0], self.hidden_size, self.radius - 1
        )
        detached_hx = detached_hx.sum(-1)

        y = (hx[:, d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    def masked_idx(self, dilated_idx):
        """All indices except the current dilation slice."""
        masked_idx              = torch.arange(1, self.radius * self.hidden_size + 1)
        masked_idx[dilated_idx] = 0
        masked_idx              = masked_idx.nonzero()
        masked_idx              = masked_idx - 1
        return masked_idx

    @property
    def dilation_idx(self):
        """Current dilation slice, advances by 1 each call."""
        dilation_idx   = self.dilation + self.index
        self.dilation  = (self.dilation + 1) % self.radius
        return dilation_idx
