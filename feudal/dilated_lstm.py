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
        hx, cx   = hidden[0].clone(), hidden[1].clone()

        hx[:, d_idx], cx[:, d_idx] = self.rnn(
            state, (hidden[0][:, d_idx], hidden[1][:, d_idx])
        )

        detached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(
            detached_hx.shape[0], self.hidden_size, self.radius - 1
        )
        detached_hx = detached_hx.sum(-1)

        y = (hx[:, d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    def masked_idx(self, dilated_idx):
        """All indices EXCEPT the current dilation slice.

        Build [1..R*H] (all nonzero), zero the positions in dilated_idx, then
        take nonzero() — which returns exactly the POSITIONS that were NOT
        zeroed, i.e. the complement of dilated_idx. These positions are already
        0-indexed into the hidden vector, so they are used directly.

        NOTE (bug fix): the reference (lweitkamp/feudalnets-pytorch) subtracts 1
        here. That is wrong — nonzero() already returns positions, so `- 1`
        shifts every index down by one (and wraps 0 -> -1 onto the last unit),
        which makes the pool collect the CURRENT slice instead of the others
        and defeats the dilated LSTM's whole purpose. We drop the `- 1` and
        squeeze the trailing dim so `hx[:, masked_idx]` reshapes cleanly to
        (batch, hidden_size, radius-1) grouped per hidden unit.
        """
        masked_idx              = torch.arange(1, self.radius * self.hidden_size + 1)
        masked_idx[dilated_idx] = 0
        masked_idx              = masked_idx.nonzero().squeeze(-1)
        return masked_idx

    @property
    def dilation_idx(self):
        """Current dilation slice, advances by 1 each call."""
        dilation_idx   = self.dilation + self.index
        self.dilation  = (self.dilation + 1) % self.radius
        return dilation_idx
