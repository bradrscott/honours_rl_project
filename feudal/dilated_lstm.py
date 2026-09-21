# Dilated LSTM
# Source - lweitkamp/feudalnets-pytorch 
# https://github.com/lweitkamp/feudalnets-pytorch/blob/master/dilated_lstm.py
# Used by the manager in the Feudal Network. At each timestep only one slice
# of the hidden state is updated and the output is the mean across all radius slices — 
# this gives the manager a longer effective memory horizon than a standard LSTM.

# torch for the LSTM cell and tensor ops
import torch
import torch.nn as nn


class DilatedLSTM(nn.Module):

    # build the cell - radius slices of hidden_size and the base slice offsets
    def __init__(self, input_size, hidden_size, radius=10):
        super().__init__()
        self.radius = radius
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.index = torch.arange(0, radius * hidden_size, radius)
        self.dilation = 0

    # one step - update only the current dilation slice, pool the output
    # across all radius previous outputs
    def forward(self, state, hidden):
        d_idx = self.dilation_idx
        hx, cx = hidden[0].clone(), hidden[1].clone()

        # LSTM step, but only written into this timestep's slice
        hx[:, d_idx], cx[:, d_idx] = self.rnn(
            state, (hidden[0][:, d_idx], hidden[1][:, d_idx])
        )

        # pool the other radius -1 slice and average with the freshly-updated slice
        detached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detached_hx = detached_hx.view(
            detached_hx.shape[0], self.hidden_size, self.radius - 1
        )
        detached_hx = detached_hx.sum(-1)
        y = (hx[:, d_idx] + detached_hx) / self.radius
        return y, (hx, cx)

    # Returns all hidden-state indices except the current dilation slice
    # (its complement) — these are the other slices forward() pools.
    # Bug fix - the reference code subtracts 1 here, which wrongly shifts the
    # result onto the current slice instead of the others.
    def masked_idx(self, dilated_idx):

        # start with every position marked (nonzero), 1-indexed
        masked_idx = torch.arange(1, self.radius * self.hidden_size + 1)

        # zero out the current slice's positions so they drop out of nonzero()
        masked_idx[dilated_idx] = 0

        # keep exactly the positions that were not zeroed (the complement)
        masked_idx = masked_idx.nonzero().squeeze(-1)
        return masked_idx

    # Current dilation slice, advances by 1 each call (cycles every radius calls).
    # @property - lets forward() read this as self.dilation_idx
    @property
    def dilation_idx(self):
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.radius
        return dilation_idx
