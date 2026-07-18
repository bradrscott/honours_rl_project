# ── Feudal Network for Go ─────────────────────────────────────
#
# Architecture: Vezhnevets et al. (2017)
# "FeUdal Networks for Hierarchical Reinforcement Learning"
# https://arxiv.org/abs/1703.01161
#
# Adapted from: lweitkamp/feudalnets-pytorch (MIT)
# https://github.com/lweitkamp/feudalnets-pytorch
#
# Adaptations for Go:
#   - CNN perception over the (n_channels, N, N) board tensor — the SAME
#     conv trunk as the PPO baseline (the scaffold's default is also a
#     CNN; the earlier flat-board MLP was a departure). This keeps the
#     feature extractor identical to PPO so the only architectural
#     difference is the Manager-Worker hierarchy (RQ3 fairness).
#   - Action masking in Worker (only legal moves can be selected)
#   - Single worker (b=1, one environment)
#   - No Atari preprocessor (Go observations are already 0/1 planes)

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import cosine_similarity as d_cos, normalize

from dilated_lstm import DilatedLSTM


# ── Helpers ────────────────────────────────────────────────────

def init_hidden(num_workers, size, device='cpu', grad=False):
    """Initialise LSTM hidden state tensors."""
    hidden = (
        torch.zeros(num_workers, size, requires_grad=grad).to(device),
        torch.zeros(num_workers, size, requires_grad=grad).to(device),
    )
    return hidden


def weight_init(module):
    """Orthogonal weight initialisation (standard for RL)."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTMCell):
        nn.init.orthogonal_(module.weight_ih)
        nn.init.orthogonal_(module.weight_hh)


# ══════════════════════════════════════════════════════════════
# PERCEPTION MODULE
# Maps the flat board observation to a shared latent state z.
# ══════════════════════════════════════════════════════════════

class Perception(nn.Module):
    """
    CNN feature extractor — the SAME conv trunk as the PPO baseline.
    Input:  board tensor (b, n_channels, N, N)   e.g. (b, 17, 9, 9)
    Output: latent state z of size d

    The original scaffold uses a CNN Perception by default; this restores
    it (the earlier flat-board MLP was an adaptation) and makes the feature
    extractor identical to PPO so the only architectural difference between
    the two agents is the Manager-Worker hierarchy.
    """

    def __init__(self, board_size, n_channels, d, n_filters=32, n_layers=3):
        super().__init__()
        conv, in_ch = [], n_channels
        for _ in range(n_layers):
            conv += [nn.Conv2d(in_ch, n_filters, 3, padding=1), nn.ReLU()]
            in_ch = n_filters
        conv.append(nn.Flatten())
        self.conv = nn.Sequential(*conv)

        flat = n_filters * board_size * board_size
        self.fc = nn.Sequential(nn.Linear(flat, d), nn.ReLU())

    def forward(self, x):
        # x: (b, n_channels, N, N) -> (b, d)
        return self.fc(self.conv(x))


# ══════════════════════════════════════════════════════════════
# MANAGER
# Operates at lower temporal resolution (every c steps).
# Sets high-level goal directions in latent space.
# Receives extrinsic rewards (+1 win / -1 loss).
# ══════════════════════════════════════════════════════════════

class Manager(nn.Module):
    """
    Produces a normalised goal vector g_t at each timestep via
    a Dilated LSTM (dLSTM). The Manager is trained to predict
    directions in latent space that lead to high returns.
    """

    def __init__(self, c, d, r, eps, device):
        super().__init__()
        self.c      = c       # Time horizon
        self.d      = d       # Hidden dimension
        self.r      = r       # Dilation radius
        self.eps    = eps     # Exploration: random goal probability
        self.device = device

        self.Mspace = nn.Linear(self.d, self.d)
        self.Mrnn   = DilatedLSTM(self.d, self.d, self.r)
        self.critic = nn.Linear(self.d, 1)

    def forward(self, z, hidden, mask):
        state        = self.Mspace(z).relu()
        hidden       = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(state, hidden)
        value_est    = self.critic(goal_hat)

        # Normalise goal to unit vector
        goal = normalize(goal_hat)
        state = state.detach()

        # With probability eps, emit a random goal (exploration)
        if self.eps > torch.rand(1)[0]:
            goal = torch.randn_like(goal, requires_grad=False)

        return goal, hidden, state, value_est

    def state_goal_cosine(self, states, goals, masks):
        """
        Manager loss: cosine similarity between
        (s_{t+c} - s_t) and g_t.
        Used as the policy gradient for the Manager.
        """
        t          = self.c
        mask       = torch.stack(masks[t: t + self.c - 1]).prod(dim=0)
        cosine_dist = d_cos(states[t + self.c] - states[t], goals[t])
        cosine_dist = mask * cosine_dist.unsqueeze(-1)
        return cosine_dist


# ══════════════════════════════════════════════════════════════
# WORKER
# Operates at every timestep.
# Selects stone placements conditioned on the Manager's goal.
# Receives intrinsic rewards based on cosine similarity.
# ══════════════════════════════════════════════════════════════

class Worker(nn.Module):
    """
    Produces an action distribution over all n_actions
    (N*N board intersections + pass) conditioned on:
      - The current latent state z
      - The Manager's goal direction g_t

    Action masking ensures only legal moves are ever selected.
    """

    def __init__(self, b, c, d, k, n_actions, device):
        super().__init__()
        self.b         = b
        self.c         = c
        self.k         = k
        self.n_actions = n_actions
        self.device    = device

        self.Wrnn  = nn.LSTMCell(d, k * n_actions)
        self.phi   = nn.Linear(d, k, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(k * n_actions, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    def forward(self, z, goals, hidden, mask, action_mask=None):
        """
        Args:
            z:           latent state from Perception, shape (b, d)
            goals:       list of c+1 goal tensors, each (b, d)
            hidden:      LSTM hidden state
            mask:        episode done mask
            action_mask: boolean tensor (b, n_actions), True = legal

        Returns:
            dist:      Categorical distribution over legal actions
            hidden:    updated hidden state
            value_est: Worker value estimate
        """
        hidden     = (mask * hidden[0], mask * hidden[1])
        u, cx      = self.Wrnn(z, hidden)
        hidden     = (u, cx)

        # Sum goal directions across time horizon (detached — no end-to-end)
        goals_sum = torch.stack(goals).detach().sum(dim=0)
        w         = self.phi(goals_sum)

        value_est = self.critic(u)

        # Compute action logits
        u_reshaped = u.reshape(u.shape[0], self.k, self.n_actions)
        logits     = torch.einsum("bk, bka -> ba", w, u_reshaped)

        # Apply action mask — set illegal actions to -inf before softmax
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        dist = Categorical(logits=logits)
        return dist, hidden, value_est

    def intrinsic_reward(self, states, goals, masks):
        """
        Intrinsic reward for the Worker (Eq. 8 from FuN paper):
        Average cosine similarity between (s_t - s_{t-i}) and g_{t-i}
        for i = 1..c.

        This gives the Worker a learning signal across long games
        even before the terminal reward arrives.
        """
        t   = self.c
        r_i = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)

        for i in range(1, self.c + 1):
            r_i_t  = d_cos(states[t] - states[t - i], goals[t - i]).unsqueeze(-1)
            r_i   += (mask * r_i_t)
            mask   = mask * masks[t - i]

        return (r_i / self.c).detach()


# ══════════════════════════════════════════════════════════════
# FEUDAL NETWORK (full model)
# ══════════════════════════════════════════════════════════════

class FeudalNetwork(nn.Module):
    """
    Full FeUdal Network (FuN) for Go.

    Combines Perception, Manager, and Worker into one forward pass.
    The Manager sets goals every c steps; the Worker acts every step.

    Reference: Vezhnevets et al. (2017) https://arxiv.org/abs/1703.01161
    Adapted from: lweitkamp/feudalnets-pytorch
    """

    def __init__(
        self,
        board_size,
        n_channels,
        n_actions,
        hidden_dim_manager = 256,
        hidden_dim_worker  = 16,
        time_horizon       = 10,
        dilation           = 10,
        eps                = 0.1,
        num_workers        = 1,
        device             = 'cpu',
        n_filters          = 32,
        n_layers           = 3,
    ):
        super().__init__()

        self.b       = num_workers
        self.c       = time_horizon
        self.d       = hidden_dim_manager
        self.k       = hidden_dim_worker
        self.r       = dilation
        self.device  = device

        self.percept = Perception(board_size, n_channels, self.d, n_filters, n_layers)
        self.manager = Manager(self.c, self.d, self.r, eps, device)
        self.worker  = Worker(self.b, self.c, self.d, self.k, n_actions, device)

        # Persistent hidden states across rollout steps
        self.hidden_m = init_hidden(num_workers, self.r * self.d,  device=device, grad=True)
        self.hidden_w = init_hidden(num_workers, self.k * n_actions, device=device, grad=True)

        self.to(device)
        self.apply(weight_init)

    def forward(self, x, goals, states, mask, action_mask=None, save=True):
        """
        Full forward pass.

        Args:
            x:           board observation tensor, shape (b, n_channels, N, N)
            goals:       FIFO list of goal tensors, length 2c+1
            states:      FIFO list of state tensors, length 2c+1
            mask:        episode done mask, shape (b, 1)
            action_mask: boolean tensor (b, n_actions), True = legal
            save:        whether to update persistent hidden states

        Returns:
            dist:       Categorical over legal actions
            goals:      updated goals list
            states:     updated states list
            value_m:    Manager value estimate
            value_w:    Worker value estimate
        """
        z = self.percept(x)

        goal, hidden_m, state, value_m = self.manager(z, self.hidden_m, mask)

        # Maintain FIFO lists of size 2c+1
        if len(goals) > (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)
        goals.append(goal)
        states.append(state)  # states never carry gradients

        # Worker sees only the first c+1 goals
        dist, hidden_w, value_w = self.worker(
            z, goals[:self.c + 1], self.hidden_w, mask, action_mask
        )

        if save:
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w

        return dist, goals, states, value_m, value_w

    def get_next_values(self, x, goals, states, mask):
        """Bootstrap values for the final step (no hidden state update)."""
        with torch.no_grad():
            _, _, _, value_m, value_w = self.forward(
                x, goals, states, mask, save=False
            )
        return value_m, value_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)

    def repackage_hidden(self):
        """Detach hidden states to prevent gradient flow across rollouts."""
        self.hidden_m = [h.detach() for h in self.hidden_m]
        self.hidden_w = [h.detach() for h in self.hidden_w]

    def init_obj(self):
        """Initialise goals, states and masks lists."""
        template = torch.zeros(self.b, self.d).to(self.device)
        goals    = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        states   = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        masks    = [torch.ones(self.b, 1).to(self.device) for _ in range(2 * self.c + 1)]
        return goals, states, masks


# ══════════════════════════════════════════════════════════════
# FEUDAL LOSS
# ══════════════════════════════════════════════════════════════

class Storage:
    """
    Simple experience storage for the n-step rollout.
    Source: lweitkamp/feudalnets-pytorch (MIT)
    """

    def __init__(self, size, keys=None):
        self.keys = keys or []
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def normalize(self, keys):
        for key in keys:
            k = torch.stack(getattr(self, key))
            k = (k - k.mean()) / (k.std() + 1e-10)
            setattr(self, key, [i for i in k])

    def stack(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.stack(x, dim=0), data)


def feudal_loss(storage, next_v_m, next_v_w, gamma_m, gamma_w, alpha,
                entropy_coef, num_steps, gae_lambda):
    """
    FuN loss with correctly-masked GAE(lambda) — the same low-variance credit
    assignment PPO uses (PPO reaches ~70% vs aggressive; raw Monte-Carlo returns
    left feudal at 0). Worker advantage is GAE on the (extrinsic+alpha*intrinsic)
    reward; manager transition-PG (state-goal cosine) is weighted by GAE on the
    extrinsic reward.

    CRITICAL: GAE is gated by `nt` (nonterminal flag for THIS step, 0 on the
    terminal step), NOT by `m` (the hidden-reset mask, 0 on the step AFTER a
    terminal). An earlier GAE used `m` and leaked the next episode's value
    across the boundary at every game end -> wild oscillation. `nt` fixes that.
    """
    (rewards, rewards_intrinsic, value_m, value_w, logps, entropy,
     state_goal_cosines, nts) = storage.stack(
        ['r', 'r_i', 'v_m', 'v_w', 'logp', 'entropy', 's_goal_cos', 'nt'])

    v_w_det = value_w.detach()
    v_m_det = value_m.detach()

    adv_w_list = [None] * num_steps
    adv_m_list = [None] * num_steps
    gae_w = torch.zeros_like(next_v_w)
    gae_m = torch.zeros_like(next_v_m)
    next_vw, next_vm = next_v_w, next_v_m

    for i in reversed(range(num_steps)):
        nt = nts[i]                      # 0 if step i ENDED the episode, else 1

        # Worker: reward = extrinsic + alpha * intrinsic
        rw       = rewards[i] + alpha * rewards_intrinsic[i]
        delta_w  = rw + gamma_w * next_vw * nt - v_w_det[i]
        gae_w    = delta_w + gamma_w * gae_lambda * nt * gae_w
        adv_w_list[i] = gae_w
        next_vw  = v_w_det[i]

        # Manager: reward = extrinsic only
        delta_m  = rewards[i] + gamma_m * next_vm * nt - v_m_det[i]
        gae_m    = delta_m + gamma_m * gae_lambda * nt * gae_m
        adv_m_list[i] = gae_m
        next_vm  = v_m_det[i]

    adv_w = torch.stack(adv_w_list, dim=0)
    adv_m = torch.stack(adv_m_list, dim=0)

    returns_w = adv_w + v_w_det           # value-regression targets
    returns_m = adv_m + v_m_det

    adv_w_norm = (adv_w - adv_w.mean()) / (adv_w.std() + 1e-8)
    adv_m_norm = (adv_m - adv_m.mean()) / (adv_m.std() + 1e-8)

    loss_worker  = (logps * adv_w_norm.detach()).mean()
    loss_manager = (state_goal_cosines * adv_m_norm.detach()).mean()

    value_w_loss = 0.5 * (returns_w - value_w).pow(2).mean()
    value_m_loss = 0.5 * (returns_m - value_m).pow(2).mean()

    entropy = entropy.mean()

    loss = (- loss_worker
            - loss_manager
            + value_w_loss
            + value_m_loss
            - entropy_coef * entropy)

    metrics = {
        'loss/total':            loss.item(),
        'loss/worker':           loss_worker.item(),
        'loss/manager':          loss_manager.item(),
        'loss/value_worker':     value_w_loss.item(),
        'loss/value_manager':    value_m_loss.item(),
        'worker/entropy':        entropy.item(),
        'worker/advantage':      adv_w.mean().item(),
        'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
        'manager/cosines':       state_goal_cosines.mean().item(),
        'manager/advantage':     adv_m.mean().item(),
    }

    return loss, metrics
