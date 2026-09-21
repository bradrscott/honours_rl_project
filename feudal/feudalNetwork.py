# Feudal Network for Go
# Adapted from - lweitkamp/feudalnets-pytorch 
# https://github.com/lweitkamp/feudalnets-pytorch

# Adaptations for Go:
#   - CNN perception over the (n_channels, N, N) board
#   - Action masking in worker (only legal moves can be selected)
#   - Single worker
#   - No Atari preprocessor (Go observations are already 0/1 planes)

# torch for the network layers, Categorical for the action distribution,
# cosine_similarity/normalize for the goal-direction maths
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import cosine_similarity as d_cos, normalize

# the Manager's dilated LSTM (long-horizon memory), defined in dilated_lstm.py
from feudal.dilated_lstm import DilatedLSTM

# build a zeroed (hidden state, cell state) LSTM hidden-state pair
def init_hidden(num_workers, size, device='cpu', grad=False):
    hidden = (
        torch.zeros(num_workers, size, requires_grad=grad).to(device),
        torch.zeros(num_workers, size, requires_grad=grad).to(device),
    )
    return hidden


# orthogonal weight initialisation, zero bias
def weight_init(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTMCell):
        nn.init.orthogonal_(module.weight_ih)
        nn.init.orthogonal_(module.weight_hh)

# perception section - maps the board observation to a shared latent state z.
class Perception(nn.Module):

    # CNN feature
    # Input - board tensor
    # Output - latent state of a certain size
    def __init__(self, board_size, n_channels, d, n_filters=32, n_layers=3):
        super().__init__()

        # conv trunk over the board planes
        conv, in_ch = [], n_channels
        for _ in range(n_layers):
            conv += [nn.Conv2d(in_ch, n_filters, 3, padding=1), nn.ReLU()]
            in_ch = n_filters
        conv.append(nn.Flatten())
        self.conv = nn.Sequential(*conv)

        # project the flattened conv features down to the latent size d
        flat = n_filters * board_size * board_size
        self.fc = nn.Sequential(nn.Linear(flat, d), nn.ReLU())


    # run the board through the conv trunk then the fully connected layer to get the latent state
    def forward(self, x):
        return self.fc(self.conv(x))

# Manager
# Operates at lower temporal resolution (every manger horizon steps).
# Sets high-level goal directions in latent space.
# Receives extrinsic rewards (+1 win, -1 loss).

class Manager(nn.Module):

    # Produces a normalised goal vector at each timestep via a Dilated
    # LSTM (dLSTM). The manager is trained to predict directions in latent
    # space that lead to high returns.
    def __init__(self, c, d, r, eps, device):
        super().__init__()
        self.c = c # time horizon
        self.d = d # hidden dimension
        self.r = r # dilation radius
        self.eps = eps # exploration: random goal probability
        self.device = device
        self.Mspace = nn.Linear(self.d, self.d)
        self.Mrnn   = DilatedLSTM(self.d, self.d, self.r)
        self.critic = nn.Linear(self.d, 1)


    # one manager step: latent state, hidden (its LSTM state), mask
    # (episode-done) - returns (goal, updated hidden, state, value estimate)
    def forward(self, z, hidden, mask):

        # project into manager space, reset hidden state at episode boundaries
        state = self.Mspace(z).relu()
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(state, hidden)
        value_est = self.critic(goal_hat)

        # normalise the goal to a unit vector
        goal = normalize(goal_hat)
        state = state.detach()

        # with probability eps, emit a random goal (exploration)
        if self.eps > torch.rand(1)[0]:
            goal = torch.randn_like(goal, requires_grad=False)
        return goal, hidden, state, value_est

    # Manager loss term - cosine similarity between the managers horizon steps and goal vector,
    # used as the policy gradient for the Manager.
    def state_goal_cosine(self, states, goals, masks):
        t = self.c
        mask = torch.stack(masks[t: t + self.c - 1]).prod(dim=0)
        cosine_dist = d_cos(states[t + self.c] - states[t], goals[t])
        cosine_dist = mask * cosine_dist.unsqueeze(-1)
        return cosine_dist

# worker
# Operates at every timestep.
# Selects stone placements conditioned on the manager's goal.
# Receives intrinsic rewards based on cosine similarity.
class Worker(nn.Module):

    # Produces an action distribution over all n_actions (N*N board
    # intersections + pass), conditioned on the latent state and the
    # manager's goal direction vector. Action masking ensures only legal
    # moves are ever selected.
    def __init__(self, b, c, d, k, n_actions, device):
        super().__init__()
        self.b = b
        self.c = c
        self.k = k
        self.n_actions = n_actions
        self.device = device

        self.Wrnn = nn.LSTMCell(d, k * n_actions)
        self.phi = nn.Linear(d, k, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(k * n_actions, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )

    # latent state - (batch size, hidden dimension), goals - list of manger horizon + 1 goal tensors (batch size, hidden dimension),
    # mask - episode-done mask, action_mask: (batch size, n_actions) boolean, True = legal.
    # Returns - (distance over legal actions, updated hidden state, worker value).
    def forward(self, z, goals, hidden, mask, action_mask=None):

        # reset hidden state at episode boundaries, step the worker LSTM
        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.Wrnn(z, hidden)
        hidden = (u, cx)

        # sum goal directions across the time horizon (detached — no end-to-end)
        goals_sum = torch.stack(goals).detach().sum(dim=0)
        w = self.phi(goals_sum)

        value_est = self.critic(u)

        # compute action logits by combining the goal embedding with the LSTM output
        u_reshaped = u.reshape(u.shape[0], self.k, self.n_actions)
        logits = torch.einsum("bk, bka -> ba", w, u_reshaped)

        # apply the action mask — illegal actions get zero probability
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        dist = Categorical(logits=logits)
        return dist, hidden, value_est

    # Intrinsic reward for the worker - average cosine similarity between 
    # For each of the last horizon steps, compare the direction the state has moved since that step,
    # against the goal that was set at that step.
    # Gives the worker a learning signal across long games even before the terminal reward arrives.
    def intrinsic_reward(self, states, goals, masks):
        t = self.c
        r_i = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)

        # sum the cosine similarity over the last horizon goal/state pairs,
        # stopping early across an episode boundary (via the mask)
        for i in range(1, self.c + 1):
            r_i_t = d_cos(states[t] - states[t - i], goals[t - i]).unsqueeze(-1)
            r_i += (mask * r_i_t)
            mask = mask * masks[t - i]
        return (r_i / self.c).detach()



# feudal network
class FeudalNetwork(nn.Module):

    # Full FeUdal Network (FuN) for Go - combines perception, manager and
    # worker into one forward pass. The manager sets goals every horizon steps,
    # the worker acts every step.
    def __init__(
        self,
        board_size,
        n_channels,
        n_actions,
        hidden_dim_manager = 256,
        hidden_dim_worker = 16,
        time_horizon = 10,
        dilation = 10,
        eps = 0.1,
        num_workers = 1,
        device = 'cpu',
        n_filters = 32,
        n_layers = 3,
    ):
        super().__init__()

        self.b = num_workers
        self.c = time_horizon
        self.d = hidden_dim_manager
        self.k = hidden_dim_worker
        self.r = dilation
        self.device = device

        # the three submodules - shared perception, manager, worker
        self.percept = Perception(board_size, n_channels, self.d, n_filters, n_layers)
        self.manager = Manager(self.c, self.d, self.r, eps, device)
        self.worker  = Worker(self.b, self.c, self.d, self.k, n_actions, device)

        # persistent hidden states, carried across rollout steps
        self.hidden_m = init_hidden(num_workers, self.r * self.d,  device=device, grad=True)
        self.hidden_w = init_hidden(num_workers, self.k * n_actions, device=device, grad=True)
        self.to(device)
        self.apply(weight_init)

    # Full forward pass for one step - runs perception, then the manager, then
    # the worker, updating the goal/state history along the way.
    def forward(self, x, goals, states, mask, action_mask=None, save=True):
        z = self.percept(x)
        goal, hidden_m, state, value_m = self.manager(z, self.hidden_m, mask)

        # maintain the FIFO lists at size 2c+1
        if len(goals) > (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)
        goals.append(goal)
        states.append(state)  

        # the worker only sees the first c+1 goals
        dist, hidden_w, value_w = self.worker(
            z, goals[:self.c + 1], self.hidden_w, mask, action_mask
        )

        # only persist the hidden state when asked 
        if save:
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w
        return dist, goals, states, value_m, value_w

    # Bootstrap values for the final step, without disturbing the network's
    # persistent state (runs on copies of the lists, restores the dilation counter afterwards).
    def get_next_values(self, x, goals, states, mask):
        saved_dilation = self.manager.Mrnn.dilation
        goals_copy, states_copy = list(goals), list(states)
        with torch.no_grad():
            _, _, _, value_m, value_w = self.forward(
                x, goals_copy, states_copy, mask, save=False
            )
        self.manager.Mrnn.dilation = saved_dilation
        return value_m, value_w


    # pass-through to the worker's intrinsic reward
    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    # pass-through to the manager's loss term
    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)

    # detach hidden states to prevent gradient flow across rollouts
    def repackage_hidden(self):
        self.hidden_m = [h.detach() for h in self.hidden_m]
        self.hidden_w = [h.detach() for h in self.hidden_w]

    # build the zeroed goals/states/masks
    def init_obj(self):
        template = torch.zeros(self.b, self.d).to(self.device)
        goals = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        states = [torch.zeros_like(template) for _ in range(2 * self.c + 1)]
        masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2 * self.c + 1)]
        return goals, states, masks


# Feudal loss
class Storage:

    # Simple experience storage for the n-step rollout.
    def __init__(self, size, keys=None):
        self.keys = keys or []
        self.size = size
        self.reset()

    # append one step's worth of values, creating a list for any new key
    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)


    # fill any key that received no data this rollout with None placeholders,
    # so every key ends up the same length
    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    # clear all stored values (called after each rollout's update)
    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    # rescale the stored values for each given key to zero mean, unit std
    def normalize(self, keys):
        for key in keys:
            k = torch.stack(getattr(self, key))
            k = (k - k.mean()) / (k.std() + 1e-10)
            setattr(self, key, [i for i in k])

    # stack a list of keys per-step values intotensors
    def stack(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.stack(x, dim=0), data)


# Computes the FuN loss - GAE advantages for both worker and manager, their
# policy-gradient and value losses, plus an entropy bonus. GAE is gated by
# the true end of episode flag, not the hidden-reset mask, so it doesn't
# leak value across episode boundaries.
def feudal_loss(storage, next_v_m, next_v_w, gamma_m, gamma_w, alpha,
                entropy_coef, num_steps, gae_lambda):
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

    # walk backwards through the rollout, accumulating GAE for both levels
    for i in reversed(range(num_steps)):
        nt = nts[i]                      # 0 if step i ENDED the episode, else 1

        # worker - reward = extrinsic + alpha * intrinsic
        rw = rewards[i] + alpha * rewards_intrinsic[i]
        delta_w = rw + gamma_w * next_vw * nt - v_w_det[i]
        gae_w = delta_w + gamma_w * gae_lambda * nt * gae_w
        adv_w_list[i] = gae_w
        next_vw = v_w_det[i]

        # manager - reward = extrinsic only
        delta_m  = rewards[i] + gamma_m * next_vm * nt - v_m_det[i]
        gae_m = delta_m + gamma_m * gae_lambda * nt * gae_m
        adv_m_list[i] = gae_m
        next_vm = v_m_det[i]

    adv_w = torch.stack(adv_w_list, dim=0)
    adv_m = torch.stack(adv_m_list, dim=0)

    # value-regression targets
    returns_w = adv_w + v_w_det
    returns_m = adv_m + v_m_det

    # normalise advantages before using them in the policy gradient
    adv_w_norm = (adv_w - adv_w.mean()) / (adv_w.std() + 1e-8)
    adv_m_norm = (adv_m - adv_m.mean()) / (adv_m.std() + 1e-8)
    loss_worker  = (logps * adv_w_norm.detach()).mean()
    loss_manager = (state_goal_cosines * adv_m_norm.detach()).mean()
    value_w_loss = 0.5 * (returns_w - value_w).pow(2).mean()
    value_m_loss = 0.5 * (returns_m - value_m).pow(2).mean()
    entropy = entropy.mean()

    # total loss - negative policy terms (maximise) + value regression - entropy bonus
    loss = (- loss_worker - loss_manager + value_w_loss + value_m_loss - entropy_coef * entropy)

    metrics = {
        'loss/total': loss.item(),
        'loss/worker': loss_worker.item(),
        'loss/manager': loss_manager.item(),
        'loss/value_worker': value_w_loss.item(),
        'loss/value_manager': value_m_loss.item(),
        'worker/entropy': entropy.item(),
        'worker/advantage': adv_w.mean().item(),
        'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
        'manager/cosines': state_goal_cosines.mean().item(),
        'manager/advantage': adv_m.mean().item(),
    }
    return loss, metrics
