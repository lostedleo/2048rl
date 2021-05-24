import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
from torch.nn.functional import relu
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Categorical

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 2        # how often to update the network

class ResBlock(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(in_channels, hid_channels)

    def forward(self, x):
        out = x
        out = F.relu(self.linear1(out))
        out += x
        return out

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, dueling=False, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling

        self.build_resnet(state_size, action_size, fc1_units, fc2_units)

        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def build_resnet(self, state_size, action_size, fc1_units, fc2_units):
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.resnet = nn.Sequential(
            ResBlock(fc1_units, fc1_units),
            ResBlock(fc1_units, fc1_units),
            ResBlock(fc1_units, fc1_units),
        )
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        if self.dueling:
            self.V = nn.Linear(fc2_units, 1)
            self.A = nn.Linear(fc2_units, action_size)
        else:
            self.Q = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = self.resnet(x)
        x = F.relu(self.fc2(x))
        if self.dueling:
            V = self.V(x)
            A = self.A(x)
            return V + (A - A.mean(dim=1, keepdim=True)), A
        else:
            return self.Q(x), None

class ResCnnBlock(nn.Module):
    def __init__(self, in_channels, hid_channels):
        super(ResCnnBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hid_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(hid_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        out = x
        out = self.conv1(relu(out))
        out = self.conv2(relu(out))
        out += x
        return out

class CNNQNetwork(nn.Module):
    def __init__(self, channel_size, size, action_size, seed, dueling=False, fc1_units=32, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(CNNQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling = dueling

        self.build_cnnnet(channel_size, size, action_size, fc1_units, fc2_units)

        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def build_cnnnet(self, channel_size, size, action_size, fc1_units, fc2_units):
        channels = self.channels = [fc1_units, fc2_units]
        self.a = nn.Sequential(
            nn.Conv2d(channel_size, channels[0], (2, 1), 1),
            ResCnnBlock(channels[0], channels[0]),
        )
        self.aa = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 2, 1),
            ResCnnBlock(channels[1], channels[1]),
        )

        self.b = nn.Sequential(
            nn.Conv2d(channel_size, channels[0], (1, 2), 1),
            ResCnnBlock(channels[0], channels[0]),
        )
        self.bb = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], 2, 1),
            ResCnnBlock(channels[1], channels[1]),
        )
        self.output_size = self.eval_output_size(size)

        if self.dueling:
            self.V = nn.Linear(self.output_size, 1)
            self.A = nn.Linear(self.output_size, action_size)
        else:
            self.Q = nn.Linear(self.output_size, action_size)

    def eval_output_size(self, size):
        output_size = (self.channels[0] * (size - 1) * size
                + self.channels[1] * (size - 2) * (size - 1)) * 2
        return output_size

    def forward(self, state):
        """Build a network that maps state -> action values."""
        N, L, H, W = state.shape
        a = self.a(state)
        aa = self.aa(a)
        b = self.b(state)
        bb = self.bb(b)
        x = torch.cat((a.view(N, -1), aa.view(N, -1), b.view(N, -1), bb.view(N, -1)), 1)

        if self.dueling:
            V = self.V(x)
            A = self.A(x)
            return V + (A - A.mean(dim=1, keepdim=True)), A
        else:
            return self.Q(x), None

class DQNAgent():
    """Interacts with and learns from the environment."""
    def __init__(self, size, channels, action_size, seed, double_q=True, dueling=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = size * size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.double_q = double_q
        self.dueling = dueling

        # Q-Network
        if channels > 1:
            self.q_eval = CNNQNetwork(channels, size, action_size, seed, dueling)
            self.q_next = CNNQNetwork(channels, size, action_size, seed, dueling)
        else:
            self.q_eval = QNetwork(state_size, action_size, seed, dueling)
            self.q_next = QNetwork(state_size, action_size, seed, dueling)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.q_eval.device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.loss = 0.0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.loss = self.learn(experiences, GAMMA)

        return self.loss

    def choose_action(self, state, eps=0., rand=False):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if rand:
           return random.choice(np.arange(self.action_size)), None
        # Epsilon-greedy action selection
        shape = state.shape
        new_shape = (1, ) + shape
        state = torch.from_numpy(state).float().reshape(new_shape).to(self.q_eval.device)
        if random.random() > eps:
            if self.dueling:
                action_values = self.q_eval(state)[1].cpu()
            else:
                action_values = self.q_eval(state)[0].cpu()
            return np.argmax(action_values.data.numpy()), action_values
        else:
            action_values = self.q_eval(state)[0].cpu()
            cat = Categorical(logits=action_values)
            action = cat.sample()
            return action, None

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_next(next_states)[0].detach()
        #  Q_target_current = self.q_next(states).detach().min(1)[0].unsqueeze(1)
        #  zeros = torch.zeros(Q_target_current.shape)
       #   Q_target_current = torch.fmin(Q_target_current, zeros)

        N, L, H, W = states.shape

        equal = torch.all(torch.eq(states, next_states).reshape(N, L*H*W), dim=1).type(torch.FloatTensor).unsqueeze(1).to(self.q_eval.device)

        if self.double_q:
            Q_eval_next, _ = self.q_eval(states)
            max_act4next = np.argmax(Q_eval_next, axis=1)
            selected_q_next = Q_targets_next[np.arange(Q_targets_next.shape[0]), max_act4next].unsqueeze(1)
        else:
            selected_q_next = Q_targets_next.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        #  delta = gamma * (selected_q_next * (1 - equal) + Q_target_current * equal - equal * rewards) * (1 - dones)
        delta = gamma * (selected_q_next * (1 - equal)) * (1 - dones)
        #  delta = gamma * (selected_q_next * (1 - equal)) * (1 - dones)
        Q_targets = rewards + delta
        #  Q_targets = rewards + gamma * selected_q_next
        #  Q_targets[dones] = 0.0

        # Get expected Q values from local model
        Q_expected = self.q_eval(states)[0].gather(1, actions)

        # Compute loss
        loss = self.q_eval.loss(Q_expected, Q_targets)
        self.q_eval.optimizer.zero_grad()
        # Minimize the loss
        loss.backward()
        self.q_eval.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_eval, self.q_next, TAU)
        return loss

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, name):
        torch.save(self.q_eval.state_dict(), name)

    def load(self, name):
        self.q_eval.load_state_dict(torch.load(name))
        self.q_next.load_state_dict(torch.load(name))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        shape = experiences[0].state.shape
        new_shape = (self.batch_size, ) + shape

        states = torch.from_numpy(np.vstack([e.state for e in experiences\
                if e is not None])).float().reshape(new_shape).to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences\
                if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences\
                if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences\
                if e is not None])).float().reshape(new_shape).to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences\
                if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
