import torch
import random
import numpy as np
from collections import namedtuple, deque

from experience_replay import ExperienceReplay

class PriorityBuffer(ExperienceReplay):
    def __init__(self, config):
        self.max_size = config["max_size"]
        self.local_model = config["local_model"]
        self.target_model = config["target_model"]
        self.estimator = config["estimator"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.priority_amount = config["priority_amount"]
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'error'])
        self.experience_buffer = deque(maxlen=config['max_size'])

    def add(self, state, action, reward, next_state, done):
        state = torch.tensor(state).float().unsqueeze(0)
        action = torch.tensor([action]).long().unsqueeze(1)
        reward = torch.tensor(reward).float().unsqueeze(0)
        next_state = torch.tensor(next_state).float().unsqueeze(0)
        done = torch.tensor(done).float().unsqueeze(0)
        *exp_tuple, _ = self.experience(state, action, reward, next_state, done, 0)
        with torch.no_grad():
            pred, target = self.estimator(exp_tuple, self.gamma, self.local_model, self.target_model)
            error = torch.pow(pred - target, 2) + self.epsilon
        exp = self.experience(state, action, reward, next_state, done, error)
        self.experience_buffer.append(exp)

    def sample(self, batch_size):
        errors = np.array([exp.error for exp in list(self.experience_buffer)])
        scaled_errors = errors ** self.priority_amount
        p = scaled_errors/sum(scaled_errors)
        experiences_idxs = np.random.choice(range(len(p)), size=batch_size, p=p, replace=False).tolist()
        experiences = [self.experience_buffer[idx] for idx in experiences_idxs]
        states = torch.vstack([exp.state for exp in experiences]).float()
        actions = torch.tensor([exp.action for exp in experiences]).long().unsqueeze(1)
        rewards = torch.tensor([exp.reward for exp in experiences]).float()
        next_states = torch.vstack([exp.next_state for exp in experiences]).float()
        dones = torch.tensor([exp.done for exp in experiences]).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.experience_buffer)