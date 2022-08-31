import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm

from experience_replay import ExperienceReplay
from value_estimators import ValueEstimator

class DQNAgent:
    def __init__(
            self,
            env,
            value_estimator: ValueEstimator,
            local_model: nn.Module,
            target_model: nn.Module,
            memory: ExperienceReplay,
            hidden_layers=[64, 64]
        ):
        self.env = env
        self.value_estimator = value_estimator

        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]

        env_info = env.reset(train_mode=True)[self.brain_name]
        self.state_size = len(env_info.vector_observations[0])
        self.action_size = self.brain.vector_action_space_size

        self.hidden_layers = hidden_layers
        self.qnetwork_local = local_model
        self.qnetwork_target = target_model
        self.memory = memory
        self.t = 0
        self.rewards = []
    
    def act(self, state, eps=0):
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(state).float().unsqueeze(0)
            q_values = self.qnetwork_local(state).detach().numpy()
            return np.argmax(q_values)

    def train(self, episodes=2000, max_steps=2000000, lr=0.001, alpha=0.01, gamma=0.99, epsilon=1, epsilon_end=0.0, epsilon_decay=1e-5, train_every=4, batch_size=64):
        train_bar = tqdm(range(1, episodes+1))
        env = self.env
        brain_name = self.brain_name
        decay = 1 - epsilon_decay
        self.rewards = [0]

        for ep in train_bar:
            env_info = env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            done = False
            total_reward = 0

            for t in range(1, max_steps+1): 
                if done:
                    break
                action = self.act(state, epsilon)
                env_info = env.step(action)[brain_name]        # send the action to the environment
                next_state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]
                self.memory.add(state, action, reward, next_state, done)

                # Train
                self.t = (self.t + 1) % train_every
                if self.t == 0:
                    if len(self.memory) > batch_size:
                        self.train_step(lr, batch_size, gamma, alpha)

                # Update variables
                epsilon = max(epsilon_end, epsilon * decay)
                total_reward += reward
                state = next_state

            train_bar.set_description(f"Avg Reward: {round(sum(self.rewards[-100:]) / (len(self.rewards[-100:]) + 1e-5), 2)} | Epsilon {round(epsilon, 3)}")

            self.rewards.append(total_reward)


    def train_step(self, lr, batch_size, gamma, alpha):
        optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        loss = nn.MSELoss()

        experience = self.memory.sample(batch_size)
        
        pred, target = self.value_estimator(experience, gamma, self.qnetwork_local, self.qnetwork_target)

        loss_val = loss(pred, target)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        self.soft_update(alpha)
        return loss_val

    def soft_update(self, alpha):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(alpha * local_param.data + (1.0-alpha) * target_param.data)

    def save(self, path):
        model = self.qnetwork_local
        torch.save(model.state_dict(), path)

    def load(self, path):
        model = self.qnetwork_local
        model.load_state_dict(torch.load(path))
