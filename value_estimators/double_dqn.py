from typing import Tuple
import torch
from torch import Tensor
from torch.nn import Module
import numpy as np

from .value_estimator import ValueEstimator

class DoubleDQNEstimator(ValueEstimator):
    def __call__(self, experience: Tuple[Tensor], gamma: float, local_nn: Module, target_nn: Module) -> Tuple[Tensor]:
        """Computes the action-value estimation for an experience tuple with the given local and
        target networks. It computes the value estimation directly from the local target, as well
        as the bellman equation value estimation with the target network.

        Args:
            experience (Tuple[Tensor]): a tuple containing tensors for states, actions, 
            rewards, next_states and dones
            gamma (float): discount factor
            local_nn (Module): the local, more frequently updated neural network
            target_nn (Module): the target, more stable neural network

        Returns:
            List[Tensor]: a list containing value estimation from the local network and the bellman update.
        """

        states, actions, rewards, next_states, dones = experience

        with torch.no_grad():
            # Implement DQN
            max_actions = np.argmax(local_nn(next_states), axis=1).unsqueeze(1)
            max_vals = target_nn(next_states)
            max_vals = max_vals.gather(1, max_actions).squeeze()
            target = rewards + gamma * max_vals * (1-dones)
        pred_values = local_nn(states)
        pred = pred_values.gather(1, actions).squeeze()

        return pred, target