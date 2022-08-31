from typing import Tuple
from torch import Tensor
from torch.nn import Module
from abc import ABC, abstractmethod

class ValueEstimator(ABC):
    @abstractmethod
    def __call__(experience: Tuple[Tensor], gamma: float, local_nn: Module, target_nn: Module) -> Tuple[Tensor]:
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
