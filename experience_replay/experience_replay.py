from torch import tensor
from abc import ABC, abstractmethod

class ExperienceReplay(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        """Creates an Experience Replay instance

        Args:
            config (dict): includes information required for the instance to work
        """
    
    @abstractmethod
    def add(self, state: tensor, action: int, reward: float, next_state: tensor, done: bool):
        """Adds an experience tuple to the memory buffer

        Args:
            state (tensor): A tensor describing the state of the environment
            action (int): The action taken by the agent
            reward (float): The reward obtained after interaction
            next_state (tensor): A tensor describing the obtained state after interaction
            done (bool): A bool indicating wether the end of the episode was reached
        """

    @abstractmethod
    def sample(self, batch_size: int):
        """Retrieves a random set of experience tuples from the memory buffer

        Args:
            batch_size (int): The number of samples to retrieve in a single batch
        """

    @abstractmethod
    def __len__(self):
        """Number of samples in the memory buffer
        """