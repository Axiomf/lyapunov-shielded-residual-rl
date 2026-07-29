import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_size: int,
        action_size: int,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.observations = np.zeros(
            (capacity, observation_size),
            dtype=np.float32,
        )
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros(
            (capacity, observation_size),
            dtype=np.float32,
        )
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)

        self.position = 0
        self.size = 0
        self.random = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.terminated[self.position] = float(terminated)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("not enough samples in replay buffer")

        indices = self.random.integers(0, self.size, size=batch_size)

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array[indices], device=device)

        return {
            "observations": tensor(self.observations),
            "actions": tensor(self.actions),
            "rewards": tensor(self.rewards),
            "next_observations": tensor(self.next_observations),
            "terminated": tensor(self.terminated),
        }

