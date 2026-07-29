from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


def make_mlp(
    input_size: int,
    hidden_sizes: Sequence[int],
    output_size: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        current_size = hidden_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)


def make_actor_body(
    input_size: int,
    hidden_sizes: Sequence[int],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        current_size = hidden_size
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int],
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.body = make_actor_body(observation_size, hidden_sizes)
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_size)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_size)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observation)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observation)
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)

        # Change-of-variables correction for tanh.
        log_probability = distribution.log_prob(raw_action)
        log_probability -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=-1, keepdim=True)
        return action, log_probability

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self(observation)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        self.network = make_mlp(
            observation_size + action_size,
            hidden_sizes,
            1,
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))
