from dataclasses import asdict
from math import log
from pathlib import Path

import numpy as np
import torch
from torch import nn

from residual_sac.config import SACConfig
from residual_sac.networks import QNetwork, SquashedGaussianActor


class SACAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        config: SACConfig,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.config = config
        self.device = torch.device(config.device)

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        actor_arguments = (
            observation_size,
            action_size,
            config.hidden_sizes,
            config.log_std_min,
            config.log_std_max,
        )
        critic_arguments = (
            observation_size,
            action_size,
            config.hidden_sizes,
        )

        self.actor = SquashedGaussianActor(*actor_arguments).to(self.device)
        self.critic_1 = QNetwork(*critic_arguments).to(self.device)
        self.critic_2 = QNetwork(*critic_arguments).to(self.device)
        self.target_critic_1 = QNetwork(*critic_arguments).to(self.device)
        self.target_critic_2 = QNetwork(*critic_arguments).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.requires_grad_(False)
        self.target_critic_2.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=config.actor_learning_rate,
        )
        critic_parameters = list(self.critic_1.parameters()) + list(
            self.critic_2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters,
            lr=config.critic_learning_rate,
        )

        self.log_alpha = torch.tensor(
            log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=config.automatic_entropy_tuning,
        )
        if config.automatic_entropy_tuning:
            self.alpha_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                [self.log_alpha],
                lr=config.alpha_learning_rate,
            )
        else:
            self.alpha_optimizer = None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action_tensor = self.actor.deterministic(observation_tensor)
            else:
                action_tensor, _ = self.actor.sample(observation_tensor)

        return action_tensor.squeeze(0).cpu().numpy()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        terminated = batch["terminated"]

        with torch.no_grad():
            next_actions, next_log_probabilities = self.actor.sample(next_observations)
            next_q_1 = self.target_critic_1(next_observations, next_actions)
            next_q_2 = self.target_critic_2(next_observations, next_actions)
            next_q = torch.minimum(next_q_1, next_q_2)
            next_value = next_q - self.alpha.detach() * next_log_probabilities
            q_target = rewards + (
                self.config.gamma * (1.0 - terminated) * next_value
            )

        q_1 = self.critic_1(observations, actions)
        q_2 = self.critic_2(observations, actions)
        critic_loss = nn.functional.mse_loss(q_1, q_target)
        critic_loss += nn.functional.mse_loss(q_2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic_1.requires_grad_(False)
        self.critic_2.requires_grad_(False)
        new_actions, log_probabilities = self.actor.sample(observations)
        actor_q_1 = self.critic_1(observations, new_actions)
        actor_q_2 = self.critic_2(observations, new_actions)
        actor_q = torch.minimum(actor_q_1, actor_q_2)
        actor_loss = (self.alpha.detach() * log_probabilities - actor_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_1.requires_grad_(True)
        self.critic_2.requires_grad_(True)

        alpha_loss_value = 0.0
        if self.alpha_optimizer is not None:
            alpha_loss = -(
                self.log_alpha
                * (log_probabilities + self.config.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.item())

        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(),
                target.parameters(),
                strict=True,
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(self.config.tau * source_parameter)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": (
                self.alpha_optimizer.state_dict()
                if self.alpha_optimizer is not None
                else None
            ),
        }
        torch.save(checkpoint, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
        load_optimizers: bool = False,
    ) -> "SACAgent":
        checkpoint = cls._read_checkpoint(path, device)
        config_values = checkpoint["config"]
        config_values["hidden_sizes"] = tuple(config_values["hidden_sizes"])
        config_values["device"] = device
        agent = cls(
            checkpoint["observation_size"],
            checkpoint["action_size"],
            SACConfig(**config_values),
        )
        agent._load_checkpoint_values(checkpoint, load_optimizers)
        return agent

    @staticmethod
    def _read_checkpoint(path: str | Path, device: str) -> dict:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    def _load_checkpoint_values(
        self,
        checkpoint: dict,
        load_optimizers: bool,
    ) -> None:
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))

        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if (
                self.alpha_optimizer is not None
                and checkpoint["alpha_optimizer"] is not None
            ):
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])

