import warnings
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN) with optional Double Q-learning (DDQN) functionality and support for
    continuous reward functions (via a reward transformation function). The DDQN approach uses
    the main network to select actions and the target network to evaluate them, reducing overestimation bias.
    
    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236

    :param double_q: If True, use Double Q-learning (DDQN) target update.
    :param reward_fn: Optional function to transform rewards. This is useful for applications like 
                      portfolio optimization where the reward is a continuous signal (e.g., returns).
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, etc.)
    :param env: The environment to learn from (if registered in Gym, can be a str)
    :param learning_rate: The learning rate, which can be a function of the current progress remaining.
    :param buffer_size: Size of the replay buffer.
    :param learning_starts: Number of steps to collect transitions before learning starts.
    :param batch_size: Minibatch size for each gradient update.
    :param tau: The soft update coefficient ("Polyak update", between 0 and 1), default 1 for hard update.
    :param gamma: The discount factor.
    :param train_freq: Frequency (or tuple) to update the model.
    :param gradient_steps: Number of gradient steps to do after each rollout.
    :param replay_buffer_class: Replay buffer class to use.
    :param replay_buffer_kwargs: Keyword arguments for replay buffer creation.
    :param optimize_memory_usage: If True, use a memory efficient replay buffer.
    :param target_update_interval: How many environment steps between target network updates.
    :param exploration_fraction: Fraction of training period over which the exploration rate is reduced.
    :param exploration_initial_eps: Initial value of random action probability.
    :param exploration_final_eps: Final value of random action probability.
    :param max_grad_norm: Maximum value for gradient clipping.
    :param stats_window_size: Window size for rollout logging.
    :param tensorboard_log: Log location for tensorboard.
    :param policy_kwargs: Additional arguments to pass to the policy on creation.
    :param verbose: Verbosity level.
    :param seed: Seed for pseudo random generators.
    :param device: Device (cpu, cuda, etc.) on which to run the code.
    :param _init_setup_model: Whether or not to build the network at instance creation.
    """
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        double_q: bool = True,  # New parameter to toggle DDQN functionality
        reward_fn: Optional[Callable[[th.Tensor], th.Tensor]] = None,  # New parameter for continuous reward transformation
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.double_q = double_q
        self.reward_fn = reward_fn  # Save the reward transformation function
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments: each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # Apply reward transformation if a function is provided.
            # This is useful for portfolio optimization where rewards (e.g., returns) are continuous.
            if self.reward_fn is not None:
                rewards = self.reward_fn(replay_data.rewards)
            else:
                rewards = replay_data.rewards

            with th.no_grad():
                if self.double_q:
                    # === Double Q-learning target computation ===
                    # 1. Use the main network to select the best next action.
                    next_q_values_main = self.q_net(replay_data.next_observations)
                    next_actions = next_q_values_main.argmax(dim=1, keepdim=True)
                    # 2. Use the target network to evaluate the Q-value of that action.
                    next_q_values_target = self.q_net_target(replay_data.next_observations)
                    next_q_value = next_q_values_target.gather(1, next_actions)
                else:
                    # Standard DQN: use target network to compute maximum Q-value.
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    next_q_value, _ = next_q_values.max(dim=1, keepdim=True)

                # 1-step TD target using (possibly transformed) continuous rewards.
                target_q_values = rewards + (1 - replay_data.dones) * self.gamma * next_q_value

            # Get current Q-value estimates from the main network.
            current_q_values = self.q_net(replay_data.observations)
            # Retrieve Q-values corresponding to the actions taken.
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers).
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy.
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm.
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter.
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base predict function to include epsilon-greedy exploration.
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
        return action, state

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []
