"""
Vectorized Environment Wrapper for OGBench environments.

This wrapper follows Isaac Lab's RSL-RL interface exactly to ensure full
compatibility with RSL-RL's OnPolicyRunner and config system.
"""

import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Callable
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class VectorizedOGBenchEnv(VecEnv):
    """
    RSL-RL VecEnv implementation for multiple OGBench environments.
    
    Compatible with RSL-RL's OnPolicyRunner and follows Isaac Lab's interface.
    Manages multiple environments internally and provides vectorized
    step/reset functionality.
    """
    
    def __init__(self, env_name: str, num_envs: int = 1, wrappers: List[Callable] = None, 
                 clip_actions: float | None = None, **env_kwargs):
        """
        Args:
            env_name: OGBench environment name (e.g., 'pointmaze-arena-v0')
            num_envs: Number of parallel environments
            wrappers: List of wrapper functions to apply to each environment
            clip_actions: The clipping value for actions. If None, then no clipping is done.
            **env_kwargs: Additional arguments passed to gym.make()
        """
        # Initialize base class
        super().__init__()
        
        self.env_name = env_name
        self.wrappers = wrappers or []
        self.env_kwargs = env_kwargs
        self.clip_actions = clip_actions
        
        # Create individual environments
        self.envs = []
        for i in range(num_envs):
            env = gym.make(env_name, **env_kwargs)
            # Apply wrappers
            for wrapper_fn in self.wrappers:
                env = wrapper_fn(env)
            self.envs.append(env)
        
        # Get environment properties from first environment
        sample_env = self.envs[0]
        
        # RSL-RL VecEnv required attributes (matching Isaac Lab)
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get observation and action dimensions
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        self.num_actions = gym.spaces.flatdim(sample_env.action_space)
        self.num_obs = gym.spaces.flatdim(sample_env.observation_space)
        
        # Privileged observations (for asymmetric actor-critic)
        # OGBench environments don't have privileged observations by default
        self.num_privileged_obs = 0
        
        # Get max episode length from environment or default
        if hasattr(sample_env, '_max_episode_steps'):
            self.max_episode_length = sample_env._max_episode_steps
        elif hasattr(sample_env, 'spec') and sample_env.spec and sample_env.spec.max_episode_steps:
            self.max_episode_length = sample_env.spec.max_episode_steps
        else:
            self.max_episode_length = 500  # Default fallback
        
        # Episode tracking buffer (managed by RSL-RL)
        self._episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state for current observations
        self._current_obs = None
        self._current_obs_dict = None
        
        # Modify action space for clipping if specified
        self._modify_action_space()
        
        # Add cfg attribute for RSL-RL logging compatibility
        # Create a simple config object that contains basic environment info
        from dataclasses import dataclass
        
        @dataclass
        class EnvConfig:
            env_name: str
            num_envs: int
            max_episode_length: int
            is_finite_horizon: bool = True  # OGBench environments have finite episodes
        
        self.cfg = EnvConfig(env_name, num_envs, self.max_episode_length)
        
        # Reset all environments to initialize (RSL-RL runner doesn't call reset)
        self.reset()
    
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.
        
        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self._episode_length_buf = value
    
    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment.
        
        Returns:
            TensorDict: Observation TensorDict with policy key for RSL-RL compatibility
        """
        if self._current_obs is None:
            self.reset()
        # Return TensorDict with policy observations that RSL-RL expects
        return TensorDict(
            {"policy": self._current_obs},
            batch_size=[self.num_envs],
        )
    
    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments and return initial observations."""
        obs_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
        
        # Stack observations and convert to tensor
        obs_array = np.stack(obs_list, axis=0)
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        
        # Store current observations
        self._current_obs = obs_tensor
        self._current_obs_dict = {
            "policy": obs_tensor,
            # Note: Add "critic" key here if privileged observations are needed
        }
        
        # Reset episode length buffer
        self._episode_length_buf.zero_()
        
        # Return TensorDict format for RSL-RL compatibility
        obs_tensordict = TensorDict(
            {"policy": obs_tensor},
            batch_size=[self.num_envs],
        )
        return obs_tensordict, {"observations": self._current_obs_dict}
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step all environments with the given actions.
        
        Args:
            actions: Tensor of actions for all environments
            
        Returns:
            tuple: (observations, rewards, dones, extras)
        """
        # Clip actions if specified
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        
        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
        
        # Step each environment
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(terminated or truncated)
            infos_list.append(info)
        
        # Convert to arrays
        obs_array = np.stack(obs_list, axis=0)
        rewards_array = np.array(rewards_list)
        dones_array = np.array(dones_list)
        
        # Convert to tensors
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        rewards_tensor = torch.from_numpy(rewards_array).float().to(self.device)
        dones_tensor = torch.from_numpy(dones_array).to(dtype=torch.long, device=self.device)
        
        # Store current observations
        self._current_obs = obs_tensor
        self._current_obs_dict = {
            "policy": obs_tensor,
            # Note: Add "critic" key here if privileged observations are needed
        }
        
        # Update episode length tracking - increment BEFORE collecting metrics
        self._episode_length_buf += 1
        
        # Collect episode rewards and detailed metrics from completed episodes BEFORE resetting lengths
        episode_rewards = []
        episode_lengths = []
        sparse_rewards = []
        dense_rewards = []
        goals_reached = []
        distances = []
        
        # First pass: collect episode completion data before any resets
        for i, (done, info) in enumerate(zip(dones_array, infos_list)):
            if done and isinstance(info, dict):
                # Record episode length BEFORE resetting (this is the current episode length)
                current_episode_length = self._episode_length_buf[i].item()
                episode_lengths.append(current_episode_length)
                
                # Use cumulative episode reward from DetailedRewardWrapper
                total_episode_reward = info.get('episode_sparse_reward', 0.0) + info.get('episode_dense_reward', 0.0)
                episode_rewards.append(total_episode_reward)
        
        # Reset episode lengths for completed episodes AFTER recording them
        self._episode_length_buf[dones_tensor.bool()] = 0
        
        # Collect detailed metrics for completed episodes
        for i, (done, info) in enumerate(zip(dones_array, infos_list)):
            if done and isinstance(info, dict):
                # Detailed reward wrapper metrics (if available)
                if 'episode_sparse_reward' in info:
                    sparse_rewards.append(info['episode_sparse_reward'])
                if 'episode_dense_reward' in info:
                    dense_rewards.append(info['episode_dense_reward'])
                if 'goal_reached' in info:
                    goals_reached.append(info['goal_reached'])
                if 'distance_to_goal' in info:
                    distances.append(info['distance_to_goal'])
        
        # Create extras dict following Isaac Lab's format
        extras = {"observations": self._current_obs_dict}
        
        # Add episode completion metrics to extras
        if episode_rewards:
            extras['episode_rewards'] = episode_rewards
            extras['episode_lengths'] = episode_lengths
        
        # Add detailed metrics to extras if available (for ANY completed episodes)
        if goals_reached:  # Any completed episodes (regardless of reward type)
            extras['episode_sparse_rewards'] = sparse_rewards
            extras['episode_dense_rewards'] = dense_rewards
            extras['goals_reached'] = goals_reached
            extras['distances_to_goal'] = distances
            
            # Add RSL-RL compatible logging metrics using extras["log"] format
            # Following RSL-RL VecEnv documentation: keys start with "/" for namespacing
            if not extras.get('log'):
                extras['log'] = {}
                
            # Add goal success rate and other episode completion metrics
            extras['log']['/Episode/goal_success_rate'] = torch.tensor([float(g) for g in goals_reached], device=self.device)
            extras['log']['/Episode/final_distance'] = torch.tensor(distances, device=self.device)
            extras['log']['/Episode/sparse_reward'] = torch.tensor(sparse_rewards, device=self.device)
            extras['log']['/Episode/dense_reward'] = torch.tensor(dense_rewards, device=self.device)
        
        # Return TensorDict for observations
        obs_tensordict = TensorDict(
            {"policy": obs_tensor},
            batch_size=[self.num_envs],
        )
        
        return obs_tensordict, rewards_tensor, dones_tensor, extras
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
    
    def seed(self, seed: int = -1) -> int:
        """Set random seed for all environments."""
        for i, env in enumerate(self.envs):
            if hasattr(env, 'seed'):
                env.seed(seed + i if seed >= 0 else seed)
        return seed
    
    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return
        
        # Create new clipped action space
        # Note: This modifies the action space bounds but doesn't affect the actual environments
        # The clipping is done in the step method
        clipped_action_space = gym.spaces.Box(
            low=-self.clip_actions, 
            high=self.clip_actions, 
            shape=(self.num_actions,),
            dtype=np.float32
        )
        
        # Update action space (this is mainly for informational purposes)
        self.action_space = clipped_action_space