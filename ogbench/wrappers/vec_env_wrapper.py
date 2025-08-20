"""
Vectorized Environment Wrapper for OGBench environments.

This wrapper extends RSL-RL's VecEnv to handle multiple OGBench environments
internally, providing proper vectorization without external dependencies.
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
    
    Manages multiple environments internally and provides vectorized
    step/reset functionality compatible with RSL-RL training.
    """
    
    def __init__(self, env_name: str, num_envs: int = 1, wrappers: List[Callable] = None, **env_kwargs):
        """
        Args:
            env_name: OGBench environment name (e.g., 'pointmaze-arena-v0')
            num_envs: Number of parallel environments
            wrappers: List of wrapper functions to apply to each environment
            **env_kwargs: Additional arguments passed to gym.make()
        """
        self.env_name = env_name
        self.wrappers = wrappers or []
        self.env_kwargs = env_kwargs
        
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
        
        # RSL-RL VecEnv required attributes
        super().__init__()
        self.num_envs = num_envs
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        self.num_actions = self.action_space.shape[0]
        
        # Get max episode length from environment or default
        if hasattr(sample_env, '_max_episode_steps'):
            self.max_episode_length = sample_env._max_episode_steps
        elif hasattr(sample_env, 'spec') and sample_env.spec and sample_env.spec.max_episode_steps:
            self.max_episode_length = sample_env.spec.max_episode_steps
        else:
            self.max_episode_length = 500  # Default fallback
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state
        self._last_obs = None
        
        # Reset all environments to initialize
        self.reset()
    
    def reset(self):
        """Reset all environments and return initial observations."""
        obs_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
        
        # Stack observations
        obs_array = np.stack(obs_list, axis=0)
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        
        # Convert to TensorDict with proper structure for RSL-RL
        self._last_obs = TensorDict({
            "policy": obs_tensor,  # Observations for policy network
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions):
        """Step all environments with the given actions."""
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
        
        # Convert observations to torch tensors
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        
        # Convert to TensorDict
        obs_tensordict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[self.num_envs], device=self.device)
        
        self._last_obs = obs_tensordict
        
        # Convert rewards and dones to torch tensors
        rewards_tensor = torch.from_numpy(rewards_array).float().to(self.device)
        dones_tensor = torch.from_numpy(dones_array).bool().to(self.device)
        
        # Update episode length tracking
        self.episode_length_buf += 1
        self.episode_length_buf[dones_tensor] = 0
        
        # Create extras dict with episode information
        extras = {}
        
        # Collect episode rewards and detailed metrics from completed episodes
        episode_rewards = []
        episode_lengths = []
        sparse_rewards = []
        dense_rewards = []
        goals_reached = []
        distances = []
        
        for i, (done, info) in enumerate(zip(dones_array, infos_list)):
            if done and isinstance(info, dict):
                # Use cumulative episode reward from DetailedRewardWrapper
                total_episode_reward = info.get('episode_sparse_reward', 0.0) + info.get('episode_dense_reward', 0.0)
                episode_rewards.append(total_episode_reward)
                episode_lengths.append(self.episode_length_buf[i].item())
                
                # Detailed reward wrapper metrics (if available)
                if 'episode_sparse_reward' in info:
                    sparse_rewards.append(info['episode_sparse_reward'])
                if 'episode_dense_reward' in info:
                    dense_rewards.append(info['episode_dense_reward'])
                if 'goal_reached' in info:
                    goals_reached.append(info['goal_reached'])
                if 'distance_to_goal' in info:
                    distances.append(info['distance_to_goal'])
        
        # Add episode completion metrics to extras
        if episode_rewards:
            extras['episode_rewards'] = episode_rewards
            extras['episode_lengths'] = episode_lengths
        
        # Add detailed metrics to extras if available
        if sparse_rewards:
            extras['episode_sparse_rewards'] = sparse_rewards
            extras['episode_dense_rewards'] = dense_rewards
            extras['goals_reached'] = goals_reached
            extras['distances_to_goal'] = distances
        
        return obs_tensordict, rewards_tensor, dones_tensor, extras
    
    def get_observations(self):
        """Return the last observations."""
        return self._last_obs
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()