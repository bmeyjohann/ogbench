"""
Vectorized Environment Wrapper for OGBench environments.

This wrapper creates multiple parallel environments using SubprocVecEnv
while maintaining compatibility with RSL-RL's VecEnv interface.
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium.vector import SubprocVecEnv
from typing import Dict, Any, List, Callable
from rsl_rl.env import VecEnv
from tensordict import TensorDict


def make_env_fn(env_name: str, wrappers: List[Callable] = None, **kwargs):
    """Factory function to create a single environment with wrappers."""
    def _make_env():
        env = gym.make(env_name, **kwargs)
        if wrappers:
            for wrapper_fn in wrappers:
                env = wrapper_fn(env)
        return env
    return _make_env


class VectorizedOGBenchEnv(VecEnv):
    """
    RSL-RL compatible vectorized environment wrapper.
    
    Takes a single environment configuration and creates multiple parallel
    instances using SubprocVecEnv for true multiprocessing.
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
        self.num_envs = num_envs
        self.wrappers = wrappers or []
        self.env_kwargs = env_kwargs
        
        # Create vectorized environment
        if num_envs == 1:
            # Single environment - no need for multiprocessing overhead
            self.vec_env = gym.make(env_name, **env_kwargs)
            if self.wrappers:
                for wrapper_fn in self.wrappers:
                    self.vec_env = wrapper_fn(self.vec_env)
            # Wrap single env to look like vec env
            self._is_single = True
        else:
            # Multiple environments - use SubprocVecEnv
            env_fns = [make_env_fn(env_name, self.wrappers, **env_kwargs) 
                      for _ in range(num_envs)]
            self.vec_env = SubprocVecEnv(env_fns)
            self._is_single = False
        
        # Get environment properties from first environment
        if self._is_single:
            sample_env = self.vec_env
        else:
            # Create a temporary environment to get properties
            temp_env = gym.make(env_name, **env_kwargs)
            if self.wrappers:
                for wrapper_fn in self.wrappers:
                    temp_env = wrapper_fn(temp_env)
            sample_env = temp_env
        
        # RSL-RL required attributes
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
        
        # Clean up temporary environment
        if not self._is_single:
            temp_env.close()
        
        # Internal state
        self._last_obs = None
    
    def reset(self):
        """Reset all environments and return initial observations."""
        if self._is_single:
            obs, info = self.vec_env.reset()
            obs = np.expand_dims(obs, 0)  # Add batch dimension
            infos = [info]
        else:
            obs, infos = self.vec_env.reset()
        
        # Convert to torch tensor
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        
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
        
        # Handle single vs multiple environments
        if self._is_single:
            actions_np = actions_np.squeeze(0)  # Remove batch dimension
            obs, reward, terminated, truncated, info = self.vec_env.step(actions_np)
            
            # Add batch dimensions back
            obs = np.expand_dims(obs, 0)
            reward = np.array([reward])
            terminated = np.array([terminated])
            truncated = np.array([truncated])
            infos = [info]
        else:
            obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions_np)
            reward = rewards
            terminated = terminateds
            truncated = truncateds
        
        # Convert observations to torch tensors
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        
        # Convert to TensorDict
        obs_tensordict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[self.num_envs], device=self.device)
        
        self._last_obs = obs_tensordict
        
        # Convert rewards and dones to torch tensors
        rewards_tensor = torch.from_numpy(reward).float().to(self.device)
        dones_tensor = torch.from_numpy(terminated | truncated).bool().to(self.device)
        
        # Update episode length tracking
        self.episode_length_buf += 1
        self.episode_length_buf[dones_tensor] = 0
        
        # Create extras dict with episode information
        extras = {}
        
        # Collect episode rewards from info dicts
        episode_rewards = []
        episode_lengths = []
        for i, info in enumerate(infos):
            if isinstance(info, dict):
                if 'episode' in info:
                    episode_rewards.append(info['episode'].get('r', 0.0))
                    episode_lengths.append(info['episode'].get('l', 0))
                elif 'final_info' in info and info['final_info']:
                    # Handle gymnasium's new episode info format
                    episode_rewards.append(info.get('episode_reward', 0.0))
                    episode_lengths.append(info.get('episode_length', 0))
        
        if episode_rewards:
            extras['episode_rewards'] = episode_rewards
            extras['episode_lengths'] = episode_lengths
        
        return obs_tensordict, rewards_tensor, dones_tensor, extras
    
    def get_observations(self):
        """Return the last observations."""
        return self._last_obs
    
    def close(self):
        """Close all environments."""
        if hasattr(self.vec_env, 'close'):
            self.vec_env.close()