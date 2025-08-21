"""
Flexible observation wrapper for OGBench environments.

This wrapper allows you to configure exactly which observation components you want:
- Agent position (always included)
- Goal position 
- Distance to goal
- Direction to goal
- Velocity information

Simple and configurable for different training scenarios.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlexibleObsWrapper(gym.ObservationWrapper):
    """
    Flexible observation wrapper that lets you choose which components to include.
    
    Available components:
    - Agent position: [agent_x, agent_y] (always included)
    - Goal position: [goal_x, goal_y] (if include_goal=True)
    - Distance: [distance_to_goal] (if include_distance=True)
    - Direction: [direction_x, direction_y] (if include_direction=True)
    - Velocity: [velocity_x, velocity_y] (if include_velocity=True)
    """
    
    def __init__(self, env, 
                 include_goal=True,
                 include_distance=False, 
                 include_direction=False,
                 include_velocity=False):
        """
        Initialize the flexible observation wrapper.
        
        Args:
            env: The base environment
            include_goal: Include goal position in observations
            include_distance: Include euclidean distance to goal
            include_direction: Include normalized direction vector to goal
            include_velocity: Include velocity information (if available)
        """
        super().__init__(env)
        
        self.include_goal = include_goal
        self.include_distance = include_distance
        self.include_direction = include_direction
        self.include_velocity = include_velocity
        
        self._last_goal = None
        self._last_velocity = None
        
        # Calculate observation dimension
        obs_dim = 2  # Agent position (always included)
        if self.include_goal:
            obs_dim += 2  # Goal position
        if self.include_distance:
            obs_dim += 1  # Distance
        if self.include_direction:
            obs_dim += 2  # Direction vector
        if self.include_velocity:
            obs_dim += 2  # Velocity
        
        # Create observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Print configuration
        components = ['agent_position']
        if self.include_goal: components.append('goal_position')
        if self.include_distance: components.append('distance')
        if self.include_direction: components.append('direction')
        if self.include_velocity: components.append('velocity')
        
        # print(f"FlexibleObsWrapper: Obs space {env.observation_space.shape} -> {self.observation_space.shape}")
        # print(f"   Components: {', '.join(components)}")
    
    def observation(self, obs):
        """Convert observation to flexible format."""
        obs = np.array(obs, dtype=np.float32)
        agent_pos = obs[:2]  # Agent position (first 2 elements)
        
        # Start with agent position
        components = [agent_pos]
        
        # Add goal position if requested
        if self.include_goal:
            if self._last_goal is not None:
                goal_pos = np.array(self._last_goal[:2], dtype=np.float32)
            else:
                goal_pos = np.zeros(2, dtype=np.float32)
                if not hasattr(self, '_goal_warning_shown'):
                    print("Warning: No goal position available, using zeros")
                    self._goal_warning_shown = True
            components.append(goal_pos)
        
        # Calculate distance and direction if needed
        if self.include_distance or self.include_direction:
            if self._last_goal is not None:
                goal_pos = np.array(self._last_goal[:2], dtype=np.float32)
                distance_vec = goal_pos - agent_pos
                distance = np.linalg.norm(distance_vec)
                
                if self.include_distance:
                    components.append(np.array([distance]))
                
                if self.include_direction:
                    # Normalized direction vector
                    if distance > 1e-8:
                        direction = distance_vec / distance
                    else:
                        direction = np.zeros(2, dtype=np.float32)
                    components.append(direction)
            else:
                # No goal available
                if self.include_distance:
                    components.append(np.array([0.0]))
                if self.include_direction:
                    components.append(np.zeros(2, dtype=np.float32))
        
        # Add velocity if requested
        if self.include_velocity:
            if self._last_velocity is not None:
                vel = np.array(self._last_velocity[:2], dtype=np.float32)
            else:
                vel = np.zeros(2, dtype=np.float32)
            components.append(vel)
        
        # Concatenate all components
        return np.concatenate(components)
    
    def reset(self, **kwargs):
        """Reset environment and extract goal information."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract goal from info
        if isinstance(info, dict) and 'goal' in info:
            self._last_goal = info['goal']
        else:
            self._last_goal = None
        
        # Reset velocity
        self._last_velocity = np.zeros(2)
        
        return self.observation(obs), info
    
    def step(self, action):
        """Step environment and update goal and velocity information."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update goal if available
        if isinstance(info, dict) and 'goal' in info:
            self._last_goal = info['goal']
        
        # Extract velocity if available
        if isinstance(info, dict):
            if 'qvel' in info:
                self._last_velocity = info['qvel'][:2] if len(info['qvel']) >= 2 else np.zeros(2)
            elif 'velocity' in info:
                self._last_velocity = info['velocity'][:2] if len(info['velocity']) >= 2 else np.zeros(2)
        
        return self.observation(obs), reward, terminated, truncated, info


def test_flexible_wrapper():
    """Test the flexible wrapper with different configurations."""
    import ogbench
    
    print("🧪 Testing FlexibleObsWrapper")
    print("=" * 50)
    
    configs = [
        {'include_goal': True},  # Just agent + goal (most common)
        {'include_goal': True, 'include_distance': True, 'include_direction': True},  # Full enhanced
        {'include_goal': False},  # Just agent position (minimal)
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. Configuration: {config}")
        
        env = gym.make("pointmaze-arena-v0", render_mode=None)
        env = FlexibleObsWrapper(env, **config)
        
        obs, info = env.reset(seed=42)
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation: {obs}")
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"   After step: {obs}")
        
        env.close()
    
    print("\n✅ Flexible wrapper tests complete!")


if __name__ == "__main__":
    test_flexible_wrapper()