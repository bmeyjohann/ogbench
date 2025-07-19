"""
Goal-conditioned observation wrapper for OGBench PointMaze environments.
This wrapper concatenates agent position with goal position to create proper goal-conditioned observations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class GoalConditionedWrapper(gym.ObservationWrapper):
    """
    Wrapper that creates goal-conditioned observations by concatenating agent position with goal position.
    
    This solves the core issue where PointMaze agents can't see where the goal is located.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get original observation space (agent position)
        original_obs_space = env.observation_space
        
        # Assume goal is also 2D position (same as agent)
        goal_dim = 2
        
        # Create new observation space: [agent_x, agent_y, goal_x, goal_y]
        if isinstance(original_obs_space, spaces.Box):
            low = np.concatenate([original_obs_space.low, np.full(goal_dim, -np.inf)])
            high = np.concatenate([original_obs_space.high, np.full(goal_dim, np.inf)])
            
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=original_obs_space.dtype
            )
        else:
            # Fallback for other space types
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(original_obs_space.shape[0] + goal_dim,),
                dtype=np.float32
            )
        
        self._last_goal = None
        print(f"GoalConditionedWrapper: Observation space changed from {original_obs_space.shape} to {self.observation_space.shape}")
    
    def observation(self, obs):
        """Convert observation to goal-conditioned format."""
        return self._create_goal_conditioned_obs(obs, self._last_goal)
    
    def _create_goal_conditioned_obs(self, agent_obs, goal_pos):
        """Create goal-conditioned observation."""
        if goal_pos is None:
            # If no goal available, use zeros (shouldn't happen in practice)
            goal_pos = np.zeros(2)
            print("Warning: No goal position available, using zeros")
        
        # Ensure both are numpy arrays
        agent_pos = np.array(agent_obs, dtype=np.float32)
        goal_pos = np.array(goal_pos, dtype=np.float32)
        
        # Concatenate agent position and goal position
        goal_conditioned_obs = np.concatenate([agent_pos, goal_pos])
        
        return goal_conditioned_obs
    
    def reset(self, **kwargs):
        """Reset environment and extract goal from info."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract goal from info
        if isinstance(info, dict) and 'goal' in info:
            self._last_goal = info['goal']
        else:
            print("Warning: No goal found in reset info")
            self._last_goal = np.zeros(2)
        
        return self.observation(obs), info
    
    def step(self, action):
        """Step environment and update goal if needed."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update goal if it's in step info (it should be)
        if isinstance(info, dict) and 'goal' in info:
            self._last_goal = info['goal']
        
        return self.observation(obs), reward, terminated, truncated, info


class RelativeGoalWrapper(gym.ObservationWrapper):
    """
    Alternative wrapper that provides relative goal position instead of absolute.
    Observation format: [agent_x, agent_y, goal_relative_x, goal_relative_y]
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get original observation space
        original_obs_space = env.observation_space
        
        # Create new observation space for [agent_pos, relative_goal]
        if isinstance(original_obs_space, spaces.Box):
            # Relative goals can be anywhere, so use wide bounds
            low = np.concatenate([original_obs_space.low, np.full(2, -50.0)])  # Assume max maze size ~50
            high = np.concatenate([original_obs_space.high, np.full(2, 50.0)])
            
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=original_obs_space.dtype
            )
        else:
            self.observation_space = spaces.Box(
                low=-50.0,
                high=50.0,
                shape=(original_obs_space.shape[0] + 2,),
                dtype=np.float32
            )
        
        self._last_goal = None
        print(f"RelativeGoalWrapper: Observation space changed from {original_obs_space.shape} to {self.observation_space.shape}")
    
    def observation(self, obs):
        """Convert observation to relative goal format."""
        return self._create_relative_goal_obs(obs, self._last_goal)
    
    def _create_relative_goal_obs(self, agent_obs, goal_pos):
        """Create relative goal observation."""
        if goal_pos is None:
            # If no goal available, use zeros
            relative_goal = np.zeros(2)
            print("Warning: No goal position available, using zeros")
        else:
            # Calculate relative goal position
            agent_pos = np.array(agent_obs[:2], dtype=np.float32)  # Take first 2 dims as position
            goal_pos = np.array(goal_pos[:2], dtype=np.float32)    # Take first 2 dims as goal
            relative_goal = goal_pos - agent_pos
        
        # Concatenate agent position and relative goal
        agent_pos = np.array(agent_obs, dtype=np.float32)
        obs_with_relative_goal = np.concatenate([agent_pos, relative_goal])
        
        return obs_with_relative_goal
    
    def reset(self, **kwargs):
        """Reset environment and extract goal from info."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract goal from info
        if isinstance(info, dict) and 'goal' in info:
            self._last_goal = info['goal']
        else:
            print("Warning: No goal found in reset info")
            self._last_goal = np.zeros(2)
        
        return self.observation(obs), info
    
    def step(self, action):
        """Step environment and update goal if needed."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update goal if it's in step info
        if isinstance(info, dict) and 'goal' in info:
            self._last_goal = info['goal']
        
        return self.observation(obs), reward, terminated, truncated, info


def test_wrappers():
    """Test both goal-conditioned wrappers."""
    import ogbench
    
    print("🧪 Testing Goal-Conditioned Wrappers")
    print("=" * 50)
    
    # Test original environment
    print("\n1. Original Environment:")
    env = gym.make("pointmaze-medium-v0", render_mode=None)
    obs, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    print(f"   Goal from info: {info.get('goal', 'NOT AVAILABLE')}")
    env.close()
    
    # Test GoalConditionedWrapper
    print("\n2. GoalConditionedWrapper:")
    env = gym.make("pointmaze-medium-v0", render_mode=None)
    env = GoalConditionedWrapper(env)
    obs, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    print(f"   Agent position: {obs[:2]}")
    print(f"   Goal position: {obs[2:4]}")
    
    # Take a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Step {i+1} - Agent: {obs[:2]}, Goal: {obs[2:4]}, Distance: {np.linalg.norm(obs[:2] - obs[2:4]):.3f}")
    env.close()
    
    # Test RelativeGoalWrapper  
    print("\n3. RelativeGoalWrapper:")
    env = gym.make("pointmaze-medium-v0", render_mode=None)
    env = RelativeGoalWrapper(env)
    obs, info = env.reset(seed=42)
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    print(f"   Agent position: {obs[:2]}")
    print(f"   Relative goal: {obs[2:4]}")
    
    # Take a few steps
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"   Step {i+1} - Agent: {obs[:2]}, Relative: {obs[2:4]}, Distance: {np.linalg.norm(obs[2:4]):.3f}")
    env.close()
    
    print("\n✅ Wrapper tests complete!")


if __name__ == "__main__":
    test_wrappers() 