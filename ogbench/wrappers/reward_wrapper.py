"""
Reward wrapper for OGBench environments with detailed reward tracking.

This wrapper provides:
- Sparse rewards (goal reached = 1.0, otherwise 0.0) 
- Dense rewards (distance-based progress)
- Combined rewards (sparse + dense)
- Detailed metrics tracking
"""

import numpy as np
import gymnasium as gym


class DetailedRewardWrapper(gym.RewardWrapper):
    """
    Detailed reward wrapper that separates sparse and dense rewards for analysis.
    
    Provides three reward types:
    - sparse: Goal-only rewards (1.0 for goal, 0.0 otherwise)
    - dense: Distance-based progress rewards
    - combined: sparse + dense_weight * dense
    """
    
    def __init__(self, env, 
                 reward_type='sparse',
                 dense_reward_scale=0.1,
                 goal_reward=1.0,
                 step_penalty=0.0):
        """
        Initialize the detailed reward wrapper.
        
        Args:
            env: The base environment
            reward_type: 'sparse', 'dense', or 'combined'
            dense_reward_scale: Scale factor for dense rewards
            goal_reward: Reward for reaching goal
            step_penalty: Small penalty per step (negative reward for time)
        """
        super().__init__(env)
        
        self.reward_type = reward_type
        self.dense_reward_scale = dense_reward_scale
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # Track previous position for dense reward calculation
        self._prev_agent_pos = None
        self._goal_pos = None
        self._prev_distance = None
        
        # Episode statistics
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._episode_steps = 0
        self._goal_reached = False
    
    def _extract_positions(self, obs):
        """Extract agent and goal positions from observation."""
        # Assume obs format from FlexibleObsWrapper: [agent_x, agent_y, goal_x, goal_y, ...]
        if len(obs) >= 4:
            agent_pos = obs[:2]
            goal_pos = obs[2:4]
        else:
            # Fallback: use info from environment
            agent_pos = obs[:2]
            goal_pos = self._goal_pos if self._goal_pos is not None else np.zeros(2)
        
        return np.array(agent_pos), np.array(goal_pos)
    
    def reset(self, **kwargs):
        """Reset environment and tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract positions
        self._prev_agent_pos, self._goal_pos = self._extract_positions(obs)
        
        # Calculate initial distance
        self._prev_distance = np.linalg.norm(self._goal_pos - self._prev_agent_pos)
        
        # Reset episode tracking
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._episode_steps = 0
        self._goal_reached = False
        
        # Add initial metrics to info
        info.update({
            'episode_sparse_reward': self._episode_sparse_reward,
            'episode_dense_reward': self._episode_dense_reward,
            'episode_steps': self._episode_steps,
            'goal_reached': self._goal_reached,
            'distance_to_goal': self._prev_distance,
        })
        
        return obs, info
    
    def reward(self, reward):
        """Calculate detailed reward based on type."""
        # Get current observation (last observation from environment)
        try:
            # Get current state
            current_obs = self.unwrapped.get_ob()  # Get raw observation
            current_agent_pos, current_goal_pos = self._extract_positions(current_obs)
        except:
            # Fallback if get_ob() not available
            current_agent_pos = self._prev_agent_pos
            current_goal_pos = self._goal_pos
        
        # Calculate current distance
        current_distance = np.linalg.norm(current_goal_pos - current_agent_pos)
        
        # Calculate rewards
        sparse_reward = float(reward)  # Original reward (1.0 for goal, 0.0 otherwise)
        
        # Dense reward based on distance progress
        if self._prev_distance is not None:
            distance_progress = self._prev_distance - current_distance
            dense_reward = distance_progress * self.dense_reward_scale
        else:
            dense_reward = 0.0
        
        # Step penalty
        step_reward = -self.step_penalty
        
        # Update tracking
        self._episode_sparse_reward += sparse_reward
        self._episode_dense_reward += dense_reward
        self._episode_steps += 1
        
        if sparse_reward > 0:
            self._goal_reached = True
        
        # Update previous state
        self._prev_agent_pos = current_agent_pos.copy()
        self._prev_distance = current_distance
        
        # Return reward based on type
        if self.reward_type == 'sparse':
            dense_reward = 0.0
        elif self.reward_type == 'dense':
            sparse_reward = 0.0
        elif self.reward_type == 'combined':
            pass
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        total_reward = sparse_reward + dense_reward + step_reward
        
        return total_reward
    
    def step(self, action):
        """Step with detailed reward tracking."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate detailed reward
        detailed_reward = self.reward(reward)
        
        # Get current positions for metrics
        try:
            current_agent_pos, current_goal_pos = self._extract_positions(obs)
            current_distance = np.linalg.norm(current_goal_pos - current_agent_pos)
        except:
            current_distance = self._prev_distance if self._prev_distance is not None else 0.0
        
        # Update info with detailed metrics
        info.update({
            'sparse_reward': float(reward),  # Original reward
            'dense_reward': (self._episode_dense_reward - (self._episode_dense_reward - 
                           (self._prev_distance - current_distance) * self.dense_reward_scale)),
            'total_reward': detailed_reward,
            'episode_sparse_reward': self._episode_sparse_reward,
            'episode_dense_reward': self._episode_dense_reward,
            'episode_steps': self._episode_steps,
            'goal_reached': self._goal_reached,
            'distance_to_goal': current_distance,
            'reward_type': self.reward_type,
        })
        
        return obs, detailed_reward, terminated, truncated, info


def test_reward_wrapper():
    """Test the reward wrapper with different configurations."""
    import ogbench
    from ogbench.wrappers import FlexibleObsWrapper
    
    print("🧪 Testing DetailedRewardWrapper")
    print("=" * 50)
    
    reward_types = ['sparse', 'dense', 'combined']
    
    for reward_type in reward_types:
        print(f"\n{reward_type.upper()} Rewards:")
        
        env = gym.make("pointmaze-arena-v0", render_mode=None)
        env = FlexibleObsWrapper(env, include_goal=True)
        env = DetailedRewardWrapper(env, 
                                  reward_type=reward_type,
                                  dense_reward_scale=0.01)
        
        obs, info = env.reset(seed=42)
        print(f"   Initial distance: {info['distance_to_goal']:.3f}")
        
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step+1}: reward={reward:.4f}, sparse={info.get('sparse_reward', 0):.4f}, "
                  f"distance={info['distance_to_goal']:.3f}")
            
            if term:
                print(f"   🎯 Goal reached!")
                break
        
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Goal reached: {info['goal_reached']}")
        
        env.close()
    
    print("\n✅ Reward wrapper tests complete!")


if __name__ == "__main__":
    test_reward_wrapper()