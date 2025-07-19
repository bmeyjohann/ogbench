"""
Speed wrapper to make PointMaze movement responsive.
"""

import numpy as np
import gymnasium as gym

class SpeedWrapper(gym.ActionWrapper):
    """
    Wrapper to make agent movement faster and more responsive.
    
    Applies consistent speed boost regardless of mode to ensure
    uniform training experience and replay buffer consistency.
    """
    
    def __init__(self, env, speed_multiplier=3.0):
        super().__init__(env)
        self.speed_multiplier = speed_multiplier
        print(f"SpeedWrapper: {speed_multiplier}x faster movement (consistent speed)")
    
    def action(self, action):
        """Scale action for faster, more responsive movement."""
        return np.array(action) * self.speed_multiplier


def test_speed_wrapper():
    """Test the speed wrapper."""
    import ogbench
    
    print("🏃 Testing Speed Wrapper")
    print("=" * 40)
    
    # Test original environment
    print("\n1. Original Environment (slow):")
    env = gym.make("pointmaze-medium-v0", render_mode=None)
    obs, _ = env.reset(seed=42)
    initial_pos = obs.copy()
    print(f"   Initial position: {obs}")
    
    action = np.array([1.0, 0.0])  # Move right
    for i in range(3):
        obs, _, _, _, _ = env.step(action)
        distance_moved = np.linalg.norm(obs - initial_pos)
        print(f"   Step {i+1}: {obs} (total distance: {distance_moved:.3f})")
    env.close()
    
    # Test with speed wrapper
    print("\n2. With SpeedWrapper (faster and consistent):")
    env = gym.make("pointmaze-medium-v0", render_mode=None)
    env = SpeedWrapper(env, speed_multiplier=3.0)
    obs, _ = env.reset(seed=42)
    initial_pos = obs.copy()
    print(f"   Initial position: {obs}")
    
    action = np.array([1.0, 0.0])  # Move right
    for i in range(3):
        obs, _, _, _, _ = env.step(action)
        distance_moved = np.linalg.norm(obs - initial_pos)
        print(f"   Step {i+1}: {obs} (total distance: {distance_moved:.3f})")
    env.close()
    
    print("\n✅ Speed test complete!")
    print("Consistent speed across all modes ensures uniform training!")


if __name__ == "__main__":
    test_speed_wrapper() 