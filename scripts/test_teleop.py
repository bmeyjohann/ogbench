#!/usr/bin/env python3
"""
Simple test script to verify the teleoperation system is working correctly.
This runs a basic test without requiring user interaction.
"""

import gymnasium as gym
import numpy as np

# Import ogbench to register environments
import ogbench

# Import our teleoperation components
from ogbench.ui import TeleopPoint2D
from ogbench.wrappers import HumanInterventionWrapper, DirectTeleopWrapper


class MockTeleopInterface:
    """
    Mock teleoperation interface for testing without actual gamepad/keyboard input.
    """
    def __init__(self):
        self.step_counter = 0
        
    def get_action(self):
        """Return a simple test action that moves right."""
        self.step_counter += 1
        # Move right for first 10 steps, then stop
        if self.step_counter <= 10:
            return np.array([0.5, 0.0])
        else:
            return np.array([0.0, 0.0])
    
    def get_button_states(self):
        return {}
    
    def reset(self):
        self.step_counter = 0


def test_direct_teleop():
    """Test direct teleoperation wrapper."""
    print("Testing DirectTeleopWrapper...")
    
    try:
        # Create environment
        env = gym.make('pointmaze-medium-v0', render_mode=None)  # No rendering for test
        
        # Create mock teleop interface
        teleop = MockTeleopInterface()
        
        # Wrap environment
        env = DirectTeleopWrapper(env, teleop)
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"  Initial obs shape: {obs.shape}")
        
        # Run a few steps
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step()
            print(f"  Step {i+1}: reward={reward:.3f}, info keys: {list(info.keys())}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("  ✓ DirectTeleopWrapper test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ DirectTeleopWrapper test failed: {e}")
        return False


def test_intervention_wrapper():
    """Test human intervention wrapper."""
    print("Testing HumanInterventionWrapper...")
    
    try:
        # Create environment
        env = gym.make('pointmaze-medium-v0', render_mode=None)  # No rendering for test
        
        # Create mock teleop interface
        teleop = MockTeleopInterface()
        
        # Wrap environment
        env = HumanInterventionWrapper(env, teleop, threshold=0.1, hold_time=0.2)
        
        # Test basic functionality
        obs, info = env.reset()
        print(f"  Initial obs shape: {obs.shape}")
        
        # Run a few steps with policy actions
        for i in range(15):
            policy_action = env.action_space.sample() * 0.1  # Small random actions
            obs, reward, terminated, truncated, info = env.step(policy_action)
            
            override_status = "HUMAN" if info.get("human_override", False) else "POLICY"
            print(f"  Step {i+1}: {override_status} control, reward={reward:.3f}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("  ✓ HumanInterventionWrapper test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ HumanInterventionWrapper test failed: {e}")
        return False


def test_teleop_interface():
    """Test teleoperation interface creation."""
    print("Testing TeleopPoint2D interface...")
    
    try:
        # Test with keyboard fallback enabled (should work without gamepad)
        teleop = TeleopPoint2D(deadzone=0.1, use_keyboard_fallback=True)
        
        # Test getting action (should return zeros since no input)
        action = teleop.get_action()
        print(f"  Action shape: {action.shape}, values: {action}")
        
        # Test getting button states
        buttons = teleop.get_button_states()
        print(f"  Button states: {buttons}")
        
        # Test reset
        teleop.reset()
        
        print("  ✓ TeleopPoint2D interface test passed!")
        return True
        
    except Exception as e:
        print(f"  ✗ TeleopPoint2D interface test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== OGBench Teleoperation System Test ===\n")
    
    tests = [
        test_teleop_interface,
        test_direct_teleop,
        test_intervention_wrapper,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print(f"=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The teleoperation system is ready to use.")
        print("\nTo try it interactively, run:")
        print("  python teleoperation_example.py")
        print("  python intervention_example.py")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 