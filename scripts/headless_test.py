#!/usr/bin/env python3
"""
Headless test for OGBench teleoperation system.
This script tests the system without any visual rendering, 
perfect for WSL or headless environments.
"""

import time
import numpy as np
import gymnasium as gym

# Import ogbench to register environments
import ogbench

# Import our teleoperation components
from ogbench.ui import TeleopPoint2D
from ogbench.wrappers import HumanInterventionWrapper, DirectTeleopWrapper


class SimulatedInput:
    """
    Simulates human input for testing purposes.
    Moves in a simple pattern to test the intervention system.
    """
    def __init__(self, pattern="circle"):
        self.step_count = 0
        self.pattern = pattern
        
    def get_action(self):
        """Generate a simple movement pattern."""
        self.step_count += 1
        
        if self.pattern == "circle":
            # Move in a circular pattern
            angle = (self.step_count * 0.1) % (2 * np.pi)
            return np.array([np.cos(angle) * 0.3, np.sin(angle) * 0.3])
        elif self.pattern == "square":
            # Move in a square pattern
            phase = (self.step_count // 20) % 4
            if phase == 0:
                return np.array([0.5, 0.0])  # Right
            elif phase == 1:
                return np.array([0.0, 0.5])  # Up
            elif phase == 2:
                return np.array([-0.5, 0.0])  # Left
            else:
                return np.array([0.0, -0.5])  # Down
        else:
            # Random movement
            return np.random.uniform(-0.5, 0.5, 2)
    
    def get_button_states(self):
        return {}
    
    def reset(self):
        self.step_count = 0


def test_headless_direct_control():
    """Test direct teleoperation in headless mode."""
    print("=== Testing Direct Teleoperation (Headless) ===")
    
    try:
        # Create environment without rendering
        env = gym.make('pointmaze-medium-v0', render_mode=None)
        
        # Create simulated input
        sim_input = SimulatedInput(pattern="circle")
        
        # Wrap environment
        env = DirectTeleopWrapper(env, sim_input)
        
        obs, info = env.reset()
        print(f"Environment initialized. Observation shape: {obs.shape}")
        
        # Run for a few steps
        for step in range(20):
            obs, reward, terminated, truncated, info = env.step()
            
            # Print status every 5 steps
            if step % 5 == 0:
                print(f"Step {step}: Reward={reward:.3f}, Terminated={terminated}")
                
                # Try to get position if available
                try:
                    if hasattr(env.env, 'get_xy'):
                        pos = env.env.get_xy()
                        print(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}]")
                except:
                    pass
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        env.close()
        print("✅ Direct teleoperation test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Direct teleoperation test failed: {e}\n")
        return False


def test_headless_intervention():
    """Test human intervention in headless mode."""
    print("=== Testing Human Intervention (Headless) ===")
    
    try:
        # Create environment without rendering
        env = gym.make('pointmaze-medium-v0', render_mode=None)
        
        # Create simulated input (will trigger interventions)
        sim_input = SimulatedInput(pattern="square")
        
        # Wrap environment
        env = HumanInterventionWrapper(env, sim_input, threshold=0.1, hold_time=0.3)
        
        obs, info = env.reset()
        print(f"Environment initialized. Observation shape: {obs.shape}")
        
        intervention_count = 0
        policy_count = 0
        
        # Run for more steps to see intervention behavior
        for step in range(50):
            # Simple policy: move towards goal (if we can detect it)
            policy_action = np.array([0.1, 0.1])  # Simple constant policy
            
            obs, reward, terminated, truncated, info = env.step(policy_action)
            
            # Track intervention vs policy control
            if info.get("human_override", False):
                intervention_count += 1
                if step % 10 == 0:
                    print(f"Step {step}: HUMAN intervention active")
            else:
                policy_count += 1
                if step % 10 == 0:
                    print(f"Step {step}: POLICY control active")
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        total_steps = intervention_count + policy_count
        intervention_pct = (intervention_count / total_steps) * 100 if total_steps > 0 else 0
        
        print(f"Intervention statistics:")
        print(f"  Human control: {intervention_count}/{total_steps} steps ({intervention_pct:.1f}%)")
        print(f"  Policy control: {policy_count}/{total_steps} steps ({100-intervention_pct:.1f}%)")
        
        env.close()
        print("✅ Human intervention test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Human intervention test failed: {e}\n")
        return False


def test_teleop_interface():
    """Test the teleoperation interface creation."""
    print("=== Testing TeleopPoint2D Interface ===")
    
    try:
        # Create interface with keyboard fallback
        teleop = TeleopPoint2D(deadzone=0.1, use_keyboard_fallback=True)
        
        # Test basic functionality
        action = teleop.get_action()
        print(f"Action shape: {action.shape}, values: {action}")
        
        buttons = teleop.get_button_states()
        print(f"Button states type: {type(buttons)}")
        
        teleop.reset()
        
        print("✅ TeleopPoint2D interface test passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ TeleopPoint2D interface test failed: {e}\n")
        return False


def main():
    """Run all headless tests."""
    print("🧪 OGBench Teleoperation Headless Tests")
    print("=" * 50)
    print("These tests run without any visual rendering.")
    print("Perfect for WSL or headless environments.\n")
    
    tests = [
        ("TeleopPoint2D Interface", test_teleop_interface),
        ("Direct Teleoperation", test_headless_direct_control),
        ("Human Intervention", test_headless_intervention),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if test_func():
            passed += 1
        time.sleep(0.5)  # Brief pause between tests
    
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The teleoperation system is working correctly.")
        print("\nThe system is ready for use. You can now:")
        print("1. Run interactive scripts with visual mode (if you fix rendering)")
        print("2. Use the system in your own code for headless operation")
        print("3. Integrate with your RL training loops")
    else:
        print("❌ Some tests failed. Check the error messages above.")
        print("This might indicate missing dependencies or environment issues.")
    
    print("\n📚 Next steps:")
    print("- Fix OpenGL/rendering issues for visual mode")
    print("- Try the interactive scripts: teleoperation_example.py, intervention_example.py")
    print("- Integrate teleoperation into your own projects")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main()) 