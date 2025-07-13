#!/usr/bin/env python3
"""
Human intervention example for OGBench PointMaze environments.
This script runs a simple policy (random or basic heuristic) and allows human 
intervention via gamepad or keyboard when needed.
"""

import time
import pygame
import gymnasium as gym
import numpy as np
import os

# Fix WSL window positioning issues  
os.environ['SDL_VIDEO_CENTERED'] = '1'  # Center windows

# Import ogbench to register environments
import ogbench

# Import our teleoperation interface and wrapper
from ogbench.ui import TeleopPoint2D
from ogbench.wrappers import HumanInterventionWrapper


class SimplePolicy:
    """
    A simple policy for demonstration purposes.
    Can be random or a basic goal-directed heuristic.
    """
    
    def __init__(self, policy_type="heuristic"):
        self.policy_type = policy_type
        
    def get_action(self, obs, env):
        """
        Get action based on the policy type.
        
        Args:
            obs: Current observation
            env: Environment (to access goal information)
            
        Returns:
            action: Action to take
        """
        if self.policy_type == "random":
            return env.action_space.sample()
        elif self.policy_type == "heuristic":
            return self._heuristic_action(obs, env)
        else:
            return np.zeros(env.action_space.shape)
    
    def _heuristic_action(self, obs, env):
        """
        Simple heuristic: move towards the goal.
        """
        try:
            # Try to get current position and goal from observation/info
            # This assumes the observation contains position information
            if hasattr(env, 'cur_goal_xy') and hasattr(env, 'get_xy'):
                current_pos = env.get_xy()
                goal_pos = env.cur_goal_xy
                
                # Calculate direction to goal
                direction = goal_pos - current_pos
                
                # Normalize and scale
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                    
                # Scale to reasonable action magnitude
                action = direction * 0.5
                return action
            else:
                # Fallback to random if we can't access position/goal
                return env.action_space.sample() * 0.3
                
        except Exception:
            # Safe fallback
            return env.action_space.sample() * 0.3





def main():
    """
    Main function to run human intervention demo.
    """
    # List of available point maze environments
    available_envs = [
        'pointmaze-medium-v0',
        'pointmaze-large-v0', 
        'pointmaze-giant-v0',
        'pointmaze-teleport-v0'
    ]
    
    print("=== OGBench Point Maze Human Intervention Demo ===")
    
    # Use visual rendering mode
    render_mode = "human"
    print("🎯 Starting in visual mode...")
    
    print("\nAvailable environments:")
    for i, env_name in enumerate(available_envs):
        print(f"  {i+1}. {env_name}")
    
    # Let user choose environment
    while True:
        try:
            choice = int(input(f"\nSelect environment (1-{len(available_envs)}): ")) - 1
            if 0 <= choice < len(available_envs):
                selected_env = available_envs[choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Let user choose policy type
    policy_types = ["heuristic", "random", "zero"]
    print("\nPolicy types:")
    for i, policy_type in enumerate(policy_types):
        print(f"  {i+1}. {policy_type}")
    
    while True:
        try:
            choice = int(input(f"\nSelect policy type (1-{len(policy_types)}): ")) - 1
            if 0 <= choice < len(policy_types):
                selected_policy = policy_types[choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nStarting {selected_env} with {selected_policy} policy...")
    
    # Create the environment with larger window
    try:
        env = gym.make(
            selected_env,
            render_mode=render_mode,
            max_episode_steps=1000,
            width=800,
            height=600
        )
    except Exception as e:
        print(f"Failed to create environment: {e}")
        print("This might be a MuJoCo/OpenGL issue.")
        return

    try:
        # Initialize policy
        policy = SimplePolicy(selected_policy)
        
        # Initialize teleoperation interface (with keyboard fallback)
        teleop_agent = TeleopPoint2D(deadzone=0.15, use_keyboard_fallback=True)
        
        # Wrap environment for human intervention
        env = HumanInterventionWrapper(
            env, 
            teleop_agent,
            threshold=0.2,   # Higher threshold to avoid accidental overrides
            hold_time=0.5    # Stay in override mode for 0.5s after input
        )
        
        # Print controls
        teleop_agent.print_controls()
        
        print("\n=== INTERVENTION DEMO ===")
        print("🤖 The policy will control the agent by default.")
        print("🎮 Use your controller or keyboard to intervene when needed!")
        print("📊 Watch the console for intervention statistics")
        print("❌ Press Ctrl+C to exit.\n")
        
        obs, info = env.reset()
        
        running = True
        episode_count = 0
        step_count = 0
        total_intervention_steps = 0
        
        while running:
            # Get policy action
            policy_action = policy.get_action(obs, env.env)  # env.env to access unwrapped env
            
            # Step environment (wrapper will decide between policy and human action)
            obs, reward, terminated, truncated, info = env.step(policy_action)
            step_count += 1
            
            # Track intervention
            if info.get("human_override", False):
                total_intervention_steps += 1
                human_action_str = np.array2string(info['intervene_action'], 
                                                 formatter={'float_kind': lambda x: "%.2f" % x})
                print(f"\rHuman override ACTIVE! Action: {human_action_str}        ", end="", flush=True)
            else:
                policy_action_str = np.array2string(info.get('policy_action_used', policy_action),
                                                  formatter={'float_kind': lambda x: "%.2f" % x})
                if render_mode is None:
                    # In headless mode, print occasionally
                    if step_count % 20 == 0:
                        print(f"\rStep {step_count}: Policy ACTIVE, Action: {policy_action_str}        ", flush=True)
                else:
                    print(f"\rPolicy ACTIVE!        Action: {policy_action_str}        ", end="", flush=True)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Check for episode termination
            if terminated or truncated:
                episode_count += 1
                intervention_percentage = (total_intervention_steps / step_count) * 100 if step_count > 0 else 0
                
                print(f"\n\nEpisode {episode_count} finished after {step_count} steps!")
                print(f"Human intervention: {total_intervention_steps}/{step_count} steps ({intervention_percentage:.1f}%)")
                
                if terminated:
                    print("Goal reached! 🎉")
                else:
                    print("Episode truncated (time limit reached)")
                    
                print("Resetting environment...")
                time.sleep(1)
                obs, info = env.reset()
                step_count = 0
                total_intervention_steps = 0

            # Small delay for smooth operation
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("This might be related to MuJoCo/OpenGL setup.")
    finally:
        print("Closing environment.")
        try:
            env.close()
        except:
            pass
        pygame.quit()


if __name__ == "__main__":
    main() 