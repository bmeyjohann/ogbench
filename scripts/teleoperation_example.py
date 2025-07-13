#!/usr/bin/env python3
"""
Direct teleoperation example for OGBench PointMaze environments.
This script allows you to directly control the point agent using gamepad or keyboard.
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
from ogbench.wrappers import DirectTeleopWrapper





def main():
    """
    Main function to run direct teleoperation on a point maze environment.
    """
    # List of available point maze environments
    available_envs = [
        'pointmaze-medium-v0',
        'pointmaze-large-v0', 
        'pointmaze-giant-v0',
        'pointmaze-teleport-v0'
    ]
    
    print("=== OGBench Point Maze Teleoperation ===")
    
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
    
    print(f"\nStarting {selected_env}...")
    
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
        print("This might be a MuJoCo/OpenGL issue. Try installing proper graphics drivers.")
        return

    try:
        # Initialize teleoperation interface (with keyboard fallback)
        teleop_agent = TeleopPoint2D(deadzone=0.1, use_keyboard_fallback=True)
        
        # Wrap environment for direct teleoperation
        env = DirectTeleopWrapper(env, teleop_agent)
        
        # Print controls
        teleop_agent.print_controls()
        
        print("\n=== TELEOPERATION MODE ===")
        print("🎮 You have full control of the agent!")
        print("🎯 Navigate to the goal (shown in the visualization)")
        print("❌ Press Ctrl+C to exit.\n")
        
        obs, info = env.reset()
        
        running = True
        episode_count = 0
        step_count = 0
        
        while running:
            # Get action from teleop interface and step environment
            obs, reward, terminated, truncated, info = env.step()
            step_count += 1
            
            # In headless mode, provide text feedback
            if render_mode is None and step_count % 10 == 0:
                # Try to get position information
                try:
                    if hasattr(env.env, 'get_xy'):
                        current_pos = env.env.get_xy()
                        goal_pos = getattr(env.env, 'cur_goal_xy', [0, 0])
                        distance = np.linalg.norm(current_pos - goal_pos)
                        print(f"Step {step_count}: Pos={current_pos}, Goal={goal_pos}, Dist={distance:.2f}")
                except:
                    print(f"Step {step_count}: Reward={reward:.3f}")
            
            # Handle pygame events (for window close and keyboard input)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            # Check for episode termination
            if terminated or truncated:
                episode_count += 1
                print(f"\nEpisode {episode_count} finished after {step_count} steps!")
                
                if terminated:
                    print("Goal reached! 🎉")
                else:
                    print("Episode truncated (time limit reached)")
                    
                print("Resetting environment...")
                time.sleep(1)
                obs, info = env.reset()
                step_count = 0

            # Small delay for smooth operation
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError during execution: {e}")
        print("This might be related to MuJoCo/OpenGL setup.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Try running in headless mode")
        print("2. Check MuJoCo installation: python -c 'import mujoco; print(mujoco.__version__)'")
        print("3. Install missing graphics libraries")
    finally:
        print("Closing environment.")
        try:
            env.close()
        except:
            pass
        pygame.quit()


if __name__ == "__main__":
    main() 