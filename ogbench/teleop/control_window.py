"""
A teleoperation interface that uses a dedicated Pygame window for keyboard input.
"""

import numpy as np
import pygame
from typing import Optional, Dict, Any


class ControlWindowTeleop:
    """Teleoperation interface with dedicated control window for keyboard input."""
    
    def __init__(self, width=500, height=400, show_debug_info=True):
        pygame.init()
        self.width = width
        self.height = height
        self.show_debug_info = show_debug_info
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("🎮 Teleoperation Control Panel")
        
        # Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.blue = (0, 0, 255)
        self.yellow = (255, 255, 0)
        
        # Font
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        self.quit_requested = False
        self.current_obs = None
        self.current_info = None
        self.step_count = 0
        
        print("🎮 Control Window created!")
        print("   Focus the control window and use arrow keys to move")
        print("   ESC to exit")
        
    def get_action(self):
        """Get action from keyboard input."""
        # Handle pygame events
        pygame.event.pump()
        keys_pressed = pygame.key.get_pressed()
        action = np.array([0.0, 0.0])
        
        # Handle discrete key presses for movement
        if keys_pressed[pygame.K_UP]:
            action[1] = 1.0
        if keys_pressed[pygame.K_DOWN]:
            action[1] = -1.0
        if keys_pressed[pygame.K_LEFT]:
            action[0] = -1.0
        if keys_pressed[pygame.K_RIGHT]:
            action[0] = 1.0
        
        # Handle other events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_requested = True
        
        # Update display
        self._update_display()
        
        return action
    
    def _update_display(self):
        """Update the control window display."""
        self.screen.fill(self.black)
        
        # Title
        title = self.large_font.render("🎮 TELEOPERATION", True, self.yellow)
        self.screen.blit(title, (10, 10))
        
        # Step counter
        step_text = self.font.render(f"Step: {self.step_count}", True, self.white)
        self.screen.blit(step_text, (10, 50))
        
        # Controls section
        controls_y = 80
        controls = [
            "KEYBOARD CONTROLS:",
            "↑↓←→ Arrow Keys: Move agent",
            "ESC: Exit",
            "",
            "FOCUS THIS WINDOW for control!"
        ]
        
        for i, control in enumerate(controls):
            if i == 0:
                color = self.yellow
                font = self.font
            elif "FOCUS" in control:
                color = self.green
                font = self.font
            else:
                color = self.white
                font = self.small_font
            
            text = font.render(control, True, color)
            self.screen.blit(text, (10, controls_y + i * 25))
        
        # Current observation info (if available and debug enabled)
        if self.show_debug_info and self.current_obs is not None:
            obs_y = 200
            obs_title = self.font.render("CURRENT STATE:", True, self.blue)
            self.screen.blit(obs_title, (10, obs_y))
            
            obs = self.current_obs
            if hasattr(obs, '__len__') and len(obs) >= 2:
                pos_text = self.small_font.render(f"Position: ({obs[0]:.3f}, {obs[1]:.3f})", True, self.white)
                self.screen.blit(pos_text, (10, obs_y + 25))
                
                if len(obs) == 4:
                    rel_goal_text = self.small_font.render(f"Rel Goal: ({obs[2]:.3f}, {obs[3]:.3f})", True, self.white)
                    self.screen.blit(rel_goal_text, (10, obs_y + 45))
                    
                    distance = np.linalg.norm(obs[2:4])
                    dist_text = self.small_font.render(f"Distance: {distance:.3f}", True, self.white)
                    self.screen.blit(dist_text, (10, obs_y + 65))
                
                # Goal info from environment info
                if self.current_info and 'goal' in self.current_info:
                    goal = self.current_info['goal']
                    goal_text = self.small_font.render(f"Goal: ({goal[0]:.3f}, {goal[1]:.3f})", True, self.white)
                    self.screen.blit(goal_text, (10, obs_y + 85))
        
        pygame.display.flip()
    
    def update_state(self, obs, info, step_count):
        """Update the current state for display."""
        self.current_obs = obs
        self.current_info = info
        self.step_count = step_count
    
    def reset(self):
        """Reset the interface."""
        self.step_count = 0
        self.current_obs = None
        self.current_info = None
        self.quit_requested = False
    
    def should_quit(self):
        """Check if quit was requested."""
        return self.quit_requested
    
    def close(self):
        """Close the control window."""
        pygame.quit()
    
    def print_controls(self):
        """Print control information (for compatibility)."""
        print("🎮 Control Window Teleoperation:")
        print("   Focus the control window and use arrow keys")
        print("   ESC to exit") 