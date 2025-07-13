import pygame
import numpy as np
from collections import defaultdict

# -----------------------------------------------------------------------------
# PygameGamepad: A simple wrapper for reading joystick state.
# -----------------------------------------------------------------------------
class PygameGamepad:
    """
    Initializes a PyGame joystick and provides methods to read its state.
    Handles the case where no joystick is connected gracefully.
    """
    def __init__(self):
        pygame.init()
        pygame.joystick.init()  # Explicitly initialize joystick subsystem
        self.has_joystick = False
        
        try:
            # Force refresh of joystick detection
            pygame.joystick.quit()
            pygame.joystick.init()
            
            joystick_count = pygame.joystick.get_count()
            print(f"[DEBUG] Scanning for controllers... Found {joystick_count}")
            
            if joystick_count > 0:
                # Try to initialize the first joystick
                self.joy = pygame.joystick.Joystick(0)
                self.joy.init()
                self.has_joystick = True
                
                controller_name = self.joy.get_name()
                print(f"[INFO] 🎮 Controller Connected: {controller_name}")
                print(f"[INFO] Axes: {self.joy.get_numaxes()}, Buttons: {self.joy.get_numbuttons()}")
                
                # Special handling for Xbox 360 controllers
                if 'xbox' in controller_name.lower() or '360' in controller_name.lower():
                    print("[INFO] ✅ Xbox 360 controller detected!")
            else:
                print("[INFO] No joystick detected. Will use keyboard fallback.")
                print("[DEBUG] Try: lsusb | grep -i xbox")
                print("[DEBUG] Or: ls /dev/input/js*")
                
        except pygame.error as e:
            print(f"[INFO] Joystick initialization failed: {e}")
            print("[INFO] Will use keyboard fallback.")
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            print("[INFO] Will use keyboard fallback.")

    def get_axes(self):
        """
        Reads the raw axis values from the joystick.
        Returns a dictionary of relevant axis values in the range [-1.0, 1.0].
        Returns zeros if no joystick is available.
        """
        if not self.has_joystick:
            return {"x": 0.0, "y": 0.0}
            
        # This is crucial for updating the joystick state
        pygame.event.pump()

        # Standard mapping for most controllers (e.g., Xbox, PS4/5)
        # Left Stick X: -1 (left) to 1 (right)
        # Left Stick Y: -1 (up) to 1 (down) <-- Note: Pygame's Y is often inverted
        ax_x = self.joy.get_axis(0)
        ax_y = self.joy.get_axis(1)
        
        return {"x": ax_x, "y": ax_y}

    def get_buttons(self):
        """
        Reads the state of the face buttons and shoulder buttons.
        Returns a dictionary with boolean values.
        Returns empty dict if no joystick is available.
        """
        if not self.has_joystick:
            return {}
            
        pygame.event.pump()
        # Common button mapping for PlayStation controllers
        return {
            "cross":    bool(self.joy.get_button(0)),  # X button
            "circle":   bool(self.joy.get_button(1)),  # O button
            "square":   bool(self.joy.get_button(2)),
            "triangle": bool(self.joy.get_button(3)),
            "l1":       bool(self.joy.get_button(4)),
            "r1":       bool(self.joy.get_button(5)),
        }

# -----------------------------------------------------------------------------
# KeyboardInterface: Fallback keyboard input handling
# -----------------------------------------------------------------------------
class KeyboardInterface:
    """
    Handles keyboard input as a fallback when no gamepad is available.
    Uses arrow keys for movement and space for action buttons.
    """
    def __init__(self):
        pygame.init()
        self.key_state = defaultdict(bool)
        
    def get_axes(self):
        """
        Converts keyboard input to joystick-like axis values.
        Arrow keys simulate joystick movement.
        """
        # Update pygame events to get current key state
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # Convert arrow keys to axis values
        ax_x = 0.0
        ax_y = 0.0
        
        if keys[pygame.K_LEFT]:
            ax_x = -1.0
        elif keys[pygame.K_RIGHT]:
            ax_x = 1.0
            
        if keys[pygame.K_UP]:
            ax_y = -1.0  # Negative Y for up (matching joystick convention)
        elif keys[pygame.K_DOWN]:
            ax_y = 1.0
            
        return {"x": ax_x, "y": ax_y}
        
    def get_buttons(self):
        """
        Maps keyboard keys to gamepad button equivalents.
        """
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        return {
            "cross": keys[pygame.K_SPACE],      # Space for cross/X button
            "circle": keys[pygame.K_ESCAPE],    # Escape for circle button
            "square": keys[pygame.K_q],         # Q for square
            "triangle": keys[pygame.K_e],       # E for triangle
            "l1": keys[pygame.K_LSHIFT],        # Left shift for L1
            "r1": keys[pygame.K_RSHIFT],        # Right shift for R1
        }

# -----------------------------------------------------------------------------
# TeleopPoint2D: Main class for 2D teleoperation with gamepad/keyboard support
# -----------------------------------------------------------------------------
class TeleopPoint2D:
    """
    Reads input from gamepad or keyboard and converts them to a 2D action vector [force_x, force_y]
    for the PointMaze environment.
    
    The action space of PointMaze is Box(-1.0, 1.0, (2,)).
    - action[0]: Force in the x-direction.
    - action[1]: Force in the y-direction.
    
    Automatically falls back to keyboard if no gamepad is detected.
    """
    def __init__(self, deadzone=0.15, use_keyboard_fallback=True):
        """
        Initializes the teleoperation agent.
        
        Args:
            deadzone (float): Stick inputs with an absolute value smaller than this
                              will be ignored (range 0.0 to 1.0).
            use_keyboard_fallback (bool): Whether to use keyboard as fallback when no gamepad is available.
        """
        self.deadzone = deadzone
        self.action = np.zeros(2, dtype=np.float32)
        self.use_keyboard_fallback = use_keyboard_fallback
        
        # Try to initialize gamepad first
        self.gamepad = PygameGamepad()
        
        # If no gamepad and fallback is enabled, use keyboard
        if not self.gamepad.has_joystick and use_keyboard_fallback:
            self.device = KeyboardInterface()
            self.using_keyboard = True
            print("[INFO] Using keyboard controls: Arrow keys to move, Space to reset")
            print("[INFO] Controls: ← → ↑ ↓ for movement, Space for actions")
        else:
            self.device = self.gamepad
            self.using_keyboard = False
            
        if not self.gamepad.has_joystick and not use_keyboard_fallback:
            raise IOError("No joystick detected and keyboard fallback is disabled.")

    def get_action(self):
        """
        Calculates and returns the 2D action vector based on input device.
        
        Returns:
            np.ndarray: A 2D numpy array [force_x, force_y] with values in [-1, 1].
        """
        axes = self.device.get_axes()
        
        # Raw stick values
        raw_x = axes.get("x", 0.0)
        raw_y = axes.get("y", 0.0)

        # Apply deadzone (for keyboard, values are already 0 or ±1, so deadzone has less effect)
        force_x = raw_x if abs(raw_x) > self.deadzone else 0.0
        
        # Invert the Y-axis for intuitive control (up on stick/key = positive y force)
        force_y = -raw_y if abs(raw_y) > self.deadzone else 0.0
        
        self.action[0] = force_x
        self.action[1] = force_y
        
        return self.action.copy()

    def get_button_states(self):
        """
        A helper function to pass through button states from the device.
        Useful for triggering events like resetting the environment.
        """
        return self.device.get_buttons()
        
    def reset(self):
        """
        Reset method for compatibility with intervention wrappers.
        """
        pass
        
    def print_controls(self):
        """
        Print control instructions based on the current input method.
        """
        if self.using_keyboard:
            print("\n=== KEYBOARD CONTROLS ===")
            print("Movement: Arrow Keys (← → ↑ ↓)")
            print("Reset: Space")
            print("Quit: Escape")
            print("==========================\n")
        else:
            print("\n=== GAMEPAD CONTROLS ===")
            print("Movement: Left Stick")
            print("Reset: X/Cross Button")
            print("========================\n") 