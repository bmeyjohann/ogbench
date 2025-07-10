import time
import numpy as np
import gymnasium as gym


class HumanInterventionWrapper(gym.Wrapper):
    """
    Wraps a Gymnasium environment (like PointMaze) to allow for human intervention.
    
    This wrapper compares a policy's proposed action with an action from a
    teleoperation device (e.g., a joystick or keyboard). If the human provides input,
    their action overrides the policy's action for a short duration.
    
    This is useful for:
    - Guiding a learning agent out of a difficult state.
    - Providing demonstrations.
    - Debugging environment interactions.
    - Collecting human-corrected trajectories.
    """

    def __init__(self, env: gym.Env, teleop_interface, *, threshold: float = 0.1, hold_time: float = 0.5):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            teleop_interface: An object with a `get_action()` method that returns
                              a NumPy array of the same shape as the env's action space.
                              (e.g., an instance of TeleopPoint2D).
            threshold (float): The minimum norm of the human's action vector to
                               trigger an override. Prevents drift from a loose
                               joystick from taking over.
            hold_time (float): How long (in seconds) the human override should
                               remain active after the last detected movement.
        """
        super().__init__(env)
        
        # --- Type and Shape Checks for Robustness ---
        if not hasattr(teleop_interface, "get_action"):
            raise TypeError("teleop_interface must have a 'get_action' method.")
        
        self.teleop = teleop_interface
        self.threshold = threshold
        self.hold_time = hold_time
        
        # Timestamp of the last time a significant human action was detected.
        self._last_override_ts = 0.0
        
        print("[HumanInterventionWrapper] Initialized.")
        print(f"  - Override Threshold: {self.threshold}")
        print(f"  - Override Hold Time: {self.hold_time}s")

    def reset(self, **kwargs):
        """
        Resets the environment and the override timer.
        """
        # The standard Gymnasium reset returns obs, info
        obs, info = self.env.reset(**kwargs)
        
        if hasattr(self.teleop, "reset"):
            self.teleop.reset()
            
        self._last_override_ts = 0.0
        return obs, info

    def step(self, policy_action: np.ndarray):
        """
        Executes a step in the environment, deciding whether to use the policy's
        action or the human's action.

        Args:
            policy_action (np.ndarray): The action proposed by the autonomous policy.

        Returns:
            The standard (obs, reward, terminated, truncated, info) tuple from
            the wrapped environment's step function. The `info` dictionary is
            annotated with override information.
        """
        # 1. Read the action from the human teleoperation device.
        human_action = self.teleop.get_action()

        # 2. Check if the human's action is significant enough to trigger an override.
        #    We use the L2 norm (Euclidean distance) of the action vector.
        human_action_norm = np.linalg.norm(human_action)
        
        if human_action_norm > self.threshold:
            # If human is actively providing input, update the timestamp.
            self._last_override_ts = time.perf_counter()

        # 3. Determine if the override is currently active.
        #    It's active if the last significant input was within the `hold_time`.
        is_human_intervening = (time.perf_counter() - self._last_override_ts) < self.hold_time
        
        # 4. Select the action to send to the environment.
        action_to_take = human_action if is_human_intervening else policy_action
        
        # 5. Step the wrapped environment with the chosen action.
        obs, reward, terminated, truncated, info = self.env.step(action_to_take)

        # 6. Annotate the `info` dictionary with override details.
        #    This is useful for logging and analysis.
        info["human_override"] = is_human_intervening
        if is_human_intervening:
            info["intervene_action"] = human_action
            info["policy_action_ignored"] = policy_action
        else:
            info["policy_action_used"] = policy_action
        
        return obs, reward, terminated, truncated, info


class DirectTeleopWrapper(gym.Wrapper):
    """
    A wrapper that allows direct human control of the environment.
    Unlike HumanInterventionWrapper, this provides complete human control
    without any autonomous policy.
    
    This is useful for:
    - Manual environment exploration.
    - Collecting human demonstrations.
    - Testing environment mechanics.
    """

    def __init__(self, env: gym.Env, teleop_interface):
        """
        Initializes the wrapper.

        Args:
            env (gym.Env): The Gymnasium environment to wrap.
            teleop_interface: An object with a `get_action()` method that returns
                              a NumPy array of the same shape as the env's action space.
        """
        super().__init__(env)
        
        if not hasattr(teleop_interface, "get_action"):
            raise TypeError("teleop_interface must have a 'get_action' method.")
        
        self.teleop = teleop_interface
        
        print("[DirectTeleopWrapper] Initialized for direct human control.")

    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        obs, info = self.env.reset(**kwargs)
        
        if hasattr(self.teleop, "reset"):
            self.teleop.reset()
            
        return obs, info

    def step(self, policy_action=None):
        """
        Executes a step in the environment using only human input.
        The policy_action parameter is ignored.

        Args:
            policy_action: Ignored. Kept for compatibility.

        Returns:
            The standard (obs, reward, terminated, truncated, info) tuple.
        """
        # Always use human action
        human_action = self.teleop.get_action()
        
        # Step the wrapped environment
        obs, reward, terminated, truncated, info = self.env.step(human_action)

        # Annotate info with action source
        info["human_action"] = human_action
        info["control_mode"] = "human"
        
        return obs, reward, terminated, truncated, info 