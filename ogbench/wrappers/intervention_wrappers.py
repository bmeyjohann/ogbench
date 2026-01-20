import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym


@dataclass
class _InterventionStats:
    episode_steps: int = 0
    intervention_steps: int = 0
    num_interventions: int = 0
    num_safety: int = 0
    num_divergence: int = 0
    num_progress: int = 0
    episode_gate_enabled: int = 1
    episode_gate_prob: float = 1.0
    _in_burst: bool = False
    _current_burst_len: int = 0
    _burst_lens_sum: int = 0
    _burst_count: int = 0

    def start_burst(self):
        if not self._in_burst:
            self._in_burst = True
            self._current_burst_len = 0
            self._burst_count += 1

    def step_burst(self):
        if self._in_burst:
            self._current_burst_len += 1

    def end_burst_if_needed(self):
        if self._in_burst:
            self._in_burst = False
            self._burst_lens_sum += self._current_burst_len
            self._current_burst_len = 0

    def flush_metrics(self) -> dict:
        # Ensure an open burst is closed at episode end
        self.end_burst_if_needed()
        frac = (self.intervention_steps / max(1, self.episode_steps)) if self.episode_steps > 0 else 0.0
        avg_burst = (self._burst_lens_sum / max(1, self._burst_count)) if self._burst_count > 0 else 0.0
        return dict(
            teacher_num_interventions=int(self.num_interventions),
            teacher_intervention_steps=int(self.intervention_steps),
            teacher_fraction_steps=float(frac),
            teacher_avg_burst_len=float(avg_burst),
            teacher_num_safety_interventions=int(self.num_safety),
            teacher_num_divergence_interventions=int(self.num_divergence),
            teacher_num_progress_interventions=int(self.num_progress),
            teacher_episode_steps=int(self.episode_steps),
            teacher_episode_gate_enabled=int(self.episode_gate_enabled),
            teacher_episode_gate_prob=float(self.episode_gate_prob),
        )


class InterventionWrapper(gym.Wrapper):
    """
    Generic intervention wrapper supporting human teleop and agent (teacher) modes.

    Modes:
    - mode='human': override with teleop input above threshold for a hold duration
    - mode='agent': override based on teacher policy (e.g., BFS) using safety and tolerance rules
    """

    def __init__(
        self,
        env: gym.Env,
        teleop_interface=None,
        *,
        mode: str = 'human',  # 'human' or 'agent'
        teacher_type: str = 'bfs',
        # Human params
        threshold: float = 0.1,
        hold_time: float = 0.5,
        # Agent/teacher params
        tolerance_type: str = 'angle',  # 'angle' or 'l2'
        tolerance_value: float = 30.0,  # degrees for angle, absolute for l2
        hard_block_lethal: bool = True,  # intervene if student's step enters lethal/danger cell
        enable_after_steps: int = 0,     # warmup steps per env before enabling interventions
        agent_mode: str = 'divergence',  # 'divergence', 'safety_align', 'safety_progress'
        safety_margin_frac: float = 0.0,  # fraction of maze cell size for safety margin
        release_steps: int = 3,  # consecutive steps to release intervention
        episode_intervention_prob: float = 1.0,
        episode_intervention_prob_min: float = 0.0,
        episode_intervention_prob_decay_steps: int = 0,
        episode_intervention_prob_decay_start: int = 0,
        episode_intervention_seed: Optional[int] = None,
    ):
        super().__init__(env)

        assert mode in ('human', 'agent')
        assert tolerance_type in ('angle', 'l2')
        assert agent_mode in ('divergence', 'safety_align', 'safety_progress')
        self.mode = mode
        self.teacher_type = teacher_type
        self.teleop = teleop_interface
        self.threshold = float(threshold)
        self.hold_time = float(hold_time)
        self.tolerance_type = tolerance_type
        self.tolerance_value = float(tolerance_value)
        self.hard_block_lethal = bool(hard_block_lethal)
        self.enable_after_steps = int(enable_after_steps)
        self.agent_mode = agent_mode
        self.safety_margin_frac = float(safety_margin_frac)
        self.release_steps = int(release_steps)
        self.episode_intervention_prob = float(episode_intervention_prob)
        self.episode_intervention_prob_min = float(episode_intervention_prob_min)
        self.episode_intervention_prob_decay_steps = int(episode_intervention_prob_decay_steps)
        self.episode_intervention_prob_decay_start = int(episode_intervention_prob_decay_start)
        self._rng = np.random.default_rng(episode_intervention_seed)
        

        # Internal timers (human)
        self._last_override_ts = 0.0

        # Episode stats
        self._stats = _InterventionStats()

        # Teacher cache (for future types); BFS uses env oracle per-step
        print(
            "[InterventionWrapper] Initialized."
            f" mode={self.mode}, teacher={self.teacher_type},"
            f" agent_mode={self.agent_mode}"
        )
        # Global per-env step counter across episodes
        self._global_step_env = 0
        self._danger_centers: Optional[np.ndarray] = None
        self._maze_unit: Optional[float] = None
        self._align_count = 0
        self._progress_good_count = 0
        self._progress_violation = False
        self._last_distance: Optional[float] = None
        self._safety_align_active = False
        self._progress_active = False
        self._episode_interventions_enabled = True

    def _current_episode_prob(self) -> float:
        if self.episode_intervention_prob_decay_steps <= 0:
            return self.episode_intervention_prob
        progress = max(0, self._global_step_env - self.episode_intervention_prob_decay_start)
        frac = min(1.0, progress / float(self.episode_intervention_prob_decay_steps))
        return self.episode_intervention_prob + (self.episode_intervention_prob_min - self.episode_intervention_prob) * frac

    # ---------------
    # Human utilities
    # ---------------
    def _human_action(self) -> Optional[np.ndarray]:
        if self.teleop is None or not hasattr(self.teleop, 'get_action'):
            return None
        human_action = self.teleop.get_action()
        if human_action is None:
            return None
        if np.linalg.norm(human_action) > self.threshold:
            self._last_override_ts = time.perf_counter()
        is_active = (time.perf_counter() - self._last_override_ts) < self.hold_time
        return human_action if is_active else None

    # ----------------
    # Agent utilities
    # ----------------
    def _bfs_teacher_action(self) -> Optional[np.ndarray]:
        """Use env oracle subgoal to compute a direction action towards the next waypoint."""
        try:
            agent_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
            # Goal from env state; fallback to observation if needed
            goal_xy = np.array(getattr(self.unwrapped, 'cur_goal_xy', None), dtype=np.float32)
            if goal_xy is None or goal_xy.shape != (2,):
                # Fallback: best-effort from observation
                obs = getattr(self.unwrapped, 'get_ob', lambda: None)()
                if isinstance(obs, np.ndarray) and obs.shape[0] >= 4:
                    goal_xy = obs[2:4].astype(np.float32)
                else:
                    return None

            # Query env for oracle subgoal (BFS one-step waypoint)
            subgoal_out = self.unwrapped.get_oracle_subgoal(agent_xy, goal_xy)
            if isinstance(subgoal_out, (list, tuple)):
                subgoal_xy = np.array(subgoal_out[0], dtype=np.float32)
            else:
                subgoal_xy = np.array(subgoal_out, dtype=np.float32)
            if subgoal_xy is None or subgoal_xy.shape != (2,):
                return None

            direction = subgoal_xy - agent_xy
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                return np.zeros_like(direction)
            unit = direction / norm
            # Scale to action space range
            if hasattr(self.env.action_space, 'high'):
                max_mag = float(np.min(self.env.action_space.high))
                max_mag = 1.0 if not np.isfinite(max_mag) or max_mag <= 0 else max_mag
            else:
                max_mag = 1.0
            return unit * max_mag
        except Exception:
            return None

    def _teacher_action(self) -> Optional[np.ndarray]:
        if self.teacher_type == 'bfs':
            return self._bfs_teacher_action()
        return None

    def _angle_deg(self, a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an < 1e-8 or bn < 1e-8:
            return 180.0
        cos = float(np.clip(np.dot(a, b) / (an * bn), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _predict_next_xy(self, action: np.ndarray) -> Optional[Tuple[int, int]]:
        """Predict next grid cell if we applied this action (PointEnv dynamics)."""
        try:
            cur_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
            # PointEnv applies action scaled by 0.2 per step
            next_xy = cur_xy + 0.2 * action
            i, j = self.unwrapped.xy_to_ij(next_xy)
            return i, j
        except Exception:
            return None

    def _is_traversable_cell(self, ij: Tuple[int, int]) -> bool:
        """Delegate to env.is_traversable(i, j) when available; default True if unknown."""
        try:
            i, j = ij
            return bool(self.unwrapped.is_traversable(i, j))
        except Exception:
            # Fallback: be permissive if API missing
            return True

    def _danger_centers_xy(self) -> Optional[np.ndarray]:
        """Cache and return dangerous tile centers as an (N, 2) array."""
        if self._danger_centers is not None:
            return self._danger_centers
        base = self.unwrapped
        maze_map = getattr(base, 'maze_map', None)
        dangerous_id = getattr(base, '_dangerous_tile_id', None)
        ij_to_xy = getattr(base, 'ij_to_xy', None)
        maze_unit = getattr(base, '_maze_unit', None)
        if maze_map is None or dangerous_id is None or ij_to_xy is None:
            return None
        centers = []
        for i in range(maze_map.shape[0]):
            for j in range(maze_map.shape[1]):
                if maze_map[i, j] == dangerous_id:
                    centers.append(ij_to_xy((i, j)))
        if not centers:
            self._danger_centers = None
            return None
        self._danger_centers = np.asarray(centers, dtype=np.float32)
        self._maze_unit = float(maze_unit) if maze_unit is not None else 1.0
        return self._danger_centers

    def _near_danger_margin(self) -> bool:
        if self.safety_margin_frac <= 0:
            return False
        centers = self._danger_centers_xy()
        if centers is None:
            return False
        try:
            agent_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
        except Exception:
            return False
        maze_unit = self._maze_unit if self._maze_unit is not None else 1.0
        margin = self.safety_margin_frac * maze_unit
        dists = np.linalg.norm(centers - agent_xy[None, :], axis=1)
        min_dist = float(np.min(dists))
        dist_to_boundary = max(0.0, min_dist - 0.5 * maze_unit)
        return dist_to_boundary <= margin

    def _goal_distance(self) -> Optional[float]:
        try:
            agent_xy = np.array(self.unwrapped.get_xy(), dtype=np.float32)
            goal_xy = np.array(getattr(self.unwrapped, 'cur_goal_xy', None), dtype=np.float32)
        except Exception:
            return None
        if goal_xy is None or goal_xy.shape != (2,):
            return None
        return float(np.linalg.norm(goal_xy - agent_xy))

    # --------------
    # Gym overrides
    # --------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # reset stats
        self._stats = _InterventionStats()
        # reset human timer
        self._last_override_ts = 0.0
        self._align_count = 0
        self._progress_good_count = 0
        self._progress_violation = False
        self._safety_align_active = False
        self._progress_active = False
        self._last_distance = self._goal_distance()
        prob = float(np.clip(self._current_episode_prob(), 0.0, 1.0))
        self._episode_interventions_enabled = bool(self._rng.random() < prob)
        self._stats.episode_gate_enabled = 1 if self._episode_interventions_enabled else 0
        self._stats.episode_gate_prob = prob
        return obs, info

    def step(self, policy_action: np.ndarray):
        # Decide override
        teacher_action = None
        reason = None

        if self.mode == 'human':
            human = self._human_action()
            if human is not None:
                teacher_action = human
                reason = 'human'
        else:  # agent mode
            if not self._episode_interventions_enabled:
                teacher_action = None
            else:
                # Respect warmup schedule
                if self._global_step_env < self.enable_after_steps:
                    teacher_action = None
                else:
                    teacher_action = self._teacher_action()
                # Safety check
                safety_violation = False
                if self.hard_block_lethal and teacher_action is not None and policy_action is not None:
                    predicted = self._predict_next_xy(policy_action)
                    if predicted is not None and (not self._is_traversable_cell(predicted)):
                        safety_violation = True
                safety_margin = self._near_danger_margin()

                if self.agent_mode == 'divergence':
                    diverged = False
                    if teacher_action is not None and policy_action is not None and not safety_violation:
                        if self.tolerance_type == 'angle':
                            ang = self._angle_deg(policy_action, teacher_action)
                            diverged = ang > self.tolerance_value
                        else:
                            diverged = float(np.linalg.norm(policy_action - teacher_action)) > self.tolerance_value
                    if safety_violation:
                        reason = 'safety'
                    elif diverged:
                        reason = 'divergence'
                    else:
                        teacher_action = None
                elif self.agent_mode == 'safety_align':
                    aligned = False
                    if teacher_action is not None and policy_action is not None:
                        if self.tolerance_type == 'angle':
                            aligned = self._angle_deg(policy_action, teacher_action) <= self.tolerance_value
                        else:
                            aligned = float(np.linalg.norm(policy_action - teacher_action)) <= self.tolerance_value
                    if aligned:
                        self._align_count += 1
                    else:
                        self._align_count = 0

                    if safety_violation or safety_margin:
                        self._safety_align_active = True

                    if self._safety_align_active:
                        if (not safety_violation and not safety_margin and self._align_count >= self.release_steps):
                            self._safety_align_active = False
                        else:
                            reason = 'safety'
                    if not self._safety_align_active:
                        teacher_action = None
                else:  # safety_progress
                    if safety_violation or safety_margin or self._progress_violation:
                        self._progress_active = True

                    if self._progress_active:
                        if (not safety_violation and not safety_margin and self._progress_good_count >= self.release_steps):
                            self._progress_active = False
                        else:
                            reason = 'safety' if (safety_violation or safety_margin) else 'progress'
                    if not self._progress_active:
                        teacher_action = None

        # Apply action
        intervened = teacher_action is not None
        action_to_take = teacher_action if intervened else policy_action
        obs, reward, terminated, truncated, info = self.env.step(action_to_take)

        # Update progress tracking based on actual next state
        new_distance = self._goal_distance()
        if new_distance is not None and self._last_distance is not None:
            delta = new_distance - self._last_distance
            self._progress_violation = delta > 0.0
            if self._progress_active:
                if delta < 0.0:
                    self._progress_good_count += 1
                else:
                    self._progress_good_count = 0
            else:
                self._progress_good_count = 0
        self._last_distance = new_distance

        # Stats update
        self._stats.episode_steps += 1
        if intervened:
            self._stats.intervention_steps += 1
            self._stats.start_burst()
            self._stats.step_burst()
            if reason == 'safety':
                self._stats.num_safety += 1
            elif reason == 'divergence' or reason == 'human':
                self._stats.num_divergence += 1
            elif reason == 'progress':
                self._stats.num_progress += 1
            # Count new interventions when a burst starts at this step
            if self._stats._current_burst_len == 1:
                self._stats.num_interventions += 1
        else:
            self._stats.end_burst_if_needed()
        # Increment global step counter regardless of intervention
        self._global_step_env += 1

        # Annotate info
        info['teacher_intervened'] = bool(intervened)
        if intervened:
            info['teacher_reason'] = reason
            info['teacher_action'] = np.array(teacher_action, dtype=np.float32)
            info['student_action'] = np.array(policy_action, dtype=np.float32)
        else:
            info['teacher_reason'] = None
            info['student_action'] = np.array(policy_action, dtype=np.float32)

        # On episode end, flush metrics for logging
        if terminated or truncated:
            ep_metrics = self._stats.flush_metrics()
            info.update(ep_metrics)

        return obs, reward, terminated, truncated, info


# Backwards-compatible alias for existing imports/usages
HumanInterventionWrapper = InterventionWrapper


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
