import re
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
    num_gripper: int = 0
    num_component_movement_xyz: int = 0
    num_component_yaw: int = 0
    num_component_gripper: int = 0
    num_component_mixed: int = 0
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
            teacher_num_gripper_interventions=int(self.num_gripper),
            teacher_num_component_movement_xyz_interventions=int(self.num_component_movement_xyz),
            teacher_num_component_yaw_interventions=int(self.num_component_yaw),
            teacher_num_component_gripper_interventions=int(self.num_component_gripper),
            teacher_num_component_mixed_interventions=int(self.num_component_mixed),
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
        tolerance_channel_weights: Optional[object] = None,  # optional per-action weights for l2 metric
        binary_gripper_actions: bool = False,
        binary_gripper_threshold: float = 0.0,
        hard_gripper_intervention: bool = False,
        gripper_intervene_pick_radius: float = 0.06,
        gripper_intervene_place_radius: float = 0.06,
        gripper_intervene_contact_threshold: float = 0.3,
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
        self._action_dim = int(np.prod(getattr(getattr(self.env, "action_space", None), "shape", (0,))))
        self.tolerance_channel_weights = self._parse_tolerance_channel_weights(
            tolerance_channel_weights,
            self._action_dim,
        )
        self.binary_gripper_actions = bool(binary_gripper_actions)
        self.binary_gripper_threshold = float(binary_gripper_threshold)
        self.hard_gripper_intervention = bool(hard_gripper_intervention)
        self.gripper_intervene_pick_radius = float(gripper_intervene_pick_radius)
        self.gripper_intervene_place_radius = float(gripper_intervene_place_radius)
        self.gripper_intervene_contact_threshold = float(gripper_intervene_contact_threshold)
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
        # Avoid flooding stdout when vectorized envs create many wrapper instances.
        cls = type(self)
        if not getattr(cls, "_init_logged_once", False):
            print(
                "[InterventionWrapper] Initialized."
                f" mode={self.mode}, teacher={self.teacher_type},"
                f" agent_mode={self.agent_mode}"
            )
            cls._init_logged_once = True
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
        self._oracle = None
        self._oracle_type: Optional[str] = None
        self._oracle_target_block: Optional[int] = None
        self._last_obs = None
        self._last_info: Optional[dict] = None

    def _parse_tolerance_channel_weights(self, raw: Optional[object], action_dim: int) -> Optional[np.ndarray]:
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if raw == "":
                return None
            parts = [p for p in re.split(r"[,\s;]+", raw) if p]
            values = np.asarray([float(p) for p in parts], dtype=np.float32)
        elif np.isscalar(raw):
            values = np.asarray([float(raw)], dtype=np.float32)
        else:
            values = np.asarray(raw, dtype=np.float32).reshape(-1)

        if np.any(~np.isfinite(values)):
            raise ValueError("tolerance_channel_weights must be finite")
        if np.any(values <= 0.0):
            raise ValueError("tolerance_channel_weights must be > 0")

        if action_dim <= 0:
            return values
        if values.size == 1:
            return np.full((action_dim,), float(values[0]), dtype=np.float32)
        if values.size != action_dim:
            raise ValueError(
                f"tolerance_channel_weights has {values.size} entries but action_dim is {action_dim}. "
                "Provide one value or exactly one per action channel."
            )
        return values.astype(np.float32)

    def _l2_delta(self, a: np.ndarray, b: np.ndarray, *, weighted: bool) -> float:
        delta = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
        if weighted and self.tolerance_channel_weights is not None and delta.shape[-1] == self.tolerance_channel_weights.shape[0]:
            delta = delta * self.tolerance_channel_weights
        return float(np.linalg.norm(delta))

    def _apply_gripper_binary(self, action: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if action is None:
            return None
        out = np.asarray(action, dtype=np.float32).copy()
        if not self.binary_gripper_actions or out.shape[-1] < 5:
            return out
        out[..., 4] = 1.0 if out[..., 4] >= self.binary_gripper_threshold else -1.0
        return out

    def _gripper_sign(self, action: Optional[np.ndarray]) -> int:
        if action is None:
            return 0
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < 5:
            return 0
        return 1 if float(arr[4]) >= self.binary_gripper_threshold else -1

    def _extract_scalar(self, value, default: float = 0.0) -> float:
        try:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return float(default)
            return float(arr[0])
        except Exception:
            return float(default)

    def _hard_gripper_violation(self, info: Optional[dict], policy_action: Optional[np.ndarray], teacher_action: Optional[np.ndarray]):
        if (
            not self.hard_gripper_intervention
            or self.teacher_type not in {"cube_plan", "cube_markov"}
            or info is None
            or policy_action is None
            or teacher_action is None
        ):
            return False, {}
        try:
            target_block = int(info.get("privileged/target_block", 0))
        except Exception:
            target_block = 0
        try:
            eff_pos = np.asarray(info.get("proprio/effector_pos"), dtype=np.float32).reshape(-1)
            block_pos = np.asarray(info.get(f"privileged/block_{target_block}_pos"), dtype=np.float32).reshape(-1)
            target_pos = np.asarray(info.get("privileged/target_block_pos"), dtype=np.float32).reshape(-1)
        except Exception:
            return False, {}
        if eff_pos.size < 3 or block_pos.size < 3 or target_pos.size < 3:
            return False, {}

        eff_block_dist = float(np.linalg.norm(eff_pos[:3] - block_pos[:3]))
        block_target_dist = float(np.linalg.norm(block_pos[:3] - target_pos[:3]))
        gripper_contact = self._extract_scalar(info.get("proprio/gripper_contact"), default=0.0)
        holding = gripper_contact >= self.gripper_intervene_contact_threshold
        near_pick = eff_block_dist <= self.gripper_intervene_pick_radius
        near_place = block_target_dist <= self.gripper_intervene_place_radius
        critical = bool(near_pick or near_place or holding)

        policy_sign = self._gripper_sign(policy_action)
        teacher_sign = self._gripper_sign(teacher_action)
        mismatch = policy_sign != teacher_sign and policy_sign != 0 and teacher_sign != 0
        violation = bool(critical and mismatch)
        diag = {
            "teacher_gripper_critical": float(1.0 if critical else 0.0),
            "teacher_gripper_mismatch": float(1.0 if mismatch else 0.0),
            "teacher_gripper_hard_violation": float(1.0 if violation else 0.0),
            "teacher_gripper_policy_sign": float(policy_sign),
            "teacher_gripper_teacher_sign": float(teacher_sign),
            "teacher_gripper_eff_block_dist": float(eff_block_dist),
            "teacher_gripper_block_target_dist": float(block_target_dist),
            "teacher_gripper_contact": float(gripper_contact),
        }
        return violation, diag

    def _classify_intervention_component(
        self,
        policy_action: Optional[np.ndarray],
        teacher_action: Optional[np.ndarray],
        reason: Optional[str],
    ):
        diag = {
            "teacher_delta_xyz_l2": 0.0,
            "teacher_delta_yaw_abs": 0.0,
            "teacher_delta_gripper_abs": 0.0,
            "teacher_reason_component": "none",
            "teacher_reason_component_is_movement_xyz": 0.0,
            "teacher_reason_component_is_yaw": 0.0,
            "teacher_reason_component_is_gripper": 0.0,
            "teacher_reason_component_is_mixed": 0.0,
        }
        if policy_action is None or teacher_action is None:
            return "none", diag
        p = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        t = np.asarray(teacher_action, dtype=np.float32).reshape(-1)
        if p.size == 0 or t.size == 0:
            return "none", diag
        n = min(p.size, t.size)
        d = np.abs(p[:n] - t[:n]).astype(np.float32)

        xyz_l2 = float(np.linalg.norm(d[:3])) if n >= 3 else 0.0
        yaw_abs = float(d[3]) if n >= 4 else 0.0
        gripper_abs = float(d[4]) if n >= 5 else 0.0
        eps = 1e-6
        has_xyz = xyz_l2 > eps
        has_yaw = yaw_abs > eps
        has_gripper = gripper_abs > eps

        # If hard gripper branch decided intervention, keep label stable.
        if reason == "gripper":
            component = "gripper"
        else:
            active_count = int(has_xyz) + int(has_yaw) + int(has_gripper)
            if active_count >= 2:
                component = "mixed"
            elif has_xyz:
                component = "movement_xyz"
            elif has_yaw:
                component = "yaw"
            elif has_gripper:
                component = "gripper"
            else:
                component = "none"

        diag["teacher_delta_xyz_l2"] = xyz_l2
        diag["teacher_delta_yaw_abs"] = yaw_abs
        diag["teacher_delta_gripper_abs"] = gripper_abs
        diag["teacher_reason_component"] = component
        diag["teacher_reason_component_is_movement_xyz"] = float(1.0 if component == "movement_xyz" else 0.0)
        diag["teacher_reason_component_is_yaw"] = float(1.0 if component == "yaw" else 0.0)
        diag["teacher_reason_component_is_gripper"] = float(1.0 if component == "gripper" else 0.0)
        diag["teacher_reason_component_is_mixed"] = float(1.0 if component == "mixed" else 0.0)
        return component, diag

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

    def _teacher_action(self, obs, info) -> Optional[np.ndarray]:
        if self.teacher_type == 'bfs':
            return self._bfs_teacher_action()
        if self.teacher_type in {"cube_plan", "cube_markov"} and self._oracle is not None:
            try:
                current_target_block = info.get('privileged/target_block') if isinstance(info, dict) else None
                try:
                    current_target_block = int(current_target_block)
                except Exception:
                    current_target_block = None

                oracle_done = bool(getattr(self._oracle, 'done', False))
                target_switched = (
                    current_target_block is not None
                    and self._oracle_target_block is not None
                    and current_target_block != self._oracle_target_block
                )
                if oracle_done or target_switched:
                    self._oracle.reset(obs, info or {})
                    if current_target_block is not None:
                        self._oracle_target_block = current_target_block

                return self._oracle.select_action(obs, info or {})
            except Exception:
                return None
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
        self._last_obs = obs
        self._last_info = info
        if self.teacher_type in {"cube_plan", "cube_markov"}:
            if self._oracle is None or self._oracle_type != self.teacher_type:
                if self.teacher_type == "cube_plan":
                    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
                    self._oracle = CubePlanOracle(self.unwrapped)
                else:
                    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
                    self._oracle = CubeMarkovOracle(self.unwrapped)
                self._oracle_type = self.teacher_type
            if self._oracle is not None:
                try:
                    self._oracle.reset(obs, info)
                    try:
                        self._oracle_target_block = int(info.get('privileged/target_block'))
                    except Exception:
                        self._oracle_target_block = None
                except Exception:
                    pass
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
        policy_action = self._apply_gripper_binary(policy_action)
        teacher_action = None
        teacher_candidate_action = None
        reason = None
        teacher_delta_l2 = 0.0
        teacher_delta_l2_raw = 0.0
        teacher_delta_angle_deg = 0.0
        teacher_candidate_available = False
        gripper_diag = {}
        component_reason = "none"
        component_diag = {}

        if self.mode == 'human':
            human = self._human_action()
            if human is not None:
                teacher_action = self._apply_gripper_binary(human)
                reason = 'human'
        else:  # agent mode
            if not self._episode_interventions_enabled:
                teacher_action = None
            else:
                # Respect warmup schedule
                if self._global_step_env < self.enable_after_steps:
                    teacher_action = None
                else:
                    teacher_action = self._teacher_action(self._last_obs, self._last_info or {})
                    teacher_action = self._apply_gripper_binary(teacher_action)
                teacher_candidate_action = teacher_action
                teacher_candidate_available = teacher_candidate_action is not None
                if teacher_candidate_available and policy_action is not None:
                    teacher_delta_l2_raw = self._l2_delta(policy_action, teacher_candidate_action, weighted=False)
                    teacher_delta_l2 = self._l2_delta(policy_action, teacher_candidate_action, weighted=True)
                    teacher_delta_angle_deg = float(self._angle_deg(policy_action, teacher_candidate_action))
                hard_gripper_violation, gripper_diag = self._hard_gripper_violation(
                    self._last_info or {},
                    policy_action,
                    teacher_action,
                )
                # Safety check (maze-only)
                safety_violation = False
                safety_margin = False
                if self.teacher_type == "bfs":
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
                            diverged = self._l2_delta(policy_action, teacher_action, weighted=True) > self.tolerance_value
                    if hard_gripper_violation:
                        reason = 'gripper'
                    elif safety_violation:
                        reason = 'safety'
                    elif diverged:
                        reason = 'divergence'
                    else:
                        teacher_action = None
                elif self.agent_mode == 'safety_align':
                    if hard_gripper_violation:
                        reason = 'gripper'
                    else:
                        aligned = False
                        if teacher_action is not None and policy_action is not None:
                            if self.tolerance_type == 'angle':
                                aligned = self._angle_deg(policy_action, teacher_action) <= self.tolerance_value
                            else:
                                aligned = self._l2_delta(policy_action, teacher_action, weighted=True) <= self.tolerance_value
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
                    if hard_gripper_violation:
                        reason = 'gripper'
                    else:
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
        self._last_obs = obs
        self._last_info = info

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
            component_reason, component_diag = self._classify_intervention_component(policy_action, teacher_action, reason)
            self._stats.intervention_steps += 1
            self._stats.start_burst()
            self._stats.step_burst()
            if reason == 'safety':
                self._stats.num_safety += 1
            elif reason == 'divergence' or reason == 'human':
                self._stats.num_divergence += 1
            elif reason == 'progress':
                self._stats.num_progress += 1
            elif reason == 'gripper':
                self._stats.num_gripper += 1
            if component_reason == 'movement_xyz':
                self._stats.num_component_movement_xyz += 1
            elif component_reason == 'yaw':
                self._stats.num_component_yaw += 1
            elif component_reason == 'gripper':
                self._stats.num_component_gripper += 1
            elif component_reason == 'mixed':
                self._stats.num_component_mixed += 1
            # Count new interventions when a burst starts at this step
            if self._stats._current_burst_len == 1:
                self._stats.num_interventions += 1
        else:
            self._stats.end_burst_if_needed()
        # Increment global step counter regardless of intervention
        self._global_step_env += 1

        # Annotate info
        info['teacher_intervened'] = bool(intervened)
        info['teacher_candidate_available'] = bool(teacher_candidate_available)
        info['teacher_delta_l2'] = float(teacher_delta_l2)
        info['teacher_delta_l2_raw'] = float(teacher_delta_l2_raw)
        info['teacher_delta_angle_deg'] = float(teacher_delta_angle_deg)
        info['teacher_tolerance_value'] = float(self.tolerance_value)
        if self.tolerance_channel_weights is not None:
            info['teacher_tolerance_channel_weights'] = np.array(self.tolerance_channel_weights, dtype=np.float32)
        if gripper_diag:
            info.update(gripper_diag)
        if component_diag:
            info.update(component_diag)
        if intervened:
            info['teacher_reason'] = reason
            info['teacher_reason_component'] = component_reason
            info['teacher_action'] = np.array(teacher_action, dtype=np.float32)
            info['student_action'] = np.array(policy_action, dtype=np.float32)
        else:
            info['teacher_reason'] = None
            info['teacher_reason_component'] = 'none'
            if teacher_candidate_available and teacher_candidate_action is not None:
                info['teacher_action'] = np.array(teacher_candidate_action, dtype=np.float32)
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
        self._last_obs = obs
        self._last_info = info

        # Annotate info with action source
        info["human_action"] = human_action
        info["control_mode"] = "human"
        
        return obs, reward, terminated, truncated, info 
