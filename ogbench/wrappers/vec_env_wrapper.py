"""
Vectorized Environment Wrapper for OGBench environments.

This wrapper follows Isaac Lab's RSL-RL interface exactly to ensure full
compatibility with RSL-RL's OnPolicyRunner and config system.
"""

import torch
import numpy as np
import gymnasium as gym
import warnings
from typing import Dict, Any, List, Callable, Optional
try:
    from rsl_rl.env import VecEnv
except Exception:
    class VecEnv:  # type: ignore[no-redef]
        """Minimal fallback base when rsl_rl is not installed.

        The FastSAC/OGBench paths only rely on the concrete methods implemented
        below, while PPO/RSL-RL users will still get the real base class when
        the dependency is available.
        """

        def __init__(self, *args, **kwargs):
            super().__init__()

from tensordict import TensorDict
try:
    import mujoco
except Exception:
    mujoco = None


class VectorizedOGBenchEnv(VecEnv):
    """
    RSL-RL VecEnv implementation for multiple OGBench environments.
    
    Compatible with RSL-RL's OnPolicyRunner and follows Isaac Lab's interface.
    Manages multiple environments internally and provides vectorized
    step/reset functionality.
    """

    _PIXEL_CAMERA_KEYS = (
        'pixel_camera_mode',
        'pixel_local_view_size',
        'pixel_local_camera_height',
        'pixel_first_person_distance',
        'pixel_first_person_height',
        'pixel_first_person_lookahead',
        'pixel_first_person_pitch',
    )
    
    def __init__(self, env_name: str, num_envs: int = 1, wrappers: List[Callable] = None, 
                 clip_actions: float | None = None, auto_reset_on_init: bool = True, **env_kwargs):
        """
        Args:
            env_name: OGBench environment name (e.g., 'pointmaze-arena-v0')
            num_envs: Number of parallel environments
            wrappers: List of wrapper functions to apply to each environment
            clip_actions: The clipping value for actions. If None, then no clipping is done.
            **env_kwargs: Additional arguments passed to gym.make()
        """
        # Initialize base class
        super().__init__()
        
        self.env_name = env_name
        self.wrappers = wrappers or []
        self.env_kwargs = dict(env_kwargs)
        self.clip_actions = clip_actions
        self._render_mode = str(self.env_kwargs.get("render_mode", "") or "").lower()
        self._visualize_intervention_colors = bool(self.env_kwargs.pop("visualize_intervention_colors", True))
        self._passive_viewer_enabled = False
        self._passive_viewer_info: Optional[dict[str, Any]] = None
        self._viewer_cached_model_ptr: Optional[int] = None
        self._viewer_arm_material_ids: list[int] = []
        self._viewer_gripper_material_ids: list[int] = []
        self._viewer_original_rgba: dict[int, np.ndarray] = {}
        self._viewer_last_visual_state: Optional[str] = None
        self._viewer_visuals_disabled = False
        self._viewer_visuals_warned = False
        self._viewer_sync_warned = False
        
        # Create individual environments
        self.envs = []
        for i in range(num_envs):
            env = self._make_env(env_name)
            # Apply wrappers
            for wrapper_fn in self.wrappers:
                env = wrapper_fn(env)
            self.envs.append(env)
        
        # Get environment properties from first environment
        sample_env = self.envs[0]
        
        # RSL-RL VecEnv required attributes (matching Isaac Lab)
        self.num_envs = num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get observation and action dimensions
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        self.num_actions = gym.spaces.flatdim(sample_env.action_space)
        self.num_obs = gym.spaces.flatdim(sample_env.observation_space)
        
        # Privileged observations (for asymmetric actor-critic)
        # OGBench environments don't have privileged observations by default
        self.num_privileged_obs = 0
        
        # Get max episode length from environment or default
        if hasattr(sample_env, '_max_episode_steps'):
            self.max_episode_length = sample_env._max_episode_steps
        elif hasattr(sample_env, 'spec') and sample_env.spec and sample_env.spec.max_episode_steps:
            self.max_episode_length = sample_env.spec.max_episode_steps
        else:
            self.max_episode_length = 500  # Default fallback
        
        # Episode tracking buffer (managed by RSL-RL)
        self._episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state for current observations
        self._current_obs = None
        self._current_obs_dict = None
        
        # Modify action space for clipping if specified
        self._modify_action_space()
        
        # Add cfg attribute for RSL-RL logging compatibility
        # Create a simple config object that contains basic environment info
        from dataclasses import dataclass
        
        @dataclass
        class EnvConfig:
            env_name: str
            num_envs: int
            max_episode_length: int
            is_finite_horizon: bool = True  # OGBench environments have finite episodes
        
        self.cfg = EnvConfig(env_name, num_envs, self.max_episode_length)
        
        # Optionally reset all environments to initialize
        if auto_reset_on_init:
            self.reset()

    def _maybe_launch_passive_viewer(self):
        if self._passive_viewer_enabled:
            return
        if self._render_mode != "human":
            return
        if not self.envs:
            return
        base = self.envs[0].unwrapped
        launch_fn = getattr(base, "launch_passive_viewer", None)
        if not callable(launch_fn):
            return
        try:
            launch_fn(show_left_ui=False, show_right_ui=False)
            self._passive_viewer_enabled = True
        except Exception:
            self._passive_viewer_enabled = False

    def _sync_passive_viewer(self):
        if not self._passive_viewer_enabled:
            return
        if not self.envs:
            return
        base = self.envs[0].unwrapped
        sync_fn = getattr(base, "sync_passive_viewer", None)
        if not callable(sync_fn):
            return
        try:
            if self._visualize_intervention_colors and not self._viewer_visuals_disabled:
                try:
                    self._apply_intervention_visuals()
                except Exception as exc:
                    self._viewer_visuals_disabled = True
                    if not self._viewer_visuals_warned:
                        print(
                            f"[VecEnvViewer] intervention-color visuals failed; "
                            f"disabling viewer tinting for this run: {exc}",
                            flush=True,
                        )
                        self._viewer_visuals_warned = True
                    try:
                        self._restore_original_materials()
                    except Exception:
                        pass
            else:
                self._restore_original_materials()
            sync_fn()
        except Exception as exc:
            if not self._viewer_sync_warned:
                print(f"[VecEnvViewer] passive viewer sync failed; disabling live viewer sync: {exc}", flush=True)
                self._viewer_sync_warned = True
            self._passive_viewer_enabled = False

    def _close_passive_viewer(self):
        if not self.envs:
            self._passive_viewer_enabled = False
            return
        base = self.envs[0].unwrapped
        close_fn = getattr(base, "close_passive_viewer", None)
        if callable(close_fn):
            try:
                self._restore_original_materials()
                close_fn()
            except Exception:
                pass
        self._passive_viewer_enabled = False
        self._viewer_last_visual_state = None
        self._viewer_visuals_disabled = False
        self._viewer_visuals_warned = False
        self._viewer_sync_warned = False
    
    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self._episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.
        
        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self._episode_length_buf = value
    
    def get_observations(self) -> TensorDict:
        """Returns the current observations of the environment.
        
        Returns:
            TensorDict: Observation TensorDict with policy key for RSL-RL compatibility
        """
        if self._current_obs is None:
            self.reset()
        # Return TensorDict with policy observations that RSL-RL expects
        return TensorDict(
            {"policy": self._current_obs},
            batch_size=[self.num_envs],
        )
    
    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments and return initial observations."""
        obs_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
        
        # Stack observations and convert to tensor
        obs_array = np.stack(obs_list, axis=0)
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        
        # Store current observations
        self._current_obs = obs_tensor
        self._current_obs_dict = {
            "policy": obs_tensor,
            # Note: Add "critic" key here if privileged observations are needed
        }
        
        # Reset episode length buffer
        self._episode_length_buf.zero_()
        self._passive_viewer_info = None
        self._maybe_launch_passive_viewer()
        
        # Return TensorDict format for RSL-RL compatibility
        obs_tensordict = TensorDict(
            {"policy": obs_tensor},
            batch_size=[self.num_envs],
        )
        return obs_tensordict, {"observations": self._current_obs_dict}
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step all environments with the given actions.
        
        Args:
            actions: Tensor of actions for all environments
            
        Returns:
            tuple: (observations, rewards, dones, extras)
        """
        # Clip actions to max L2 norm if specified (avoid diagonal speedup)
        if self.clip_actions is not None:
            max_norm = float(self.clip_actions)
            norms = torch.linalg.norm(actions, dim=-1, keepdim=True)
            scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)
            actions = actions * scale
        
        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
        
        # Step each environment
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            done = bool(terminated or truncated)
            # Collect info before any reset
            if isinstance(info, dict):
                info = dict(info)
                info['terminated'] = bool(terminated)
                info['truncated'] = bool(truncated)
            infos_list.append(info)
            rewards_list.append(reward)
            dones_list.append(done)
            # Auto-reset done envs to provide the next episode observation
            if done:
                obs_reset, _ = env.reset()
                obs_list.append(obs_reset)
            else:
                obs_list.append(obs)

        if infos_list and isinstance(infos_list[0], dict):
            self._passive_viewer_info = dict(infos_list[0])
        else:
            self._passive_viewer_info = None
        
        # Convert to arrays
        obs_array = np.stack(obs_list, axis=0)
        rewards_array = np.array(rewards_list)
        dones_array = np.array(dones_list)
        
        # Convert to tensors
        obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
        rewards_tensor = torch.from_numpy(rewards_array).float().to(self.device)
        dones_tensor = torch.from_numpy(dones_array).to(dtype=torch.long, device=self.device)
        
        # Store current observations
        self._current_obs = obs_tensor
        self._current_obs_dict = {
            "policy": obs_tensor,
            # Note: Add "critic" key here if privileged observations are needed
        }
        
        # Update episode length tracking - increment BEFORE collecting metrics
        self._episode_length_buf += 1
        
        # Collect episode rewards and detailed metrics from completed episodes BEFORE resetting lengths
        episode_rewards = []
        episode_lengths = []
        sparse_rewards = []
        dense_rewards = []
        goals_reached = []
        distances = []
        killed = []
        subgoal_avgs = []
        subgoal_returns = []
        subgoal_transitions = []
        cube_solved_counts = []
        cube_total_counts = []
        cube_solved_fracs = []
        cube_max_errors = []
        
        # First pass: collect episode completion data before any resets
        for i, (done, info) in enumerate(zip(dones_array, infos_list)):
            if done and isinstance(info, dict):
                # Record episode length BEFORE resetting (this is the current episode length)
                current_episode_length = self._episode_length_buf[i].item()
                episode_lengths.append(current_episode_length)
                
                # Use cumulative episode reward from DetailedRewardWrapper
                total_episode_reward = info.get('episode_sparse_reward', 0.0) + info.get('episode_dense_reward', 0.0)
                episode_rewards.append(total_episode_reward)
        
        # Reset episode lengths for completed episodes AFTER recording them
        self._episode_length_buf[dones_tensor.bool()] = 0
        
        # Collect detailed metrics for completed episodes
        timeouts = []
        for i, (done, info) in enumerate(zip(dones_array, infos_list)):
            if done and isinstance(info, dict):
                # Detailed reward wrapper metrics (if available)
                if 'episode_sparse_reward' in info:
                    sparse_rewards.append(info['episode_sparse_reward'])
                if 'episode_dense_reward' in info:
                    dense_rewards.append(info['episode_dense_reward'])
                if 'goal_reached' in info:
                    goals_reached.append(info['goal_reached'])
                if 'distance_to_goal' in info:
                    distances.append(info['distance_to_goal'])
                killed.append(info.get('killed', 0.0))
                if 'truncated' in info:
                    timeouts.append(1.0 if bool(info['truncated']) else 0.0)
                else:
                    timeouts.append(0.0)
                if 'episode_avg_distance_to_subgoal' in info:
                    subgoal_avgs.append(info['episode_avg_distance_to_subgoal'])
                if 'episode_subgoal_shaping_return' in info:
                    subgoal_returns.append(info['episode_subgoal_shaping_return'])
                if 'episode_subgoal_transitions' in info:
                    subgoal_transitions.append(info['episode_subgoal_transitions'])
                if 'diag/cubes_solved' in info:
                    cube_solved_counts.append(float(info['diag/cubes_solved']))
                if 'diag/cubes_total' in info:
                    cube_total_counts.append(float(info['diag/cubes_total']))
                if 'diag/cubes_solved_fraction' in info:
                    cube_solved_fracs.append(float(info['diag/cubes_solved_fraction']))
                elif 'diag/cubes_solved' in info and 'diag/cubes_total' in info:
                    total = float(info['diag/cubes_total'])
                    if total > 0.0:
                        cube_solved_fracs.append(float(info['diag/cubes_solved']) / total)
                if 'diag/cube_max_target_error' in info:
                    cube_max_errors.append(float(info['diag/cube_max_target_error']))

        # Aggregate teacher intervention metrics if provided by InterventionWrapper
        teacher_metrics = {
            'teacher_num_interventions': [],
            'teacher_intervention_steps': [],
            'teacher_fraction_steps': [],
            'teacher_avg_burst_len': [],
            'teacher_num_safety_interventions': [],
            'teacher_num_divergence_interventions': [],
            'teacher_num_progress_interventions': [],
            'teacher_episode_steps': [],
            'teacher_episode_gate_enabled': [],
            'teacher_episode_gate_prob': [],
        }
        for done, info in zip(dones_array, infos_list):
            if done and isinstance(info, dict):
                for k in list(teacher_metrics.keys()):
                    if k in info:
                        teacher_metrics[k].append(info[k])
        
        # Create extras dict following Isaac Lab's format
        extras = {"observations": self._current_obs_dict}

        # Compute applied/student/teacher actions per env.
        # Prefer the exact executed action forwarded by the inner intervention wrapper.
        # Fall back to reconstruction only if older wrappers do not provide it.
        applied_actions = []
        student_actions = []
        teacher_actions = []
        teacher_intervened_mask = []
        teacher_candidate_available = []
        teacher_delta_l2 = []
        teacher_delta_angle_deg = []
        teacher_delta_xyz_l2 = []
        teacher_delta_yaw_abs = []
        teacher_delta_gripper_abs = []
        teacher_component_threshold_xyz = []
        teacher_component_threshold_yaw = []
        teacher_component_threshold_gripper = []
        teacher_component_threshold_scale = []
        teacher_component_diverged_any = []
        teacher_tolerance_value = []
        teacher_target_block = []
        teacher_cubes_solved = []
        teacher_cube_max_target_error = []
        for i, info in enumerate(infos_list):
            info_dict = info if isinstance(info, dict) else {}
            teacher_candidate_available.append(1.0 if bool(info_dict.get('teacher_candidate_available', False)) else 0.0)
            teacher_delta_l2.append(float(info_dict.get('teacher_delta_l2', 0.0)))
            teacher_delta_angle_deg.append(float(info_dict.get('teacher_delta_angle_deg', 0.0)))
            teacher_delta_xyz_l2.append(float(info_dict.get('teacher_delta_xyz_l2', 0.0)))
            teacher_delta_yaw_abs.append(float(info_dict.get('teacher_delta_yaw_abs', 0.0)))
            teacher_delta_gripper_abs.append(float(info_dict.get('teacher_delta_gripper_abs', 0.0)))
            teacher_component_threshold_xyz.append(float(info_dict.get('teacher_component_threshold_xyz', 0.0)))
            teacher_component_threshold_yaw.append(float(info_dict.get('teacher_component_threshold_yaw', 0.0)))
            teacher_component_threshold_gripper.append(float(info_dict.get('teacher_component_threshold_gripper', 0.0)))
            teacher_component_threshold_scale.append(float(info_dict.get('teacher_component_threshold_scale', 1.0)))
            teacher_component_diverged_any.append(float(info_dict.get('teacher_component_diverged_any', 0.0)))
            teacher_tolerance_value.append(float(info_dict.get('teacher_tolerance_value', 0.0)))
            teacher_target_block.append(float(info_dict.get('diag/target_block_dynamic', info_dict.get('privileged/target_block', 0.0))))
            teacher_cubes_solved.append(float(info_dict.get('diag/cubes_solved', 0.0)))
            teacher_cube_max_target_error.append(float(info_dict.get('diag/cube_max_target_error', 0.0)))

            student_action = info_dict.get('student_action')
            if student_action is None:
                student_action = actions_np[i]
            student_action_arr = np.asarray(student_action, dtype=np.float32)
            student_actions.append(student_action_arr)

            teacher_action = info_dict.get('teacher_action')
            if teacher_action is None:
                teacher_action = student_action
            teacher_action_arr = np.asarray(teacher_action, dtype=np.float32)
            teacher_actions.append(teacher_action_arr)

            applied_action = info_dict.get('applied_action')
            if applied_action is None:
                applied_action_arr = teacher_action_arr if bool(info_dict.get('teacher_intervened', False)) else student_action_arr
            else:
                applied_action_arr = np.asarray(applied_action, dtype=np.float32)
            applied_actions.append(applied_action_arr)

            teacher_intervened_raw = bool(info_dict.get('teacher_intervened', False))
            try:
                action_override = bool(np.abs(applied_action_arr - student_action_arr).sum() > 1e-6)
            except Exception:
                action_override = teacher_intervened_raw
            teacher_intervened = bool(teacher_intervened_raw or action_override)
            teacher_intervened_mask.append(teacher_intervened)

        if applied_actions:
            extras['applied_actions'] = torch.tensor(np.stack(applied_actions, axis=0), device=self.device, dtype=torch.float32)
            extras['student_actions'] = torch.tensor(np.stack(student_actions, axis=0), device=self.device, dtype=torch.float32)
            extras['teacher_actions'] = torch.tensor(np.stack(teacher_actions, axis=0), device=self.device, dtype=torch.float32)
            extras['teacher_intervened_mask'] = torch.tensor(teacher_intervened_mask, device=self.device, dtype=torch.bool)
            if not extras.get('log'):
                extras['log'] = {}
            candidate_arr = np.asarray(teacher_candidate_available, dtype=np.float32)
            extras['log']['/Teacher/diag_candidate_available'] = torch.tensor(candidate_arr, device=self.device, dtype=torch.float32)
            extras['log']['/Teacher/diag_intervened_step'] = torch.tensor(
                np.asarray(teacher_intervened_mask, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_delta_l2'] = torch.tensor(
                np.asarray(teacher_delta_l2, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_delta_angle_deg'] = torch.tensor(
                np.asarray(teacher_delta_angle_deg, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_delta_xyz_l2'] = torch.tensor(
                np.asarray(teacher_delta_xyz_l2, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_delta_yaw_abs'] = torch.tensor(
                np.asarray(teacher_delta_yaw_abs, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_delta_gripper_abs'] = torch.tensor(
                np.asarray(teacher_delta_gripper_abs, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_component_threshold_xyz'] = torch.tensor(
                np.asarray(teacher_component_threshold_xyz, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_component_threshold_yaw'] = torch.tensor(
                np.asarray(teacher_component_threshold_yaw, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_component_threshold_gripper'] = torch.tensor(
                np.asarray(teacher_component_threshold_gripper, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_component_threshold_scale'] = torch.tensor(
                np.asarray(teacher_component_threshold_scale, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_component_diverged_any'] = torch.tensor(
                np.asarray(teacher_component_diverged_any, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_tolerance_value'] = torch.tensor(
                np.asarray(teacher_tolerance_value, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_target_block'] = torch.tensor(
                np.asarray(teacher_target_block, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_cubes_solved'] = torch.tensor(
                np.asarray(teacher_cubes_solved, dtype=np.float32), device=self.device, dtype=torch.float32
            )
            extras['log']['/Teacher/diag_cube_max_target_error'] = torch.tensor(
                np.asarray(teacher_cube_max_target_error, dtype=np.float32), device=self.device, dtype=torch.float32
            )
        
        # Add episode completion metrics to extras
        if episode_rewards:
            extras['episode_rewards'] = episode_rewards
            extras['episode_lengths'] = episode_lengths
        
        # Add detailed metrics to extras if available (for ANY completed episodes)
        if goals_reached:  # Any completed episodes (regardless of reward type)
            extras['episode_sparse_rewards'] = sparse_rewards
            extras['episode_dense_rewards'] = dense_rewards
            extras['goals_reached'] = goals_reached
            extras['distances_to_goal'] = distances
            if killed:
                extras['lethal_terminations'] = killed
            if timeouts:
                extras['timeouts'] = timeouts
            
            # Add RSL-RL compatible logging metrics using extras["log"] format
            # Following RSL-RL VecEnv documentation: keys start with "/" for namespacing
            if not extras.get('log'):
                extras['log'] = {}
                
            # Add goal success rate and other episode completion metrics
            extras['log']['/Episode/goal_success_rate'] = torch.tensor([float(g) for g in goals_reached], device=self.device)
            extras['log']['/Episode/final_distance'] = torch.tensor(distances, device=self.device)
            extras['log']['/Episode/sparse_reward'] = torch.tensor(sparse_rewards, device=self.device)
            extras['log']['/Episode/dense_reward'] = torch.tensor(dense_rewards, device=self.device)
            if killed:
                extras['log']['/Episode/lethal'] = torch.tensor([float(k) for k in killed], device=self.device)
            if timeouts:
                extras['log']['/Episode/timeout'] = torch.tensor([float(t) for t in timeouts], device=self.device)
            if subgoal_avgs:
                extras['log']['/Episode/avg_distance_to_subgoal'] = torch.tensor(subgoal_avgs, device=self.device)
            if subgoal_returns:
                extras['log']['/Episode/subgoal_shaping_return'] = torch.tensor(subgoal_returns, device=self.device)
            if subgoal_transitions:
                extras['log']['/Episode/subgoal_transitions'] = torch.tensor(subgoal_transitions, device=self.device)

        if cube_solved_counts:
            extras['episode_cubes_solved'] = cube_solved_counts
        if cube_total_counts:
            extras['episode_cubes_total'] = cube_total_counts
        if cube_solved_fracs:
            extras['episode_cubes_solved_fraction'] = cube_solved_fracs
        if cube_max_errors:
            extras['episode_cube_max_target_error'] = cube_max_errors
        if cube_solved_counts or cube_total_counts or cube_solved_fracs or cube_max_errors:
            if not extras.get('log'):
                extras['log'] = {}
            if cube_solved_counts:
                extras['log']['/Episode/cubes_solved'] = torch.tensor(cube_solved_counts, device=self.device)
            if cube_total_counts:
                extras['log']['/Episode/cubes_total'] = torch.tensor(cube_total_counts, device=self.device)
            if cube_solved_fracs:
                extras['log']['/Episode/cubes_solved_fraction'] = torch.tensor(cube_solved_fracs, device=self.device)
                # Alias to make panel naming obvious for intervention-progress tracking.
                extras['log']['/Episode/subgoal_progress'] = torch.tensor(cube_solved_fracs, device=self.device)
            if cube_max_errors:
                extras['log']['/Episode/cube_max_target_error'] = torch.tensor(cube_max_errors, device=self.device)

        # Add teacher metrics to extras['log'] if any episodes completed with them
        if any(len(v) > 0 for v in teacher_metrics.values()):
            if not extras.get('log'):
                extras['log'] = {}
            for k, v in teacher_metrics.items():
                if v:
                    # namespace as /Teacher/
                    extras['log'][f'/Teacher/{k}'] = torch.tensor(v, device=self.device, dtype=torch.float32)
        
        # Return TensorDict for observations
        # Also include a raw observation copy to ease off-policy adapters
        extras.setdefault('observations', {})
        extras['observations'].setdefault('raw', {})
        extras['observations']['raw']['obs'] = obs_tensor
        obs_tensordict = TensorDict(
            {"policy": obs_tensor},
            batch_size=[self.num_envs],
        )
        self._sync_passive_viewer()
        
        return obs_tensordict, rewards_tensor, dones_tensor, extras

    def _apply_intervention_visuals(self) -> None:
        if mujoco is None or not self.envs:
            return
        self._ensure_viewer_material_cache()
        if not self._viewer_original_rgba:
            return
        visual_state = self._visual_state_from_info(self._passive_viewer_info)
        if visual_state == self._viewer_last_visual_state:
            return
        if visual_state == "none":
            self._restore_original_materials()
            self._viewer_last_visual_state = visual_state
            return
        base = self.envs[0].unwrapped
        model = getattr(base, "_model", None)
        if model is None:
            return
        handle = getattr(base, "_passive_viewer_handle", None)
        lock_ctx = handle.lock() if handle is not None else None
        if lock_ctx is not None:
            lock_ctx.__enter__()
        try:
            self._restore_original_rgba_in_place(model)
            if visual_state == "gripper_only":
                self._tint_materials(
                    model,
                    self._viewer_gripper_material_ids,
                    np.array([1.00, 0.72, 0.18, 1.0], dtype=np.float32),
                )
            elif visual_state == "full":
                self._tint_materials(
                    model,
                    self._viewer_arm_material_ids,
                    np.array([0.95, 0.18, 0.18, 1.0], dtype=np.float32),
                )
                self._tint_materials(
                    model,
                    self._viewer_gripper_material_ids,
                    np.array([1.00, 0.30, 0.30, 1.0], dtype=np.float32),
                )
        finally:
            if lock_ctx is not None:
                lock_ctx.__exit__(None, None, None)
        self._viewer_last_visual_state = visual_state

    def _ensure_viewer_material_cache(self) -> None:
        if mujoco is None or not self.envs:
            return
        base = self.envs[0].unwrapped
        model = getattr(base, "_model", None)
        if model is None:
            return
        model_ptr = self._model_cache_key(model)
        if self._viewer_cached_model_ptr == model_ptr and self._viewer_original_rgba:
            return
        self._viewer_cached_model_ptr = model_ptr
        self._viewer_arm_material_ids = []
        self._viewer_gripper_material_ids = []
        self._viewer_original_rgba = {}
        nmat = int(getattr(model, "nmat", 0))
        for mat_id in range(nmat):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mat_id) or ""
            self._viewer_original_rgba[mat_id] = np.asarray(model.mat_rgba[mat_id], dtype=np.float32).copy()
            if name.startswith("ur5e/robotiq/") or "/robotiq/" in name:
                self._viewer_gripper_material_ids.append(mat_id)
            elif name.startswith("ur5e/"):
                self._viewer_arm_material_ids.append(mat_id)
        self._viewer_last_visual_state = None

    @staticmethod
    def _model_cache_key(model) -> int:
        ptr = getattr(model, "ptr", None)
        if ptr is not None:
            try:
                return int(ptr)
            except Exception:
                pass
        return id(model)

    def _restore_original_rgba_in_place(self, model) -> None:
        for mat_id, rgba in self._viewer_original_rgba.items():
            model.mat_rgba[mat_id] = rgba

    def _restore_original_materials(self) -> None:
        if mujoco is None or not self.envs or not self._viewer_original_rgba:
            return
        base = self.envs[0].unwrapped
        model = getattr(base, "_model", None)
        handle = getattr(base, "_passive_viewer_handle", None)
        if model is None or handle is None:
            return
        with handle.lock():
            self._restore_original_rgba_in_place(model)
        self._viewer_last_visual_state = "none"

    @staticmethod
    def _tint_materials(model, material_ids: list[int], rgba: np.ndarray) -> None:
        for mat_id in material_ids:
            orig = np.asarray(model.mat_rgba[mat_id], dtype=np.float32).copy()
            blended = 0.25 * orig + 0.75 * rgba
            blended[3] = float(orig[3])
            model.mat_rgba[mat_id] = blended

    @staticmethod
    def _visual_state_from_info(info: Optional[dict[str, Any]]) -> str:
        if not isinstance(info, dict):
            return "none"
        if not bool(info.get("teacher_intervened", False)):
            return "none"
        reason = str(info.get("teacher_reason", "") or "")
        component = str(info.get("teacher_reason_component", "") or "")
        if reason in {"gripper", "gripper_lock", "gripper_sync"} or component == "gripper":
            return "gripper_only"
        return "full"
    
    def close(self):
        """Close all environments."""
        self._close_passive_viewer()
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()

    def switch_env(
        self,
        env_name: str,
        wrappers: Optional[List[Callable]] = None,
        curriculum_steps: Optional[int] = None,
        **env_kwargs,
    ):
        """Rebuild internal environments with a new OGBench variant.

        Args:
            env_name: New gym environment id to instantiate.
            wrappers: Optional wrapper callables to apply. Defaults to the
                current wrapper list when omitted.
            **env_kwargs: Extra arguments forwarded to ``gym.make``.

        Notes:
            This method is primarily used for curriculum runs where the hazard
            dynamics change after a certain number of global steps (e.g.,
            switching from ``danger-wall`` to ``danger-lethal``). We fully
            reconstruct the underlying gym environments to avoid any hidden
            state carrying over between phases.
        """

        # Close existing envs before rebuilding to free simulator resources.
        self.close()

        self.env_name = env_name
        if wrappers is not None:
            self.wrappers = wrappers
        if env_kwargs:
            self.env_kwargs = dict(env_kwargs)
            self._visualize_intervention_colors = bool(self.env_kwargs.pop("visualize_intervention_colors", self._visualize_intervention_colors))

        self.envs = []
        detailed_wrappers: list[Any] = []
        intervention_wrappers: list[Any] = []
        for _ in range(self.num_envs):
            env = self._make_env(env_name)
            for wrapper_fn in self.wrappers:
                env = wrapper_fn(env)
            # Track particular wrappers so we can restore curriculum progress later
            detailed_wrappers.append(_find_wrapper(env, 'DetailedRewardWrapper'))
            intervention_wrappers.append(_find_wrapper(env, 'InterventionWrapper'))
            self.envs.append(env)

        sample_env = self.envs[0]
        # Update derived properties in case the observation/action spaces differ.
        self.observation_space = sample_env.observation_space
        self.action_space = sample_env.action_space
        self.num_actions = gym.spaces.flatdim(sample_env.action_space)
        self.num_obs = gym.spaces.flatdim(sample_env.observation_space)
        if hasattr(sample_env, '_max_episode_steps'):
            self.max_episode_length = sample_env._max_episode_steps
        elif hasattr(sample_env, 'spec') and sample_env.spec and sample_env.spec.max_episode_steps:
            self.max_episode_length = sample_env.spec.max_episode_steps

        # If requested, restore curriculum counters so we don't restart warmups from scratch.
        if curriculum_steps is not None:
            try:
                from ogbench.wrappers.reward_wrapper import DetailedRewardWrapper
            except Exception:  # pragma: no cover - optional import guard
                DetailedRewardWrapper = None
            try:
                from ogbench.wrappers.intervention_wrappers import InterventionWrapper
            except Exception:  # pragma: no cover
                InterventionWrapper = None

            for w in detailed_wrappers:
                if DetailedRewardWrapper is not None and isinstance(w, DetailedRewardWrapper):
                    w._global_step_env = int(curriculum_steps)
            for w in intervention_wrappers:
                if InterventionWrapper is not None and isinstance(w, InterventionWrapper):
                    w._global_step_env = int(curriculum_steps)

        # Reset the new environments so the adapter receives a clean observation batch.
        self.reset()

    def _make_env(self, env_name: str):
        try:
            return gym.make(env_name, **self.env_kwargs)
        except TypeError as exc:
            if self._strip_pixel_camera_kwargs(exc):
                return gym.make(env_name, **self.env_kwargs)
            raise

    def _strip_pixel_camera_kwargs(self, exc: Exception) -> bool:
        message = str(exc)
        if not any(key in message for key in self._PIXEL_CAMERA_KEYS):
            return False
        for key in self._PIXEL_CAMERA_KEYS:
            if key in self.env_kwargs:
                self.env_kwargs.pop(key)
        warnings.warn(
            f"Environment '{self.env_name}' does not accept pixel camera kwargs; falling back to defaults."
        )
        return True

    def seed(self, seed: int = -1) -> int:
        """Set random seed for all environments."""
        for i, env in enumerate(self.envs):
            if hasattr(env, 'seed'):
                env.seed(seed + i if seed >= 0 else seed)
        return seed

    def _modify_action_space(self):
        """Modifies the action space to the clip range."""
        if self.clip_actions is None:
            return

        # Create new clipped action space
        # Note: This modifies the action space bounds but doesn't affect the actual environments
        # The clipping is done in the step method
        clipped_action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        # Update action space (this is mainly for informational purposes)
        self.action_space = clipped_action_space


def _find_wrapper(env: gym.Env, class_name: str):
    """Traverse wrapper chain to find a wrapper by class name."""
    current = env
    while hasattr(current, 'env'):
        if current.__class__.__name__ == class_name:
            return current
        current = current.env
    return current if current.__class__.__name__ == class_name else None
