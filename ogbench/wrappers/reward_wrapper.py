"""
Reward wrapper for OGBench environments with detailed reward tracking.

This wrapper provides:
- Sparse rewards (goal reached = 1.0, otherwise 0.0) 
- Dense rewards (distance-based progress)
- Combined rewards (sparse + dense)
- Detailed metrics tracking
"""

import numpy as np
import gymnasium as gym


class DetailedRewardWrapper(gym.RewardWrapper):
    """
    Detailed reward wrapper that separates sparse and dense rewards for analysis.
    
    Provides three reward types:
    - sparse: Goal-only rewards (1.0 for goal, 0.0 otherwise)
    - dense: Distance-based progress rewards
    - combined: sparse + dense_weight * dense
    """
    
    def __init__(self, env,
                 reward_type='sparse',
                 dense_reward_scale=0.1,
                 goal_reward=1.0,
                 step_penalty=0.0,
                 normalize_success_reward: bool = True,
                 # Subgoal shaping (reward-only) options
                 use_subgoal_shaping: bool = False,
                 subgoal_shaping_coef: float = 1.0,
                 subgoal_shaping_gamma: float = 0.99,
                 curriculum_stage1_steps_per_env: int = 0,
                 log_subgoal_metrics: bool = True,
                 # Reward schedule: switch to sparse after N per-env steps (0 disables)
                 switch_reward_to_sparse_after_steps_per_env: int = 0):
        """
        Initialize the detailed reward wrapper.
        
        Args:
            env: The base environment
            reward_type: 'sparse', 'dense', or 'combined'
            dense_reward_scale: Scale factor for dense rewards
            goal_reward: Reward for reaching goal
            step_penalty: Small penalty per step (negative reward for time)
        """
        super().__init__(env)
        
        self.reward_type = reward_type
        self.dense_reward_scale = dense_reward_scale
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.normalize_success_reward = bool(normalize_success_reward)

        # Subgoal shaping configuration
        self.use_subgoal_shaping = use_subgoal_shaping
        self.subgoal_shaping_coef = float(subgoal_shaping_coef)
        self.subgoal_shaping_gamma = float(subgoal_shaping_gamma)
        self.curriculum_stage1_steps_per_env = int(curriculum_stage1_steps_per_env or 0)
        self.log_subgoal_metrics = log_subgoal_metrics
        # Reward schedule config (per-env)
        self.switch_reward_to_sparse_after_steps_per_env = int(switch_reward_to_sparse_after_steps_per_env or 0)

        # Track previous position for dense reward calculation
        self._prev_agent_pos = None
        self._goal_pos = None
        self._prev_distance = None

        # Subgoal shaping internals
        self._prev_potential = None  # Phi(s_{t-1}) wrt active subgoal
        self._last_active_subgoal = None  # cache last subgoal position for logging
        self._last_subgoal_index = None
        self._global_step_env = 0  # per-env step counter for curriculum cutoff

        # Episode statistics
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._episode_steps = 0
        self._goal_reached = False
        # Subgoal logs
        self._episode_subgoal_shaping_return = 0.0
        self._episode_subgoal_distance_sum = 0.0
        self._episode_subgoal_steps = 0
        self._episode_subgoal_transitions = 0
        # Per-step bookkeeping for logging
        self._last_dense_reward_step = 0.0
        self._last_sparse_reward_step = 0.0
    
    def _extract_positions(self, obs):
        """Extract agent and goal positions from observation or environment state."""
        agent_pos, goal_pos = self._agent_goal_from_env()
        if agent_pos is not None and goal_pos is not None:
            return agent_pos, goal_pos

        # Assume obs format from FlexibleObsWrapper: [agent_x, agent_y, goal_x, goal_y, ...]
        if isinstance(obs, np.ndarray) and obs.ndim == 1 and len(obs) >= 4:
            agent_pos = obs[:2]
            goal_pos = obs[2:4]
        else:
            # Fallback for non-vector observations (e.g., pixels): reuse tracked positions.
            # This avoids shape bugs such as slicing image tensors into pseudo "positions".
            agent_pos = self._prev_agent_pos if self._prev_agent_pos is not None else np.zeros(2, dtype=np.float32)
            goal_pos = self._goal_pos if self._goal_pos is not None else np.zeros(2, dtype=np.float32)
        
        return np.array(agent_pos, dtype=np.float32), np.array(goal_pos, dtype=np.float32)

    def _agent_goal_from_env(self):
        """Try to read agent and goal positions directly from the base environment."""
        env = self.unwrapped
        agent_pos = goal_pos = None

        try:
            if hasattr(env, 'get_xy'):
                agent_pos = np.array(env.get_xy(), dtype=np.float32)
            elif hasattr(env, 'get_agent_ball_xy'):
                agent_pos = np.array(env.get_agent_ball_xy()[0], dtype=np.float32)
        except Exception:
            agent_pos = None

        try:
            if hasattr(env, 'cur_goal_xy'):
                goal_pos = np.array(env.cur_goal_xy, dtype=np.float32)
        except Exception:
            goal_pos = None

        if agent_pos is not None and goal_pos is not None:
            return agent_pos, goal_pos
        return None, None
    
    def reset(self, **kwargs):
        """Reset environment and tracking."""
        obs, info = self.env.reset(**kwargs)
        
        # Extract positions
        self._prev_agent_pos, self._goal_pos = self._extract_positions(obs)
        env_agent, env_goal = self._agent_goal_from_env()
        if env_agent is not None:
            self._prev_agent_pos = env_agent
        if env_goal is not None:
            self._goal_pos = env_goal
        
        # Calculate initial distance
        self._prev_distance = np.linalg.norm(self._goal_pos - self._prev_agent_pos)

        # Reset curriculum counter at episode start only if first episode
        # (we keep a global per-env counter across episodes)
        # Initialize subgoal potential
        self._prev_potential = None
        self._last_active_subgoal = None
        self._last_subgoal_index = None

        # Reset episode tracking
        self._episode_sparse_reward = 0.0
        self._episode_dense_reward = 0.0
        self._episode_steps = 0
        self._goal_reached = False
        self._episode_subgoal_shaping_return = 0.0
        self._episode_subgoal_distance_sum = 0.0
        self._episode_subgoal_steps = 0
        self._episode_subgoal_transitions = 0

        # Add initial metrics to info
        info.update({
            'episode_sparse_reward': self._episode_sparse_reward,
            'episode_dense_reward': self._episode_dense_reward,
            'episode_steps': self._episode_steps,
            'goal_reached': self._goal_reached,
            'distance_to_goal': self._prev_distance,
        })

        if self.log_subgoal_metrics:
            # Compute initial subgoal (if API available)
            subgoal_xy, subgoal_index = self._get_active_subgoal(self._prev_agent_pos, self._goal_pos)
            if subgoal_xy is not None:
                self._last_active_subgoal = subgoal_xy
                self._last_subgoal_index = subgoal_index
                dist_to_subgoal = float(np.linalg.norm(subgoal_xy - self._prev_agent_pos))
                self._prev_potential = -dist_to_subgoal
                self._episode_subgoal_distance_sum += dist_to_subgoal
                self._episode_subgoal_steps += 1
                info.update({
                    'distance_to_subgoal': dist_to_subgoal,
                    'subgoal_index': subgoal_index if subgoal_index is not None else -1,
                })

        return obs, info
    
    def reward(self, reward):
        """Calculate detailed reward based on type."""
        # Get current observation (last observation from environment)
        try:
            # Get current state
            current_obs = self.unwrapped.get_ob()  # Get raw observation
            current_agent_pos, current_goal_pos = self._extract_positions(current_obs)
        except:
            # Fallback if get_ob() not available
            current_agent_pos = self._prev_agent_pos
            current_goal_pos = self._goal_pos
        
        # Calculate current distance
        current_distance = np.linalg.norm(current_goal_pos - current_agent_pos)

        # Calculate rewards
        sparse_reward = float(reward)  # Original reward (1.0 for goal, 0.0 otherwise)
        dense_reward = 0.0

        # Dense reward based on either classic distance progress (to final goal)
        # or potential-based shaping to oracle subgoal (reward-only subgoals)
        if self.reward_type in ('dense', 'combined'):
            if self.use_subgoal_shaping and self._is_stage1_enabled():
                # Potential-based shaping towards active subgoal
                subgoal_xy, subgoal_index = self._get_active_subgoal(current_agent_pos, current_goal_pos)
                if subgoal_xy is not None:
                    # Track subgoal transitions for logging
                    if self._last_active_subgoal is not None:
                        if np.linalg.norm(self._last_active_subgoal - subgoal_xy) > 1e-6:
                            self._episode_subgoal_transitions += 1
                    self._last_active_subgoal = subgoal_xy
                    self._last_subgoal_index = subgoal_index

                    # Compute potential difference
                    prev_pos = self._prev_agent_pos if self._prev_agent_pos is not None else current_agent_pos
                    phi_prev = self._prev_potential if self._prev_potential is not None else -np.linalg.norm(subgoal_xy - prev_pos)
                    phi_curr = -np.linalg.norm(subgoal_xy - current_agent_pos)
                    shaping = self.subgoal_shaping_coef * (self.subgoal_shaping_gamma * phi_curr - phi_prev)
                    dense_reward = float(shaping)
                    self._prev_potential = phi_curr

                    # Per-step logging accumulators
                    self._episode_subgoal_shaping_return += dense_reward
                    self._episode_subgoal_distance_sum += float(np.linalg.norm(subgoal_xy - current_agent_pos))
                    self._episode_subgoal_steps += 1
                else:
                    # Fallback to classic dense in case API unavailable
                    if self._prev_distance is not None:
                        distance_progress = self._prev_distance - current_distance
                        dense_reward = distance_progress * self.dense_reward_scale
            else:
                # Classic dense progress to final goal
                if self._prev_distance is not None:
                    distance_progress = self._prev_distance - current_distance
                    dense_reward = distance_progress * self.dense_reward_scale

        # Step penalty
        step_reward = -self.step_penalty
        
        # Update tracking
        self._episode_sparse_reward += sparse_reward
        self._episode_dense_reward += dense_reward
        self._episode_steps += 1
        
        if sparse_reward > 0:
            self._goal_reached = True
        
        # Update previous state
        self._prev_agent_pos = current_agent_pos.copy()
        self._prev_distance = current_distance
        
        # Return reward based on effective type (supports switch to sparse)
        effective_type = self._effective_reward_type()
        if effective_type == 'sparse':
            dense_reward = 0.0
        elif effective_type == 'dense':
            sparse_reward = 0.0
        elif effective_type == 'combined':
            pass
        elif effective_type == 'none':
            sparse_reward = 0.0
            dense_reward = 0.0
        else:
            raise ValueError(f"Unknown reward_type: {effective_type}")

        total_reward = sparse_reward + dense_reward + step_reward

        # Save step-level components for logging
        self._last_sparse_reward_step = sparse_reward
        self._last_dense_reward_step = dense_reward

        return total_reward

    def _effective_reward_type(self) -> str:
        """Return reward type with schedule: switch to sparse after configured steps."""
        if self.reward_type == 'none':
            return 'none'
        if self.switch_reward_to_sparse_after_steps_per_env > 0 and \
           self._global_step_env >= self.switch_reward_to_sparse_after_steps_per_env:
            return 'sparse'
        return self.reward_type
    
    def step(self, action):
        """Step with detailed reward tracking."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Some manipulation tasks expose episode completion via `success` while
        # returning non-positive sparse rewards (e.g., {-N, ..., 0}). Mirror
        # that signal into `goal_reached` so training/eval success metrics align.
        success_flag = False
        if isinstance(info, dict) and "success" in info:
            raw_success = info.get("success", False)
            if isinstance(raw_success, (np.ndarray, list, tuple)):
                raw_arr = np.asarray(raw_success).reshape(-1)
                success_flag = bool(raw_arr[0]) if raw_arr.size else False
            else:
                success_flag = bool(raw_success)
            if success_flag:
                self._goal_reached = True
        
        # Calculate detailed reward
        detailed_reward = self.reward(reward)

        # Optional normalization for manipulation-style sparse rewards:
        # convert sparse component from task-specific scale (e.g. -N..0) to 0/1.
        if self.normalize_success_reward and isinstance(info, dict) and "success" in info:
            effective_type = self._effective_reward_type()
            if effective_type in ("sparse", "combined"):
                binary_sparse = self.goal_reward if success_flag else 0.0
                # reward() already accumulated the original sparse scalar.
                self._episode_sparse_reward += float(binary_sparse - float(reward))
                self._last_sparse_reward_step = float(binary_sparse)
                dense_component = self._last_dense_reward_step if effective_type == "combined" else 0.0
                detailed_reward = float(binary_sparse + dense_component - self.step_penalty)
        
        # Get current positions for metrics
        current_agent_pos = self._prev_agent_pos
        current_goal_pos = self._goal_pos
        try:
            current_agent_pos, current_goal_pos = self._extract_positions(obs)
        except Exception:
            pass
        env_agent, env_goal = self._agent_goal_from_env()
        if env_agent is not None:
            current_agent_pos = env_agent
        if env_goal is not None:
            current_goal_pos = env_goal
        if current_agent_pos is not None and current_goal_pos is not None:
            current_distance = np.linalg.norm(current_goal_pos - current_agent_pos)
        else:
            current_distance = self._prev_distance if self._prev_distance is not None else 0.0
        
        # Curriculum bookkeeping (per-env global step)
        self._global_step_env += 1

        # Update info with detailed metrics
        effective_type = self._effective_reward_type()
        info.update({
            'sparse_reward': float(self._last_sparse_reward_step),
            'dense_reward': float(self._last_dense_reward_step),
            'total_reward': detailed_reward,
            'episode_sparse_reward': self._episode_sparse_reward,
            'episode_dense_reward': self._episode_dense_reward,
            'episode_steps': self._episode_steps,
            'goal_reached': self._goal_reached,
            'goal_reached_from_success': bool(success_flag),
            'distance_to_goal': current_distance,
            'reward_type': effective_type,
        })

        # Subgoal metrics in info (for logging only, not observed by policy)
        if self.log_subgoal_metrics:
            subgoal_xy, subgoal_index = self._last_active_subgoal, self._last_subgoal_index
            if subgoal_xy is None:
                # Try to compute once here if not available yet
                if isinstance(obs, np.ndarray) and obs.ndim == 1 and obs.shape[0] >= 4:
                    agent_pos = obs[:2]
                    goal_pos = obs[2:4]
                else:
                    agent_pos, goal_pos = self._agent_goal_from_env()
                    if agent_pos is None:
                        agent_pos = self._prev_agent_pos
                    if goal_pos is None:
                        goal_pos = self._goal_pos
                sg, idx = self._get_active_subgoal(agent_pos, goal_pos)
                subgoal_xy, subgoal_index = sg, idx
                if sg is not None and self._prev_potential is None:
                    self._prev_potential = -float(np.linalg.norm(sg - agent_pos))
            if subgoal_xy is not None:
                agent_for_metrics = current_agent_pos if current_agent_pos is not None else self._prev_agent_pos
                info.update({
                    'distance_to_subgoal': float(np.linalg.norm(subgoal_xy - agent_for_metrics)),
                    'subgoal_index': subgoal_index if subgoal_index is not None else -1,
                    'subgoal_shaping_coef': float(self.subgoal_shaping_coef if self._is_stage1_enabled() and self.use_subgoal_shaping else 0.0),
                    'curriculum_stage': 1 if (self._is_stage1_enabled() and self.use_subgoal_shaping) else 2,
                })

        # If episode ended, flush episode-level subgoal stats
        if terminated or truncated:
            if self._episode_subgoal_steps > 0:
                info.update({
                    'episode_avg_distance_to_subgoal': self._episode_subgoal_distance_sum / max(1, self._episode_subgoal_steps),
                    'episode_subgoal_shaping_return': self._episode_subgoal_shaping_return,
                    'episode_subgoal_transitions': self._episode_subgoal_transitions,
                })

        return obs, detailed_reward, terminated, truncated, info

    # -----------------
    # Helper functions
    # -----------------
    def _get_active_subgoal(self, agent_xy: np.ndarray, goal_xy: np.ndarray):
        """Query the base env for the oracle subgoal. Returns (subgoal_xy, index_or_none).

        Falls back to None if API not available.
        """
        try:
            # Gymnasium wrappers expose base env via .unwrapped
            sg = self.unwrapped.get_oracle_subgoal(agent_xy, goal_xy)
            if isinstance(sg, (list, tuple)) and len(sg) >= 1:
                subgoal_xy = np.array(sg[0], dtype=np.float32)
                index = sg[1] if len(sg) > 1 else None
                return subgoal_xy, index
            elif sg is not None:
                subgoal_xy = np.array(sg, dtype=np.float32)
                return subgoal_xy, None
        except Exception:
            pass
        return None, None

    def _is_stage1_enabled(self) -> bool:
        """Return True if curriculum Stage 1 (with shaping) is active for this env instance."""
        if not self.use_subgoal_shaping:
            return False
        if self.curriculum_stage1_steps_per_env <= 0:
            return True
        return self._global_step_env < self.curriculum_stage1_steps_per_env


def test_reward_wrapper():
    """Test the reward wrapper with different configurations."""
    import ogbench
    from ogbench.wrappers import FlexibleObsWrapper
    
    print("🧪 Testing DetailedRewardWrapper")
    print("=" * 50)
    
    reward_types = ['sparse', 'dense', 'combined']
    
    for reward_type in reward_types:
        print(f"\n{reward_type.upper()} Rewards:")
        
        env = gym.make("pointmaze-arena-v0", render_mode=None)
        env = FlexibleObsWrapper(env, include_goal=True)
        env = DetailedRewardWrapper(env, 
                                  reward_type=reward_type,
                                  dense_reward_scale=0.01)
        
        obs, info = env.reset(seed=42)
        print(f"   Initial distance: {info['distance_to_goal']:.3f}")
        
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step+1}: reward={reward:.4f}, sparse={info.get('sparse_reward', 0):.4f}, "
                  f"distance={info['distance_to_goal']:.3f}")
            
            if term:
                print(f"   🎯 Goal reached!")
                break
        
        print(f"   Total reward: {total_reward:.4f}")
        print(f"   Goal reached: {info['goal_reached']}")
        
        env.close()
    
    print("\n✅ Reward wrapper tests complete!")


if __name__ == "__main__":
    test_reward_wrapper()
