import gymnasium as gym
from gymnasium import spaces
import numpy as np

from slingpuck.opponent.scripted_opponent import ScriptedOpponent
from slingpuck.envs.goalkeeper_env import GoalkeeperEnv
from slingpuck.envs.sling_env import SlingEnv

class MatchEnv(gym.Env):
    """
    Phase 3: Hierarchical Match Environment.
    Runs at a lower frequency (e.g., 5 Hz decision rate) than the control loop.
    Actions: 0 (BLOCK), 1 (SLING), 2 (HOLD)
    """
    def __init__(self, config, phase1_policy, phase2_policy):
        super().__init__()
        self.config = config
        self.phase1_policy = phase1_policy
        self.phase2_policy = phase2_policy
        
        self.decision_dt = 0.20 # 5 Hz high-level decision
        self.control_dt = 1.0 / config['sim']['control_hz']
        self.steps_per_decision = int(self.decision_dt / self.control_dt)
        
        self.max_time = config['scoring']['max_episode_time_s']
        self.points_to_win = config['scoring']['points_to_win']
        
        # Sub-environments used purely for their physics/state logic
        self.gk_env = GoalkeeperEnv(config)
        self.sling_env = SlingEnv(config)
        self.opponent = ScriptedOpponent(config['opponent'])
        
        self.action_space = spaces.Discrete(3) # 0: BLOCK, 1: SLING, 2: HOLD
        
        # Obs: [puck_x, puck_y, puck_vx, puck_vy, agent_score, opp_score, opp_reloading, opp_tell_angle]
        high_obs = np.array([1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 1.0, 3.14], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_score = 0
        self.opp_score = 0
        self.time_elapsed = 0.0
        
        self.recovery_timer = 0.0 # Time remaining where gate is undefended
        self.gk_env.reset(seed=seed)
        self.opp_obs = self.opponent.reset()
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        
        # Execute the chosen macro-action over `steps_per_decision` control loops
        for _ in range(self.steps_per_decision):
            self.time_elapsed += self.control_dt
            
            # 1. Update Opponent & Check for fired pucks
            fired_puck, self.opp_obs = self.opponent.step(self.control_dt)
            if fired_puck is not None:
                # Opponent shot a puck; inject it into the gk_env physics state
                self.gk_env.puck_x = 0.0 # Originating from opponent center
                self.gk_env.puck_y = self.gk_env.board_length
                self.gk_env.puck_vx, self.gk_env.puck_vy = fired_puck
                
            # 2. Handle Agent Action
            if self.recovery_timer > 0:
                self.recovery_timer -= self.control_dt
                # Gate undefended; paddle is out of the way
                _ = self.gk_env.step([1.0]) # Swing paddle completely out of the way
                
            elif action == 1: # SLING
                # Agent commits to a shot
                sling_obs, _ = self.sling_env.reset()
                sling_action, _ = self.phase2_policy.predict(sling_obs, deterministic=True)
                _, sling_reward, _, _, _ = self.sling_env.step(sling_action)
                
                if sling_reward > 0:
                    self.agent_score += 1
                    reward += 1.0
                    
                # Lock agent into recovery (e.g., 1.5 seconds undefended)
                self.recovery_timer = 1.5 
                
            elif action == 0: # BLOCK
                gk_obs = self.gk_env._get_obs()
                gk_action, _ = self.phase1_policy.predict(gk_obs, deterministic=True)
                _, gk_reward, terminated, _, _ = self.gk_env.step(gk_action)
                if terminated and gk_reward < 0: # Conceded
                    self.opp_score += 1
                    reward -= 1.0
                    self.gk_env.reset() # Reset puck state after a goal
                    
            elif action == 2: # HOLD
                # Park at center
                _, gk_reward, terminated, _, _ = self.gk_env.step([0.0])
                if terminated and gk_reward < 0:
                    self.opp_score += 1
                    reward -= 1.0
                    self.gk_env.reset()

        terminated = bool(self.agent_score >= self.points_to_win or self.opp_score >= self.points_to_win)
        truncated = bool(self.time_elapsed >= self.max_time)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.array([
            self.gk_env.puck_x, self.gk_env.puck_y,
            self.gk_env.puck_vx, self.gk_env.puck_vy,
            self.agent_score, self.opp_score,
            self.opp_obs[0], self.opp_obs[1] # reloading bool, tell angle
        ], dtype=np.float32)