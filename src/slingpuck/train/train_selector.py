import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.slingpuck.envs.match_env import MatchEnv
from src.slingpuck.eval.vulnerability_window import run_rule_based_baseline

def train_and_evaluate_ablation(tell_strength, config, phase1_model, phase2_model, log_dir):
    config['opponent']['tell_strength'] = tell_strength
    run_name = f"tell_{tell_strength:.1f}"
    
    def make_env():
        return MatchEnv(config, phase1_model, phase2_model)
        
    env = make_vec_env(make_env, n_envs=8, seed=42)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
        learning_rate=3e-4,
        n_steps=2048
    )
    
    print(f"\n--- Training Selector (Tell Strength: {tell_strength}) ---")
    model.learn(total_timesteps=500_000, tb_log_name=run_name)
    model_path = os.path.join(log_dir, f"selector_model_{run_name}")
    model.save(model_path)
    
    # Evaluate RL Agent
    eval_env = make_env()
    wins, losses, draws = 0, 0, 0
    for _ in range(100):
        obs, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc
        
        if eval_env.agent_score > eval_env.opp_score:
            wins += 1
        elif eval_env.opp_score > eval_env.agent_score:
            losses += 1
        else:
            draws += 1
            
    rl_win_rate = wins / 100.0
    
    # Evaluate Baseline
    print(f"Evaluating rule-based baseline at tell={tell_strength}...")
    base_win_rate = run_rule_based_baseline(eval_env, num_episodes=100)
    
    print(f"Results for Tell Strength {tell_strength}:")
    print(f"  RL Win Rate: {rl_win_rate*100:.1f}%")
    print(f"  Baseline Win Rate: {base_win_rate*100:.1f}%")
    print(f"  Advantage: {(rl_win_rate - base_win_rate)*100:+.1f}%")
    
    return model_path

if __name__ == "__main__":
    with open("configs/physics_params.dev.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    log_dir = "logs/selector_ablations"
    os.makedirs(log_dir, exist_ok=True)
    
    # Load frozen primitives (paths assume M2 and M3 have been run)
    phase1_model = PPO.load("logs/gk_final.zip")
    phase2_model = PPO.load("logs/sling_final.zip")
    
    # Sweep over tell strengths
    for tell in [0.0, 0.5, 1.0]:
        train_and_evaluate_ablation(tell, config, phase1_model, phase2_model, log_dir)