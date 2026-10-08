import numpy as np
import yaml
from stable_baselines3 import PPO
from src.slingpuck.envs.goalkeeper_env import GoalkeeperEnv
import matplotlib.pyplot as plt

def evaluate_save_rate(model_path, config_path, num_episodes=500):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env = GoalkeeperEnv(config)
    model = PPO.load(model_path)
    
    results = [] # Tuples of (incoming_speed, saved_boolean)
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        
        # Calculate true initial speed for logging (ground truth, not tracked)
        initial_speed = np.hypot(env.puck_vx, env.puck_vy)
        
        done = False
        saved = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                # If reward > 0.5, it was a save
                if reward > 0.5:
                    saved = True
                done = True
                
        results.append((initial_speed, saved))
        
    # Process into bins
    bins = np.linspace(1.0, 4.0, 7) # 6 bins from 1.0 to 4.0 m/s
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    save_rates = []
    
    results = np.array(results)
    
    for i in range(len(bins)-1):
        mask = (results[:, 0] >= bins[i]) & (results[:, 0] < bins[i+1])
        if np.sum(mask) > 0:
            rate = np.mean(results[mask, 1])
        else:
            rate = 0.0
        save_rates.append(rate)
        print(f"Speed [{bins[i]:.1f} - {bins[i+1]:.1f} m/s]: {rate*100:.1f}% save rate")

    # Optional: Plotting
    plt.plot(bin_centers, save_rates, marker='o', linestyle='-')
    plt.xlabel('Incoming Puck Speed (m/s)')
    plt.ylabel('Save Rate')
    plt.title('Goalkeeper Save Rate vs. Puck Speed')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.savefig('save_rate_curve.png')
    
    return bins, save_rates

if __name__ == "__main__":
    # Example usage (update paths to your trained run)
    # evaluate_save_rate("logs/ppo_goalkeeper_latest/final_model", "configs/physics_params.dev.yaml")
    pass