import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
import os
import datetime

def objective_function(params, df, base_config):
    """
    Evaluates how well a set of physics parameters matches the real-world logs.
    params: [compliance, stiffness_k, hysteresis_loss]
    """
    compliance, k, hysteresis = params
    
    # Penalize out-of-bounds parameters heavily
    if compliance < 0 or k < 0 or hysteresis < 0 or hysteresis > 1.0:
        return 1e6
        
    error = 0.0
    puck_mass = base_config['puck'].get('mass_kg', 0.01) # Default if Unknown
    
    for _, row in df.iterrows():
        # Reconstruct the theoretical force and actual pullback
        theoretical_force = k * row['cmd_pull']
        actual_pullback = max(0.0, row['cmd_pull'] - (theoretical_force * compliance))
        
        # Calculate theoretical launch energy
        energy_stored = 0.5 * k * (actual_pullback ** 2)
        energy_released = energy_stored * (1.0 - hysteresis)
        
        if energy_released > 0:
            pred_vel = np.sqrt((2 * energy_released) / puck_mass)
        else:
            pred_vel = 0.0
            
        # MSE against the real measured exit velocity
        error += (pred_vel - row['launch_vel_m_s']) ** 2
        
    return error / len(df)

def run_sysid(csv_path, base_config_path, output_dir="configs/"):
    """
    Ingests real logs and outputs a versioned yaml with fitted physics parameters.
    """
    print(f"Loading real logs from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    with open(base_config_path, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Initial guesses: [compliance, stiffness_k, hysteresis_loss]
    initial_guess = [
        base_config['arm'].get('deflection_compliance_m_per_n', 0.001),
        base_config['band'].get('stiffness_k_n_m', 50.0),
        base_config['band'].get('hysteresis_loss_factor', 0.15)
    ]
    
    print("Running optimization (Nelder-Mead)...")
    res = minimize(
        objective_function, 
        initial_guess, 
        args=(df, base_config), 
        method='Nelder-Mead'
    )
    
    if res.success:
        print(f"Fit successful! Final MSE: {res.fun:.4f}")
        fit_compliance, fit_k, fit_hysteresis = res.x
        
        # Update config
        base_config['arm']['deflection_compliance_m_per_n'] = float(fit_compliance)
        base_config['band']['stiffness_k_n_m'] = float(fit_k)
        base_config['band']['hysteresis_loss_factor'] = float(fit_hysteresis)
        
        # Versioning
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        base_config['version'] = f"calibrated_{timestamp}"
        
        out_file = os.path.join(output_dir, f"physics_params_v{timestamp}.yaml")
        with open(out_file, "w") as f:
            yaml.dump(base_config, f)
            
        print(f"Calibrated configuration saved to: {out_file}")
        print(f"  - Compliance (m/N): {fit_compliance:.6f}")
        print(f"  - Stiffness (N/m): {fit_k:.2f}")
        print(f"  - Hysteresis: {fit_hysteresis:.4f}")
    else:
        print("Optimization failed:", res.message)

if __name__ == "__main__":
    # Example usage:
    # run_sysid("data/real_logs/real_shots_v1.csv", "configs/physics_params.dev.yaml")
    pass