import time
import numpy as np
import onnxruntime as ort

from slingpuck.kinematics import GoalkeeperGeometry
# from lerobot.hardware.motor import STS3215Bus # Conceptual LeRobot hardware interface

class SO101Controller:
    def __init__(self, onnx_path, config):
        self.config = config
        self.control_hz = config['sim']['control_hz']
        self.period = 1.0 / self.control_hz
        
        # Load ONNX Session
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Same action -> pan mapping as the sim (GoalkeeperEnv)
        self.geom = GoalkeeperGeometry.from_config(config)

        # Hardware limits
        self.max_rate_rad_s = np.deg2rad(config['servo']['max_rate_deg_s'])
        self.max_torque = 1.5 # N.m (Feetech STS3215 spec)
        
        # Hardware init (Stub)
        print("Initializing LeRobot hardware bus...")
        # self.bus = STS3215Bus(port="/dev/ttyUSB0")
        # self.pan_motor = self.bus.get_motor(id=1)
        # self.pan_motor.set_torque_limit(self.max_torque)
        self.current_pan_angle = 0.0

    def get_observation(self):
        """
        Polls camera tracker and hardware encoders.
        Returns: [puck_x, puck_y, puck_vx, puck_vy, pan_angle, pan_vel]

        FIXME(M5): must build the same 8-value normalized vector as GoalkeeperEnv._obs
        (GoalkeeperEnv.POLICY_OBS_NAMES). This stub still returns the old 6-value layout.
        """
        # TODO: Replace with live Kalman tracker output
        simulated_tracker = [0.0, 0.2, 0.0, -1.0] 
        # simulated_pan = self.pan_motor.get_position()
        simulated_pan = self.current_pan_angle
        simulated_vel = 0.0 
        
        return np.array([*simulated_tracker, simulated_pan, simulated_vel], dtype=np.float32)

    def write_action(self, action_vector):
        """
        Maps raw policy action to rate-limited hardware commands.
        """
        # Policy action -> board-frame pan -> shoulder_pan joint command (shared with the sim)
        target_angle = self.geom.pan_to_joint(self.geom.action_to_pan(action_vector))
        
        # Apply safety rate limits in software before dispatch
        angle_diff = target_angle - self.current_pan_angle
        max_step = self.max_rate_rad_s * self.period
        safe_step = np.clip(angle_diff, -max_step, max_step)
        
        safe_target = self.current_pan_angle + safe_step
        
        # self.pan_motor.set_position(safe_target)
        self.current_pan_angle = safe_target

    def run_loop(self):
        print("Starting control loop...")
        try:
            while True:
                start_time = time.perf_counter()
                
                # 1. Sense
                obs = self.get_observation()
                obs_batch = np.expand_dims(obs, axis=0)
                
                # 2. Infer
                action = self.session.run(None, {self.input_name: obs_batch})[0][0]
                
                # 3. Act
                self.write_action(action)
                
                # 4. Sleep to maintain frequency
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.period - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("Control loop safely terminated.")
            # self.bus.disable_torque()

if __name__ == "__main__":
    # Example execution stub
    # config = load_config("latest", strict=True)
    # controller = SO101Controller("deploy/gk_policy.onnx", config)
    # controller.run_loop()
    pass