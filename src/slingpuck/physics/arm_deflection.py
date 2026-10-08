import pybullet as p
import pybullet_data
import numpy as np

class DeflectionModel:
    """
    Models the physical deflection of the SO-101 arm under band tension.
    Can operate in a fast analytical mode or query a PyBullet URDF stub.
    """
    def __init__(self, config, use_pybullet=False):
        self.compliance = config['arm'].get('deflection_compliance_m_per_n', 0.001)
        self.use_pybullet = use_pybullet
        
        if self.use_pybullet:
            self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # TODO: Replace plane.urdf with actual SO-101 LeRobot URDF path
            # self.robot_id = p.loadURDF("assets/urdf/so101.urdf", [0,0,0], useFixedBase=True)
            self.robot_id = p.loadURDF("plane.urdf") 
            
    def get_actual_pullback(self, commanded_pullback, band_force):
        """
        Calculates the true pullback distance factoring in the 3D-printed link deflection.
        """
        if self.use_pybullet:
            # TODO: Apply wrench to the end-effector link in PyBullet and read joint states
            # p.applyExternalForce(self.robot_id, end_effector_index, forceObj=[0, -band_force, 0], ...)
            # p.stepSimulation()
            # return measured_fk_position
            pass
            
        # Analytical fallback: Delta x = Force * compliance
        deflection_loss = band_force * self.compliance
        actual_pullback = max(0.0, commanded_pullback - deflection_loss)
        return actual_pullback
        
    def __del__(self):
        if self.use_pybullet:
            p.disconnect(self.physics_client)