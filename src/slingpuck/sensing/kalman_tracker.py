import numpy as np

class KalmanTracker:
    def __init__(self, dt, process_noise=1e-4, measurement_noise=1e-2):
        self.dt = dt
        # State: [x, y, vx, vy]
        self.x = np.zeros((4, 1))
        
        # State transition matrix
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
                           
        # Measurement matrix (we only observe x, y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
                           
        self.P = np.eye(4) # Covariance
        self.Q = np.eye(4) * process_noise # Process noise
        self.R = np.eye(2) * measurement_noise # Measurement noise

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[:2].flatten(), self.x[2:].flatten()

    def update(self, measurement):
        if measurement is None:
            return # Skip update on dropout
            
        z = np.array([[measurement[0]], [measurement[1]]])
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(4)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)