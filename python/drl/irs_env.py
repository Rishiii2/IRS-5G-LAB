import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

# Add parent directory to path to import physics models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from physics.channel_model import ChannelEnvironment
from physics.irs_array import IRSArray

class IRSEnv(gym.Env):
    """
    Custom Environment for optimizing IRS phases using DRL.
    """
    def __init__(self, room_size=100.0, num_elements_x=8, num_elements_y=8):
        super(IRSEnv, self).__init__()
        
        self.room_size = room_size
        self.bs_pos = (0, room_size/2, 5)
        self.irs_pos = (room_size/2, 0, 3)
        self.irs_panel = IRSArray(num_elements_x, num_elements_y)
        self.N = self.irs_panel.N
        
        # State: [ue_x, ue_y, vel_x, vel_y, d_bs, d_irs, prev_snr]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # Action: Continuous delta-phases for each element in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)
        
        self.ue_pos = None
        self.ue_velocity = None
        self.current_phases = np.zeros(self.N, dtype=np.float32)
        self.prev_snr_db = 0.0
        
        self.env_model = ChannelEnvironment(self.bs_pos, self.irs_pos, (0,0,0))
        
        self.current_step = 0
        self.max_steps = 100 # Steps per episode (UE moving trajectory)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Random starting position for UE
        self.ue_pos = np.array([
            np.random.uniform(20, self.room_size), 
            np.random.uniform(20, self.room_size)
        ])
        
        # Deterministic smooth trajectory (constant velocity)
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.5, 1.5) # meters per step
        self.ue_velocity = np.array([speed * np.cos(angle), speed * np.sin(angle)])
        
        self.current_phases = np.zeros(self.N, dtype=np.float32)
        self.prev_snr_db = 0.0
        
        # Reset the environment fading for the new episode
        self.env_model.reset_fading(self.N)
        
        return self._get_obs(), {}

    def _get_obs(self):
        d_bs = np.linalg.norm(self.ue_pos - np.array(self.bs_pos[:2])) / self.room_size
        d_irs = np.linalg.norm(self.ue_pos - np.array(self.irs_pos[:2])) / self.room_size
        snr_norm = (self.prev_snr_db + 10.0) / 40.0
        
        return np.array([
            self.ue_pos[0] / self.room_size, 
            self.ue_pos[1] / self.room_size,
            self.ue_velocity[0] / 2.0,
            self.ue_velocity[1] / 2.0,
            d_bs,
            d_irs,
            snr_norm
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Map action to a phase delta (max pi/4 radians per step)
        max_delta = np.pi / 4
        self.current_phases += action * max_delta
        # Wrap phases to [-pi, pi] efficiently without complex exponentials
        self.current_phases = (self.current_phases + np.pi) % (2 * np.pi) - np.pi
        self.irs_panel.phases = self.current_phases
        Phi = self.irs_panel.get_reflection_matrix()
        
        # 2. Get Channels
        self.env_model.ue_pos = np.array([self.ue_pos[0], self.ue_pos[1], 1.5])
        G, h_r, h_d = self.env_model.get_channels(self.N)
        
        # 3. Calculate True SNR
        gain = np.dot(np.conj(h_r).T, np.dot(Phi, G)) + h_d
        power_linear = np.abs(gain)**2
        
        # Thermal noise at 28 GHz with 100 MHz bandwidth (-94 dBm = -124 dBW)
        noise_power_linear = 10**(-12.4)
        snr_linear = power_linear / noise_power_linear
        snr_db = 10 * np.log10(snr_linear + 1e-20)
        self.prev_snr_db = snr_db
        
        # Reward is the SNR. Normalize it for SAC.
        # Shift by +10 and scale. SNR usually -10 dB to +30 dB in this config.
        reward = (snr_db + 10.0) / 40.0
        
        # Penalty for aggressive phase flipping (Regularization)
        action_penalty = 0.05 * np.mean(action**2)
        reward -= action_penalty
        
        # 4. Simulate User Movement (Smooth bouncing trajectory)
        self.ue_pos += self.ue_velocity
        
        # Boundary bounce logic
        if self.ue_pos[0] < 10 or self.ue_pos[0] > self.room_size - 10:
            self.ue_velocity[0] *= -1
        if self.ue_pos[1] < 10 or self.ue_pos[1] > self.room_size - 10:
            self.ue_velocity[1] *= -1
            
        self.ue_pos = np.clip(self.ue_pos, 10, self.room_size - 10)
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {}
