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
    def __init__(self, room_size=100.0, num_elements_x=16, num_elements_y=16):
        super(IRSEnv, self).__init__()
        
        self.room_size = room_size
        self.bs_pos = (0, room_size/2, 5)
        self.irs_pos = (room_size/2, 0, 3)
        self.irs_panel = IRSArray(num_elements_x, num_elements_y)
        self.N = self.irs_panel.N
        
        # State: UE (x, y) coordinates normalized to [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Action: Continuous phases for each element in range [-1, 1] which will map to [-pi, pi]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.N,), dtype=np.float32)
        
        self.ue_pos = None
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
        
        # Reset the environment fading for the new episode
        self.env_model.reset_fading(self.N)
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize (x, y) to [0, 1]
        return (self.ue_pos / self.room_size).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        
        # 1. Map action [-1, 1] to phases [-pi, pi]
        phases = action * np.pi
        self.irs_panel.phases = phases
        Phi = self.irs_panel.get_reflection_matrix()
        
        # 2. Get Channels
        self.env_model.ue_pos = np.array([self.ue_pos[0], self.ue_pos[1], 1.5])
        G, h_r, h_d = self.env_model.get_channels(self.N)
        
        # 3. Calculate SNR (Reward)
        gain = np.dot(np.conj(h_r).T, np.dot(Phi, G)) + h_d
        power_db = 10 * np.log10(np.abs(gain)**2 + 1e-20)
        
        # Reward is the received power. We can normalize it slightly to help training.
        # Max theoretical power is around -110dB to -130dB in this setup.
        reward = power_db
        
        # 4. Simulate User Movement (Random walk)
        self.ue_pos += np.random.normal(0, 1.0, size=2)
        self.ue_pos = np.clip(self.ue_pos, 10, self.room_size - 10)
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {}
