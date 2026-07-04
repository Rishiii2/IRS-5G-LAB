import numpy as np

class ChannelEnvironment:
    """
    Models the physical radio propagation environment including Path Loss and Fading.
    """
    def __init__(self, bs_pos, irs_pos, ue_pos, frequency=28e9):
        """
        Args:
            bs_pos (tuple): (x, y, z) coordinates of the Base Station.
            irs_pos (tuple): (x, y, z) coordinates of the IRS center.
            ue_pos (tuple): (x, y, z) coordinates of the User Equipment.
            frequency (float): Carrier frequency in Hz.
        """
        self.bs_pos = np.array(bs_pos)
        self.irs_pos = np.array(irs_pos)
        self.ue_pos = np.array(ue_pos)
        self.freq = frequency
        self.c = 3e8
        
    def _distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def calculate_free_space_path_loss(self, d):
        """
        Calculates Free Space Path Loss in linear scale (not dB).
        L = (4 * pi * d * f / c)^2
        Return the channel gain (inverse of path loss).
        """
        if d == 0:
            return 1.0
        pl_linear = (4 * np.pi * d * self.freq / self.c) ** 2
        return 1.0 / pl_linear

    def generate_rician_channel(self, N, K_factor_dB):
        """
        Generates a Rician fading channel vector of size N.
        Useful for BS -> IRS link (usually has strong Line of Sight).
        """
        K = 10**(K_factor_dB / 10.0)
        
        # LoS component (deterministic)
        h_los = np.ones(N, dtype=complex) # Simplified LoS array response
        
        # NLoS component (Rayleigh fading)
        h_nlos = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        
        # Combine
        h = np.sqrt(K / (K + 1)) * h_los + np.sqrt(1 / (K + 1)) * h_nlos
        return h

    def generate_steering_vector(self, num_elements, pos_tx, pos_rx):
        """
        Generates a spatial steering vector for a URA (Uniform Rectangular Array).
        Assumes elements are arranged in a square grid (sqrt(N) x sqrt(N)).
        Assumes half-wavelength spacing.
        """
        N = num_elements
        side_len = int(np.sqrt(N))
        
        # Calculate direction vector
        direction = pos_rx - pos_tx
        dist = np.linalg.norm(direction)
        if dist == 0:
            return np.ones(N, dtype=complex)
            
        direction = direction / dist
        
        # Wavenumber
        k = 2 * np.pi * self.freq / self.c
        d_spacing = (self.c / self.freq) / 2 # lambda / 2
        
        # Vectorized grid calculation
        col, row = np.meshgrid(np.arange(side_len), np.arange(side_len))
        dx = (col - (side_len - 1) / 2) * d_spacing
        dz = (row - (side_len - 1) / 2) * d_spacing
        
        phase = k * (dx * direction[0] + dz * direction[2])
        return np.exp(-1j * phase).flatten()

    def reset_fading(self, irs_elements_N):
        """
        Resets the small-scale fading (NLoS) realizations for the environment.
        Should be called at the beginning of an episode to simulate a static block-fading room.
        """
        self.h_nlos_bs = (np.random.randn(irs_elements_N) + 1j * np.random.randn(irs_elements_N)) / np.sqrt(2)
        self.h_nlos_ue = (np.random.randn(irs_elements_N) + 1j * np.random.randn(irs_elements_N)) / np.sqrt(2)
        self.h_d_nlos = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)

    def get_channels(self, irs_elements_N):
        """
        Returns the G matrix (BS->IRS) and h_r vector (IRS->UE).
        Includes large-scale path loss and spatial steering vectors.
        """
        d_bs_irs = self._distance(self.bs_pos, self.irs_pos)
        d_irs_ue = self._distance(self.irs_pos, self.ue_pos)
        
        # 1. Path Loss (Large scale)
        pl_bs_irs = self.calculate_free_space_path_loss(d_bs_irs)
        pl_irs_ue = self.calculate_free_space_path_loss(d_irs_ue)
        
        # 2. Spatial Steering Vectors (LoS dominant for both links to see beams clearly)
        a_bs_irs = self.generate_steering_vector(irs_elements_N, self.bs_pos, self.irs_pos)
        a_irs_ue = self.generate_steering_vector(irs_elements_N, self.irs_pos, self.ue_pos)
        
        # 3. Add slight fading (Rician with high K factor)
        K = 10.0 # 10dB
        
        if not hasattr(self, 'h_nlos_bs') or len(self.h_nlos_bs) != irs_elements_N:
            self.reset_fading(irs_elements_N)
            
        fading_bs_irs = np.sqrt(K / (K + 1)) * a_bs_irs + np.sqrt(1 / (K + 1)) * self.h_nlos_bs
        fading_irs_ue = np.sqrt(K / (K + 1)) * a_irs_ue + np.sqrt(1 / (K + 1)) * self.h_nlos_ue
        
        # 4. Combine
        G = np.sqrt(pl_bs_irs) * fading_bs_irs
        h_r = np.sqrt(pl_irs_ue) * fading_irs_ue
        
        # Direct link (BS -> UE) - assuming heavily blocked
        d_bs_ue = self._distance(self.bs_pos, self.ue_pos)
        pl_bs_ue = self.calculate_free_space_path_loss(d_bs_ue) * 1e-6 # 60dB blockage penalty
        h_d = np.sqrt(pl_bs_ue) * self.h_d_nlos
        
        return G, h_r, h_d
