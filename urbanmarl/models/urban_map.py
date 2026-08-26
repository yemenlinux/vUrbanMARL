import torch
from torch import Tensor

import numpy as np

class VectorizedUrbanMap:
    """Tensor-accelerated ITU-R P.1410 procedural map generator and LoS engine.

    Handles procedural 3D urban heightmap generation, line-of-sight (LoS) ray tracing,
    collision checking, and agent coordinate transformations across parallel environments.

    Attributes:
        batch_size (int): Number of parallel environment instances.
        volume_size (tuple): 3D volume dimensions (sim_x, sim_y, sim_z).
        device (torch.device): PyTorch compute device.
        map_margin (int): Margin boundary in meters for map edge buildings.
        height_maps (torch.Tensor): Tensor heightmaps of shape (batch_size, sim_x, sim_y).
    """

    def __init__(
        self, 
        batch_size: int, 
        volume_size: tuple,
        device: torch.device,
        map_margin: int = 5,
    ) -> None:
        """Initializes the VectorizedUrbanMap generator.

        Args:
            batch_size (int): Number of parallel environment instances.
            volume_size (tuple): Volume dimensions (X, Y, Z).
            device (torch.device): Compute device for tensor operations.
            map_margin (int): Margin buffer from volume borders. Defaults to 5.
        """
        self.batch_size = batch_size
        self.num_envs = batch_size
        self.volume_size = volume_size
        self.device = device
        self.map_margin = map_margin

        self.x_min, self.y_min = -volume_size[0] // 2, -volume_size[1] // 2
        self.x_max, self.y_max = volume_size[0] // 2, volume_size[1] // 2
        self.sim_x, self.sim_y = volume_size[0], volume_size[1]
        self.z_min = 0.0
        if len(volume_size) > 2:
            self.z_max = volume_size[2]
            self.sim_z = volume_size[2]
        else:
            self.z_max = 500
            self.sim_z = 500

        self.height_maps = torch.zeros(
            (batch_size, self.sim_x, self.sim_y),
            dtype=torch.float32,
            device=device,
        )
        self.info = None
        self.has_envs = False

    def get_info_labels(self) -> list[str]:
        """Returns feature metadata column names for procedural parameters.

        Returns:
            list[str]: List of parameter names ('alpha', 'beta', etc.).
        """
        return [
            'alpha',
            'beta',
            'gamma',
            'E',
            'n_buildings',
            'building_width',
            'street_width',
        ]

    def create_building_faces(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                              w: np.ndarray, l: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Constructs 3D polygon face vertices for building cuboids (Fully Vectorized).
        
        Args:
            x, y, z: np.ndarray of base coordinates.
            w, l, h: np.ndarray of building dimensions.
            
        Returns:
            np.ndarray: Shape (N*5, 4, 3) defining all cuboid faces for all buildings.
        """
        # Ensure inputs are 1D arrays
        x, y, z = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
        w, l, h = np.atleast_1d(w), np.atleast_1d(l), np.atleast_1d(h)

        N = len(x)
        x1, y1, z1 = x + w, y + l, z + h

        # Create all 8 vertices for all N buildings at once: shape (N, 8, 3)
        vertices = np.zeros((N, 8, 3), dtype=np.float32)
        vertices[:, 0] = np.column_stack((x, y, z))
        vertices[:, 1] = np.column_stack((x1, y, z))
        vertices[:, 2] = np.column_stack((x1, y1, z))
        vertices[:, 3] = np.column_stack((x, y1, z))
        vertices[:, 4] = np.column_stack((x, y, z1))
        vertices[:, 5] = np.column_stack((x1, y, z1))
        vertices[:, 6] = np.column_stack((x1, y1, z1))
        vertices[:, 7] = np.column_stack((x, y1, z1))

        # Map vertices to 5 faces (front, right, back, left, top): shape (N, 5, 4, 3)
        faces = np.zeros((N, 5, 4, 3), dtype=np.float32)
        faces[:, 0] = vertices[:, [0, 1, 5, 4]]
        faces[:, 1] = vertices[:, [1, 2, 6, 5]]
        faces[:, 2] = vertices[:, [2, 3, 7, 6]]
        faces[:, 3] = vertices[:, [3, 0, 4, 7]]
        faces[:, 4] = vertices[:, [4, 5, 6, 7]]

        # Flatten into the continuous format expected by Poly3DCollection
        return faces.reshape(N * 5, 4, 3)
    
    def _build_ITU_P1410_maps(self, num_urbans: int = 72, seed: int = 0):
        """Generates urban maps based on ITU-R P1410-5.

        Args:
            num_urbans (int, optional): number of urban maps to generate. Defaults to 72.
            seed (int, optional): random seed. Defaults to 0. This is independent on the
                torch random seed, and is used to generate the procedural parameters for 
                the urban maps.
        """
        if num_urbans != self.batch_size:
            num_urbans = self.batch_size 
            
        self.building_data = [] 
        self.building_faces = [] 
        
        rng = np.random.RandomState(seed) 
        samples = 10000 
        environment_rad = np.round(np.deg2rad(np.linspace(-180, 180, num_urbans)), 4) 
        
        _alpha = np.round(rng.uniform(0.1, 0.8, samples), 2) 
        _beta = rng.uniform(100, 750, samples).astype(int) 
        _gamma = np.round(rng.uniform(8, 50, samples), 2) 
        
        building_width = 1000 * np.sqrt(_alpha / _beta) 
        street_width = 1000 / np.sqrt(_beta) - building_width 
        ex = (street_width - building_width) + 1j * (street_width - _gamma) 
        complex_repr = np.round(np.arctan2(ex.imag, ex.real), 4) 
        
        area_sq_km = (self.sim_x * self.sim_y) / 1e6 
        
        # Batch select environment configurations
        abs_diffs = np.abs(complex_repr[:, np.newaxis] - environment_rad) 
        closest_indices = abs_diffs.argmin(axis=0) 
        
        alphas = _alpha[closest_indices] 
        betas = _beta[closest_indices] 
        gammas = _gamma[closest_indices] 
        Es = complex_repr[closest_indices] 
        ws = np.maximum(1, building_width[closest_indices]).astype(int) 
        ss = np.maximum(1, street_width[closest_indices]).astype(int) 
        
        h_maps_batch = np.zeros((self.batch_size, self.sim_x, self.sim_y), dtype=np.float32) 
        info_batch = np.zeros((self.batch_size, 7), dtype=np.float32) 

        for env_idx in range(self.batch_size): 
            w, s = ws[env_idx], ss[env_idx] 
            n_buildings = int(betas[env_idx] * area_sq_km) 
            
            if n_buildings == 0: 
                self.building_data.append([]) 
                self.building_faces.append([]) 
                info_batch[env_idx] = [alphas[env_idx], betas[env_idx], gammas[env_idx], Es[env_idx], 0, w, s] 
                continue 
            
            b_x = np.arange(self.map_margin, self.sim_x - w, w + s).astype(int) 
            b_y = np.arange(self.map_margin, self.sim_y - w, w + s).astype(int) 
            nx, ny = len(b_x), len(b_y)
            total_slots = nx * ny 
            
            if total_slots > 0: 
                # Vectorized Map Population 
                XX, YY = np.meshgrid(b_x, b_y, indexing='ij')
                heights_2d = rng.rayleigh(gammas[env_idx], size=XX.shape)
                
                # Expand heights into block sizes and reshape cleanly into grid
                blocks = np.zeros((nx, w + s, ny, w + s), dtype=np.float32)
                blocks[:, :w, :, :w] = heights_2d[:, np.newaxis, :, np.newaxis]
                grid_2d = blocks.reshape(nx * (w + s), ny * (w + s))
                
                # The exact physical end of the last building
                end_x = int(b_x[-1] + w)
                end_y = int(b_y[-1] + w)
                
                # The relative length to slice from the 2D grid
                len_x = end_x - self.map_margin
                len_y = end_y - self.map_margin
                
                # Assign the cleanly sliced grid
                h_maps_batch[env_idx, self.map_margin:end_x, self.map_margin:end_y] = grid_2d[:len_x, :len_y]
                
                # Vectorized Building Faces & Data (Removes individual dict processing)
                X_flat = XX.flatten() + self.x_min
                Y_flat = YY.flatten() + self.y_min
                H_flat = heights_2d.flatten()
                W_flat = np.full(total_slots, w)
                Z_flat = np.zeros(total_slots)
                
                # Pass full arrays directly into the new face generator
                faces = self.create_building_faces(X_flat, Y_flat, Z_flat, W_flat, W_flat, H_flat)
                self.building_faces.append(faces.tolist()) 
                
                # Fast Python list comprehension for data storage
                b_data = [{"position": [x.item(), y.item(), 0], "size": [w, w, h.item()]} 
                          for x, y, h in zip(X_flat, Y_flat, H_flat)]
                self.building_data.append(b_data) 
            else:
                self.building_data.append([]) 
                self.building_faces.append([]) 

            info_batch[env_idx] = [alphas[env_idx], betas[env_idx], gammas[env_idx], Es[env_idx], n_buildings, w, s] 
            
        self.height_maps = torch.tensor(h_maps_batch, dtype=torch.float32, device=self.device) 
        self.info = torch.tensor(info_batch, dtype=torch.float32, device=self.device) 
    
    def reset(self, return_urban_info: bool = False):
        """
        Resets the map environment by regenerating the height maps with new procedural parameters.
        Optionally returns the generated map information for analysis or observation purposes.
        """
        if self.info is None:
            self.info = torch.zeros((self.batch_size, 7), # alpha, beta, gamma, E, n_buildings, w, s
                                    dtype=torch.float32, device=self.device)
        if not self.has_envs:
            self._build_ITU_P1410_maps(num_urbans=self.batch_size)
            self.has_envs = True
        # if self.has_envs:
        #     for env_idx in range(self.batch_size):
        #         self.reset_at(env_idx)
        # else:
        #     self._build_ITU_P1410_maps(num_urbans=self.batch_size)
        #     self.has_envs = True
        if return_urban_info:
            return {label: self.info[:, i] 
                    for i, label in enumerate(self.get_info_labels())}  # Return a copy to prevent external modifications

    def reset_at(self, env_idx: int, return_urban_info: bool = False):
        if not hasattr(self, 'building_data') or self.building_data is None: 
            self.building_data = [[] for _ in range(self.batch_size)] 
        if not hasattr(self, 'building_faces') or self.building_faces is None: 
            self.building_faces = [[] for _ in range(self.batch_size)] 
        if self.info is None: 
            self.reset(return_urban_info=return_urban_info) 
        if env_idx < 0 or env_idx >= self.batch_size: 
            raise ValueError(f"env_idx must be between 0 and {self.batch_size - 1}") 
            
        alpha, beta, gamma, E, n_buildings, w, s = self.info[env_idx] 
        n_buildings, w, s = int(n_buildings), int(w), int(s) 
        
        h_map = self.height_maps[env_idx].cpu().numpy() 
        b_x = np.arange(self.map_margin, self.sim_x - w, w + s).astype(int) 
        b_y = np.arange(self.map_margin, self.sim_y - w, w + s).astype(int) 
        nx, ny = len(b_x), len(b_y)
        total_slots = nx * ny 
        
        building_data, building_faces = [], [] 
        
        if total_slots > 0: 
            gamma_val = self.info[env_idx, 2].item() 
            heights_2d = np.random.rayleigh(gamma_val, size=(nx, ny)) 
            
            # Vectorized Map Population
            blocks = np.zeros((nx, w + s, ny, w + s), dtype=np.float32)
            blocks[:, :w, :, :w] = heights_2d[:, np.newaxis, :, np.newaxis]
            grid_2d = blocks.reshape(nx * (w + s), ny * (w + s))
            
            # The exact physical end of the last building
            end_x = int(b_x[-1] + w)
            end_y = int(b_y[-1] + w)
            
            # The relative length to slice from the 2D grid
            len_x = end_x - self.map_margin
            len_y = end_y - self.map_margin
            
            # Assign the cleanly sliced grid
            h_map[self.map_margin:end_x, self.map_margin:end_y] = grid_2d[:len_x, :len_y]
            
            
            # Vectorized Face Generation
            XX, YY = np.meshgrid(b_x, b_y, indexing='ij')
            X_flat = XX.flatten() + self.x_min
            Y_flat = YY.flatten() + self.y_min
            H_flat = heights_2d.flatten()
            W_flat = np.full(total_slots, w)
            Z_flat = np.zeros(total_slots)
            
            faces = self.create_building_faces(X_flat, Y_flat, Z_flat, W_flat, W_flat, H_flat)
            building_faces = faces.tolist()
            
            building_data = [{"position": [x.item(), y.item(), 0], "size": [w, w, h.item()]} 
                             for x, y, h in zip(X_flat, Y_flat, H_flat)]
                             
        self.height_maps[env_idx] = torch.tensor(h_map, dtype=torch.float32, device=self.device) 
        self.building_data[env_idx] = building_data 
        self.building_faces[env_idx] = building_faces 
        
        if return_urban_info: 
            return {label: self.info[env_idx, i].item() for i, label in enumerate(self.get_info_labels())} 

    def get_building_polygon(
        self,
        urban_idx: int, 
        face_color: str = '#696969',  # gray '#A9A9A9'
        alpha: float = 0.3
    ):
        """
        Creates a Poly3DCollection for 3D plotting of building faces.
        Args:
            urban_idx (int): Index of the urban environment.
            face_color (str): Color of the building faces.
            alpha (float): Transparency level of the building faces.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        return Poly3DCollection(
            self.building_faces[urban_idx],
            alpha=alpha,
            facecolor=face_color,
            edgecolor='black', 
            linewidth=0.5)
    
    def generate_batch_maps(
        self, 
        alpha: torch.Tensor = None, 
        beta: torch.Tensor = None, 
        gamma: torch.Tensor = None
    ):
        """
        Procedurally populates building heights based on structural distribution properties.
        Executes on CPU/Numpy for procedural layouts, then uploads directly to device buffers.
        """
        if not hasattr(self, 'building_data') or self.building_data is None:
            self.building_data = [[] for _ in range(self.batch_size)]
        if not hasattr(self, 'building_faces') or self.building_faces is None:
            self.building_faces = [[] for _ in range(self.batch_size)]
        if self.info is None:
            self.info = torch.zeros((self.batch_size, 7), dtype=torch.float32, device=self.device)
        # Initialize internal environmental generation seeds dynamically
        if alpha is None:
            """alpha is the build-up ratio to the total area in km^2.
            alpha ranges from 0.1 to 0.8, 
            where higher values indicate denser urban environments."""
            alpha = torch.rand(self.batch_size, device=self.device) * 0.7 + 0.1
        if beta is None:
            """beta is the building density per km^2,
            ranging from 100 to 750 buildings per km^2,
            where higher values indicate more densely packed urban environments."""
            beta = torch.randint(100, 750, (self.batch_size,), device=self.device).float()
        if gamma is None:
            """gamma is the average building height in meters,
            ranging from 8 to 50 meters,
            where higher values indicate taller urban structures."""
            gamma = torch.rand(self.batch_size, device=self.device) * 42.0 + 8.0
        # Convert parameter specs to local host properties for precise structural layout logic
        alpha_np = alpha.cpu().numpy()
        beta_np = beta.cpu().numpy()
        gamma_np = gamma.cpu().numpy()
        # complex representation
        # building_width = int(max(1, 1000 * np.sqrt(alpha_np / beta_np)))
        building_width =  1000 * np.sqrt(alpha_np / beta_np)
        street_width = 1000/np.sqrt(beta_np) - building_width
        ex = (street_width - building_width) + 1j * (street_width - gamma_np) # good
        complex_repr = np.round(np.arctan2(ex.imag, ex.real), 4)
        
        area_sq_km = (self.sim_x * self.sim_y) / 1e6
        
        for b in range(self.batch_size):
            h_map = np.zeros((self.sim_x, self.sim_y), dtype=np.float32)
            n_buildings = int(beta_np[b] * area_sq_km)
            if n_buildings == 0:
                continue
            
            # w = int(max(1, 1000 * np.sqrt(alpha_np[b] / beta_np[b])))
            # s = int(max(1, (1000 / np.sqrt(beta_np[b])) - w))
            w = int(max(1, building_width[b]))
            s = int(max(1, street_width[b]))
            
            # Use deterministic grid lines to rapidly populate building profiles
            b_x = np.arange(self.map_margin, self.sim_x - w, w + s)
            b_y = np.arange(self.map_margin, self.sim_y - w, w + s)
            
            # Sample heights uniformly matching global parameters
            total_slots = len(b_x) * len(b_y)
            if total_slots > 0:
                heights = np.random.rayleigh(gamma_np[b], size=total_slots)
                idx = 0
                for x in b_x:
                    for y in b_y:
                        h_map[x:x+w, y:y+w] = heights[idx]
                        idx += 1
                        
            self.height_maps[b] = torch.tensor(h_map, 
                                               dtype=torch.float32, 
                                               device=self.device)
            self.info[b] = torch.tensor([alpha_np[b].item(), 
                                         beta_np[b].item(), 
                                         gamma_np[b].item(),
                                         complex_repr[b].item(),
                                         n_buildings,
                                         building_width[b].item(),
                                         street_width[b].item()],
                                               dtype=torch.float32, 
                                               device=self.device)

    def check_los_batch(
        self, 
        p1: torch.Tensor, 
        p2: torch.Tensor, 
        n_steps: int = 20
    ) -> torch.Tensor:
        """
        Computes line-of-sight status via dense vector projection array checking.
        p1: shape (batch_size, n_agents, 3)
        p2: shape (batch_size, n_targets, 3)
        Returns: boolean tensor shape (batch_size, n_agents, n_targets)
        """
        B, N, _ = p1.shape
        _, M, _ = p2.shape
        
        # Prepare execution tensors broadcasting along continuous dimensions
        # p1_exp: (B, N, 1, 3), p2_exp: (B, 1, M, 3)
        p1_exp = p1.unsqueeze(2)
        p2_exp = p2.unsqueeze(1)
        
        # Interpolate points along direct spatial vectors
        t = torch.linspace(0, 1, steps=n_steps, device=self.device).view(1, 1, 1, n_steps, 1)
        # Segments: shape (B, N, M, n_steps, 3)
        segments = p1_exp.unsqueeze(3) + t * (p2_exp.unsqueeze(3) - p1_exp.unsqueeze(3))
        
        # Map physical coordinates directly to storage grid indices
        grid_x = torch.clamp(((segments[..., 0] - self.x_min)).long(), 0, self.sim_x - 1)
        grid_y = torch.clamp(((segments[..., 1] - self.y_min)).long(), 0, self.sim_y - 1)
        seg_z = segments[..., 2]
        
        # Batch index indexing mapping profiles
        b_idx = torch.arange(B, device=self.device).view(B, 1, 1, 1).expand(B, N, M, n_steps)
        sampled_heights = self.height_maps[b_idx, grid_x, grid_y]
        
        # A clear spatial path maintains continuous clearance directly above blocking bodies
        clearance = seg_z - sampled_heights
        min_clearance, _ = torch.min(clearance, dim=-1)
        
        return min_clearance >= 0.0 
    
    def check_collision_batch(
        self, 
        p1: torch.Tensor, 
        p2: torch.Tensor, 
        n_steps: int = 20
    ) -> torch.Tensor:
        """
        Computes collision status via dense vector projection array checking.
        p1: shape (batch_size, n_agents, 3) previous positions
        p2: shape (batch_size, n_agents, 3) current positions
        Returns: boolean tensor shape (batch_size, n_agents, 1) indicating collision status
        """
        los = self.check_los_batch(p1, p2, n_steps)
        collisions = torch.zeros(
            (p1.shape[0], p1.shape[1], 1), 
            device=self.device, 
            dtype=torch.bool)
        for a in range(p1.shape[1]):
            collisions[:, a] = ~los[:, a, a].unsqueeze(-1)
        
        return collisions
        
    
    def sort(self):
        # Placeholder for potential future sorting or indexing optimizations
        pass
    
    def gen_pos(
        self,
        num_pos: int,
        min_z: float = None,
        max_z: float = None,
        batch_idx: int = None,
        outdoor: bool = False,
        normalized: bool = False
    ) -> torch.Tensor:
        """Generates positions for agents, optionally ensuring they are outdoors."""
        min_z = self.z_min if min_z is None else min_z
        max_z = self.z_max if max_z is None else max_z
        if outdoor:
            result = self.gen_outdoor_pos(num_pos=num_pos, 
                                        min_z=min_z, 
                                        max_z=max_z, 
                                        batch_idx=batch_idx)
            if normalized:
                result[..., 0] = (result[..., 0]) / (self.x_max - self.x_min)
                result[..., 1] = (result[..., 1]) / (self.y_max - self.y_min)
                result[..., 2] = (result[..., 2] - min_z) / (max_z - min_z) if max_z > min_z else max_z
            return result
        else:
            if normalized:
                x = torch.rand((self.batch_size, num_pos), device=self.device) * 2 - 1
                y = torch.rand((self.batch_size, num_pos), device=self.device) * 2 - 1
                z = min_z + torch.rand((self.batch_size, num_pos), device=self.device) 
            else:
                x = torch.rand((self.batch_size, num_pos), device=self.device) * self.sim_x + self.x_min
                y = torch.rand((self.batch_size, num_pos), device=self.device) * self.sim_y + self.y_min
                z = torch.rand((self.batch_size, num_pos), device=self.device) * (max_z - min_z) + min_z
            return torch.stack([x, y, z], dim=-1)
    
    def gen_outdoor_pos(
        self, 
        num_pos: int,
        min_z: float = 0.0,
        max_z: float = 0.0,
        batch_idx: int = None
    ) -> torch.Tensor:
        """Generate outdoor positions"""
        # x and y where height_maps are zero (outdoors)
        if batch_idx is not None:
            xy_indices = (self.height_maps[batch_idx]==0).nonzero()
            shuffled_positions = torch.randperm(len(xy_indices))
            selected_positions = shuffled_positions[:num_pos]
            sampled_indices = xy_indices[selected_positions]
            sampled_indices[:, 0] = sampled_indices[:, 0].float() + self.x_min
            sampled_indices[:, 1] = sampled_indices[:, 1].float() + self.y_min
            z_column = min_z + torch.rand(size=(num_pos, 1), device=self.device) * (max_z - min_z)
            return torch.cat([sampled_indices.view(num_pos, 2),  
                              z_column.view(num_pos, 1)], 
                            #  device=self.device, 
                             dim=-1)
        else:
            result = torch.zeros((self.batch_size, num_pos, 3), device=self.device)
            for b in range(self.batch_size):
                xy_indices = (self.height_maps[b]==0).nonzero()
                # Generate a random permutation of the length of your indices
                shuffled_positions = torch.randperm(len(xy_indices))
                # Slice the first N positions
                selected_positions = shuffled_positions[:num_pos]
                # Extract the actual indices
                sampled_indices = xy_indices[selected_positions]
                # convert between self.x_min/y_min and self.x_max/y_max to actual coordinates
                sampled_indices[:, 0] = sampled_indices[:, 0].float()  + self.x_min
                sampled_indices[:, 1] = sampled_indices[:, 1].float()  + self.y_min

                # z_column
                z_column = min_z + torch.rand(size=(num_pos, 1), device=self.device) * (max_z - min_z)
                # pos = torch.cat([xy_indices[:, 0], xy_indices[:, 1], z_column[:,0]], dim=0)
                result[b] = torch.cat([sampled_indices,  z_column], dim=1)
            return result
        
    def norm_pos(
        self, 
        positions: torch.Tensor,
        min_z: float = None,
        max_z: float = None
    ) -> torch.Tensor:
        """Normalizes positions to [0,1] range based on the map dimensions."""
        min_z = self.z_min if min_z is None else min_z
        max_z = self.z_max if max_z is None else max_z
        x = (positions[..., 0] - self.x_min) / (self.x_max - self.x_min)
        y = (positions[..., 1] - self.y_min) / (self.y_max - self.y_min)
        z = (positions[..., 2] - min_z) / (max_z - min_z) if max_z > min_z else max_z
        return torch.stack([x, y, z], dim=-1)
        
    def denorm_pos(
        self, 
        positions: torch.Tensor, 
        min_z: float = None, 
        max_z: float = None
    ) -> torch.Tensor:
        """Denormalizes positions from [0,1] back to actual coordinate space."""
        min_z = self.z_min if min_z is None else min_z
        max_z = self.z_max if max_z is None else max_z
        x = positions[..., 0] * (self.x_max - self.x_min) + self.x_min
        y = positions[..., 1] * (self.y_max - self.y_min) + self.y_min
        z = positions[..., 2] * (max_z - min_z) + min_z  # Assuming z is already in the correct scale
        return torch.stack([x, y, z], dim=-1)
    
    def pos_to_grid(
        self, 
        positions:torch.Tensor
    ) -> torch.Tensor:
        """Converts physical positions to grid indices for height map lookups."""
        grid_x = torch.clamp(((positions[..., 0] - self.x_min)).long(), 0, self.sim_x - 1)
        grid_y = torch.clamp(((positions[..., 1] - self.y_min)).long(), 0, self.sim_y - 1)
        return torch.stack([grid_x, grid_y], dim=-1)
    
    def plot(
        self,
        batch_idx: int = 0,
        title: str = "Procedural Urban Map Height Profile",
        cmap: str = "viridis",
        axes = None
    ):
        import matplotlib.pyplot as plt
        
        if axes is None:
            plt.figure(figsize=(8, 6))
            ax = plt.gca()
        else:
            ax = axes
        
        alpha = self.info[batch_idx][0].item()
        beta = self.info[batch_idx][1].item()
        gamma = self.info[batch_idx][2].item()
        title += f"\n(alpha={alpha:.2f}, beta={beta:.1f}, gamma={gamma:.1f})"
        
        h_map = self.height_maps[batch_idx].cpu().numpy()
        im = ax.imshow(h_map.T, origin='lower', cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.colorbar(im, ax=ax, label="Building Height (m)")
        
        # if axes is None:
        #     plt.show()
        return ax

    def seed(self, seed=None):
        """
        Sets the seed for the environment
        Args:
            seed (int, optional): Seed for the environment. Defaults to None.

        """
        return self._seed(seed=seed)
    
    def _seed(self, seed=None):
        """
        Internal method to set the seed for the environment
        Args:
            seed (int, optional): Seed for the environment. Defaults to None.

        """
        if seed is None:
            seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def to(self, device: torch.device):
        """Casts the scenario to a different device.

        Args:
            device (Union[str, int, torch.device]): the device to cast to
        """
        for attr, value in self.__dict__.items():
            if isinstance(value, Tensor):
                self.__dict__[attr] = value.to(device)
    