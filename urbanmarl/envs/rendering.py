"""
UrbanMARL 3D Environment Renderer
Supports buildings, UAV base stations, and ground UEs visualization.
"""
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class UrbanRenderConfig:
    """Configuration for urban environment rendering."""
    figsize: Tuple[int, int] = (1024, 768)
    dpi: int = 100
    building_color: str = '#8B4513'      # Saddle brown
    building_alpha: float = 0.3
    uav_color: str = '#FF4444'            # Red
    ue_color: str = '#4444FF'             # Blue
    base_station_color: str = '#44FF44'   # Green
    trajectory_color: str = '#FFA500'     # Orange
    show_trajectory: bool = True
    trajectory_length: int = 100
    camera_elev: float = 45.0
    camera_azim: float = -120.0
    show_labels: bool = True


class Urban3DRenderer:
    """
    3D renderer for UrbanMARL environments.
    
    Renders:
    - Buildings as 3D cuboids
    - UAVs as flying markers with trajectories
    - Ground UEs as markers on the ground plane
    - Base stations as tall markers
    - Communication links (LoS/NLoS)
    """
    
    def __init__(self, config: Optional[UrbanRenderConfig] = None):
        self.config = config or UrbanRenderConfig()
        self._fig = None
        self._ax = None
        self._trajectories: Dict[str, List[np.ndarray]] = {}
        self._frame_count = 0
        
    def _init_plot(self, volume_size: List[float]):
        """Initialize matplotlib 3D plot."""
        try:
            import matplotlib
            # matplotlib.use('Agg')  # Headless backend
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            raise ImportError("matplotlib required for rendering. Install: pip install matplotlib")
            
        self._plt = plt
        self._Poly3DCollection = Poly3DCollection
        
        fig = plt.figure(figsize=(self.config.figsize[0]/self.config.dpi, 
                                   self.config.figsize[1]/self.config.dpi),
                         dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set volume bounds
        ax.set_xlim(-volume_size[0]/2, volume_size[0]/2)
        ax.set_ylim(-volume_size[1]/2, volume_size[1]/2)
        ax.set_zlim(0, volume_size[2])
        
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Set camera angle
        ax.view_init(
            elev=self.config.camera_elev, 
            azim=self.config.camera_azim)
        
        self._fig = fig
        self._ax = ax
        
    def _draw_buildings(self, buildings: List[Dict]):
        """Draw buildings as 3D cuboids."""
        for b in buildings:
            x, y, z = b['position']
            w, l, h = b['size']
            
            # Cuboid vertices
            vertices = np.array([
                [x, y, z], [x+w, y, z], [x+w, y+l, z], [x, y+l, z],      # bottom
                [x, y, z+h], [x+w, y, z+h], [x+w, y+l, z+h], [x, y+l, z+h]  # top
            ])
            
            # Faces
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[3], vertices[0], vertices[4], vertices[7]],  # left
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            ]
            
            poly3d = self._Poly3DCollection(faces, alpha=self.config.building_alpha,
                                             facecolor=self.config.building_color,
                                             edgecolor='black', linewidth=0.5)
            self._ax.add_collection3d(poly3d)
            
    def _draw_uavs(
        self, 
        uav_positions: np.ndarray, 
        uav_names: Optional[List[str]] = None
    ):
        """Draw UAVs as 3D markers with trajectories."""
        n_uavs = len(uav_positions)
        names = uav_names or [f'UAV-{i}' for i in range(n_uavs)]
        
        for i, pos in enumerate(uav_positions):
            pos = np.asarray(pos).flatten()
            
            # Update trajectory
            name = names[i]
            if name not in self._trajectories:
                self._trajectories[name] = []
            pos_copy = pos.copy()
            if (len(self._trajectories[name]) == 0 
                or not np.array_equal(pos_copy, self._trajectories[name][-1])):
                self._trajectories[name].append(pos_copy)
            
            if len(self._trajectories[name]) > self.config.trajectory_length:
                self._trajectories[name].pop(0)
            
            # Draw UAV marker (octahedron-like)
            self._ax.scatter(*pos, c=self.config.uav_color, s=100, marker='o', 
                            edgecolors='black', linewidths=1, alpha=0.9)
            
            # Draw trajectory
            if self.config.show_trajectory and len(self._trajectories[name]) > 1:
                traj = np.array(self._trajectories[name])
                self._ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                             color=self.config.trajectory_color, alpha=0.5, linewidth=1)
            
            # Label
            if self.config.show_labels:
                self._ax.text(pos[0], pos[1], pos[2]+5, name, fontsize=8)
                
    def _draw_ues(self, ue_positions: np.ndarray, ue_names: Optional[List[str]] = None):
        """Draw ground UEs as markers on the ground plane."""
        n_ues = len(ue_positions)
        names = ue_names or [f'UE-{i}' for i in range(n_ues)]
        
        for i, pos in enumerate(ue_positions):
            pos = np.asarray(pos).flatten()
            # UEs are on ground (z=0 or close to it)
            self._ax.scatter(
                pos[0], 
                pos[1], 
                pos[2] if pos.shape[0] > 2 else 0, 
                c=self.config.ue_color, 
                s=50, 
                marker='s', 
                edgecolors='black', 
                linewidths=0.5, 
                alpha=0.8
                )
            if self.config.show_labels:
                self._ax.text(pos[0], pos[1], 2, names[i], fontsize=6)
                
    def _draw_base_stations(self, bs_positions: np.ndarray, bs_names: Optional[List[str]] = None):
        """Draw base stations as tall cylindrical markers."""
        n_bs = len(bs_positions)
        names = bs_names or [f'BS-{i}' for i in range(n_bs)]
        
        for i, pos in enumerate(bs_positions):
            pos = np.asarray(pos).flatten()
            # Draw tall marker
            self._ax.scatter(
                pos[0], pos[1], pos[2], 
                c=self.config.base_station_color, 
                s=150, marker='^', edgecolors='black', 
                linewidths=1.5)
            # Draw vertical line to ground
            self._ax.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, pos[2]], 
                         'g--', alpha=0.5, linewidth=1)
            if self.config.show_labels:
                self._ax.text(pos[0], pos[1], pos[2]+8, names[i], fontsize=8)
                
    def _draw_links(self, links: List[Dict]):
        """Draw communication links between entities."""
        for link in links:
            src = np.asarray(link['source'])
            dst = np.asarray(link['target'])
            los = link.get('los', True)
            
            color = '#00FF00' if los else '#FF0000'  # Green for LoS, Red for NLoS
            style = '-' if los else '--'
            alpha = 0.3 if los else 0.15
            
            self._ax.plot([src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]],
                         color=color, linestyle=style, alpha=alpha, linewidth=0.8)
                         
    def _draw_ground_plane(self, volume_size: List[float]):
        """Draw semi-transparent ground plane."""
        xx, yy = np.meshgrid(
            np.linspace(-volume_size[0]/2, volume_size[0]/2, 2),
            np.linspace(-volume_size[1]/2, volume_size[1]/2, 2)
        )
        zz = np.zeros_like(xx)
        self._ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
        
    def render(
        self, 
        state: Dict, 
        mode: str = 'rgb_array'
    ) -> Optional[np.ndarray]:
        """
        Render the urban environment.
        
        Args:
            state: Dictionary containing environment state with keys:
                - 'volume_size': [x, y, z] dimensions
                - 'buildings': list of building dicts with 'position' and 'size'
                - 'uav_positions': (n_uavs, 3) array
                - 'ue_positions': (n_ues, 3) array  
                - 'base_station_positions': (n_bs, 3) array
                - 'links': optional list of link dicts
            mode: 'rgb_array' or 'human'
            
        Returns:
            RGB array if mode=='rgb_array', None otherwise
        """
        volume_size = state.get('volume_size', [500, 500, 200])
        
        # Initialize plot if needed
        if self._fig is None:
            self._init_plot(volume_size)
        else:
            self._ax.clear()
            self._ax.set_xlim(-volume_size[0]/2, volume_size[0]/2)
            self._ax.set_ylim(-volume_size[1]/2, volume_size[1]/2)
            self._ax.set_zlim(0, volume_size[2])
            self._ax.set_xlabel('X (m)')
            self._ax.set_ylabel('Y (m)')
            self._ax.set_zlabel('Z (m)')
            self._ax.view_init(
                elev=self.config.camera_elev, 
                azim=self.config.camera_azim)
        self._ax.grid(visible=False)
        # Draw ground plane
        self._draw_ground_plane(volume_size)
        
        # Draw buildings
        if 'buildings' in state:
            self._draw_buildings(state['buildings'])
            
        # Draw base stations
        if 'base_station_positions' in state:
            self._draw_base_stations(state['base_station_positions'])
            
        # Draw UEs
        if 'ue_positions' in state:
            self._draw_ues(state['ue_positions'])
            
        # Draw UAVs (on top)
        if 'uav_positions' in state:
            self._draw_uavs(state['uav_positions'])
            
        # Draw communication links
        if 'links' in state:
            self._draw_links(state['links'])
            
        # Set title with frame info
        if 'title' in state:
            title = state['title'] #+ f' - Frame {self._frame_count}'
        else:
            title = f'UrbanMARL 3D Environment - Frame {self._frame_count}'
        self._ax.set_title(title)
        self._frame_count += 1
        
        if mode == 'rgb_array':
            self._fig.tight_layout()
            self._fig.canvas.draw()
            # Get RGB buffer
            buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            # print(f"buf.shape: {buf.shape}, width_height: {self._fig.canvas.get_width_height()}, expeted: {self._fig.canvas.get_width_height()[::-1] + (4,)}")
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
            self._plt.close(self._fig)
            self._fig = None
            self._ax = None
            return buf[..., :3]  # Return RGB only
        elif mode == 'human':
            # self._plt.pause(0.001)
            return self._fig
            
    def close(self):
        """Clean up renderer resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self._trajectories.clear()