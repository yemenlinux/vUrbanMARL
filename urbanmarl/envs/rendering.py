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
    figsize: Tuple[int, int] = (1280, 820) # (1024, 768)
    dpi: int = 100
    # heat map
    heatmap_color:str = 'gist_yarg'
    heatmap_alpha: float = 0.7
    #
    building_color: str = '#C0C0C0',  # gray '#A9A9A9''#8B4513' # Saddle brown
    building_alpha: float = 0.2
    #
    uav_color: str = '#FF4444'
    uav_marker: str = "^"
    #
    ue_color: str = '#4444FF' # Blue
    ue_marker: str = "."
    #
    base_station_color: str = '#FF4444'   # Green
    base_station_marker: str = "v"
    #
    trajectory_color: str = '#FFA500'     # Orange
    show_trajectory: bool = True
    trajectory_length: int = 100
    #
    link_los_color: str = '#00FF00'  # Green
    link_nlos_color: str = '#FF0000' # Red
    #
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
        self._ax3D = None
        self._ax2D = None
        self._trajectories: Dict[str, List[np.ndarray]] = {}
        self._frame_count = 0
        
    def _init_plot(self, volume_size: List[float]):
        """Initialize matplotlib 3D and 2D plots.
        Args:
            volume_size: [x, y, z] dimensions of the environment"""
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
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.set_title('Urban Environment 3D View')
        # Set volume bounds
        ax3d.set_xlim(-volume_size[0]/2, volume_size[0]/2)
        ax3d.set_ylim(-volume_size[1]/2, volume_size[1]/2)
        ax3d.set_zlim(0, volume_size[2])
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        # ax3d.grid(visible=False)
        # Set camera angle
        ax3d.view_init(
            elev=self.config.camera_elev, 
            azim=self.config.camera_azim)
        # 2D plot for heatmap or other info
        ax2d = fig.add_subplot(122)
        ax2d.set_title('Urban Environment heatmap (top View)')
        ax2d.set_xlim(-volume_size[0]/2, volume_size[0]/2)
        ax2d.set_ylim(-volume_size[1]/2, volume_size[1]/2)
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')
        
        self._fig = fig
        self._ax3D = ax3d
        self._ax2D = ax2d
        
    def _draw_heatmap(
        self, 
        heatmap: np.ndarray, 
        volume_size: List[float]):
        """Draw a heatmap on the 2D subplot."""
        if self._ax2D is None:
            return
        extent = [
            -volume_size[0]/2, 
            volume_size[0]/2, 
            -volume_size[1]/2, 
            volume_size[1]/2]
        tpc = self._ax2D.imshow(
            heatmap.cpu().numpy().T,
            extent=extent,
            origin='lower',
            cmap=self.config.heatmap_color,
            alpha=self.config.heatmap_alpha
        )
        self._fig.colorbar(tpc, 
                           orientation='horizontal', 
                           ax=self._ax2D, 
                           fraction=0.046, 
                           pad=0.04)
        
    def _draw_buildings(
        self,
        buildings: List[Dict]
    ):
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
            self._ax3D.add_collection3d(poly3d)
            
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
            self._ax3D.scatter(*pos, 
                            c=self.config.uav_color, 
                            s=100, 
                            marker=self.config.uav_marker, 
                            edgecolors='black', 
                            linewidths=1, 
                            alpha=0.9)
            # Draw 2D projection on ground plane
            self._ax2D.scatter(pos[0], pos[1], 
                            c=self.config.uav_color, 
                            s=100, 
                            marker=self.config.uav_marker,
                            edgecolors='black',
                            label='UAV' if i == 0 else None,
                            linewidths=1,
                            alpha=0.9)
            
            # Draw trajectory
            if self.config.show_trajectory and len(self._trajectories[name]) > 1:
                traj = np.array(self._trajectories[name])
                self._ax3D.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                             color=self.config.trajectory_color, 
                             alpha=0.8, linewidth=1)
                self._ax2D.plot(traj[:, 0], traj[:, 1], 
                             color=self.config.trajectory_color, 
                             alpha=0.8, linewidth=1)
            
            # Label
            if self.config.show_labels:
                self._ax3D.text(pos[0], pos[1], pos[2]+5, name, fontsize=8)
                self._ax2D.text(pos[0], pos[1]+5, name, fontsize=8)
                
    def _draw_ues(
        self, 
        ue_positions: np.ndarray, 
        ue_names: Optional[List[str]] = None
    ):
        """Draw ground UEs as markers on the ground plane."""
        n_ues = len(ue_positions)
        names = ue_names or [f'UE-{i}' for i in range(n_ues)]
        
        for i, pos in enumerate(ue_positions):
            pos = np.asarray(pos).flatten()
            # UEs are on ground (z=0 or close to it)
            self._ax3D.scatter(
                pos[0], 
                pos[1], 
                pos[2] if pos.shape[0] > 2 else 0, 
                c=self.config.ue_color, 
                s=50, 
                marker=self.config.ue_marker, 
                edgecolors='black', 
                linewidths=0.5, 
                alpha=0.9
                )
            self._ax2D.scatter(
                pos[0], 
                pos[1], 
                c=self.config.ue_color, 
                s=50, 
                marker=self.config.ue_marker, 
                edgecolors='black', 
                linewidths=0.5, 
                alpha=0.9,
                label='UE' if i == 0 else None
                )
            if self.config.show_labels:
                self._ax3D.text(pos[0], pos[1], 2, names[i], fontsize=6)
                
    def _draw_base_stations(
        self, 
        bs_positions: np.ndarray, 
        bs_names: Optional[List[str]] = None
    ):
        """Draw base stations as tall cylindrical markers."""
        n_bs = len(bs_positions)
        names = bs_names or [f'BS-{i}' for i in range(n_bs)]
        
        for i, pos in enumerate(bs_positions):
            pos = np.asarray(pos).flatten()
            # Draw tall marker
            self._ax3D.scatter(
                pos[0], pos[1], pos[2], 
                c=self.config.base_station_color, 
                s=150, marker=self.config.base_station_marker, edgecolors='black', 
                linewidths=1.5)
            self._ax2D.scatter(
                pos[0], pos[1], 
                c=self.config.base_station_color, 
                s=150, marker=self.config.base_station_marker, edgecolors='black', 
                linewidths=1.5,
                label='BS' if i == 0 else None
                )
            # Draw vertical line to ground
            self._ax3D.plot([pos[0], pos[0]], [pos[1], pos[1]], [0, pos[2]], 
                         'g--', alpha=0.5, linewidth=1)
            if self.config.show_labels:
                self._ax3D.text(pos[0], pos[1], pos[2]+8, names[i], fontsize=8)
                
    def _draw_links(
        self, 
        links: List[Dict]
    ):
        """Draw communication links between entities."""
        for link in links:
            src = np.asarray(link['source'])
            dst = np.asarray(link['target'])
            los = link.get('los', True)
            
            # color = '#00FF00' if los else '#FF0000'  # Green for LoS, Red for NLoS
            color = self.config.link_los_color if los else self.config.link_nlos_color
            style = '-' if los else '--'
            alpha = 0.3 if los else 0.15
            
            self._ax3D.plot([src[0], dst[0]], [src[1], dst[1]], [src[2], dst[2]],
                         color=color, linestyle=style, alpha=alpha, linewidth=0.8)
            self._ax2D.plot([src[0], dst[0]], [src[1], dst[1]],
                         color=color, linestyle=style, alpha=alpha, linewidth=0.8)
    
    def _draw_ground_plane(
        self, 
        volume_size: List[float]
    ):
        """Draw semi-transparent ground plane."""
        xx, yy = np.meshgrid(
            np.linspace(-volume_size[0]/2, volume_size[0]/2, 2),
            np.linspace(-volume_size[1]/2, volume_size[1]/2, 2)
        )
        zz = np.zeros_like(xx)
        self._ax3D.plot_surface(xx, yy, zz, alpha=0.1, color='gray')
    
    def _building_faces(
        self,
        building_faces: List[List[float]]
    ):
        """Draw building faces as Poly3DCollection in 3D projection."""
        poly3d = self._Poly3DCollection(
            building_faces, 
            alpha=self.config.building_alpha,
            facecolor=self.config.building_color,
            edgecolor='black', 
            linewidth=0.5)
        self._ax3D.add_collection3d(poly3d)
    
    def _draw_collisions(
        self,
        collisions: List[np.ndarray]
    ):
        """Draw UAV collision points as red markers."""
        arr = np.array(collisions)
        if arr.shape[0] > 0:
            self._ax3D.scatter(
                arr[:, 0], arr[:, 1], arr[:, 2], 
                c='red', 
                s=250, 
                marker='X', 
                edgecolors='black', 
                linewidths=1.5, 
                alpha=0.9)
            self._ax2D.scatter(
                arr[:, 0], arr[:, 1], 
                c='red', 
                s=250, 
                marker='X', 
                edgecolors='black', 
                linewidths=1.5, 
                alpha=0.9,
                label='Collision' #if len(collisions) > 0 else None
                )
        else:
            return
        
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
        if self._fig is not None:
            # import matplotlib.pyplot as plt
            # plt.close(self._fig)
            # self._plt.close(self._fig)
            self._plt.close('all')
            self._fig = None
            self._ax3D = None
            self._ax2D = None
        if self._fig is None:
            self._init_plot(volume_size)
        
        # check if environment has been reset, if so, clear trajectories
        if 'current_frame' in state and state['current_frame'] == 0:
            self._trajectories.clear()
            self._frame_count = 0
        
        # Draw ground plane
        # self._draw_ground_plane(volume_size)
        
        # Draw heatmap if provided
        if 'heatmap' in state:
            self._draw_heatmap(state['heatmap'], volume_size)
        
        # Draw buildings
        if 'building_faces' in state:
            self._building_faces(state['building_faces'])
        elif 'buildings' in state:
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
            
        if 'collisions' in state:
            self._draw_collisions(state['collisions'])
            
        # Set title with frame info
        if 'title' in state:
            title = state['title'] #+ f' - Frame {self._frame_count}'
        else:
            title = f'UrbanMARL 3D Environment - Frame {self._frame_count}'
        self._fig.suptitle(title)
        
        self._ax2D.legend(loc="upper right")
        self._ax3D.grid(visible=False)
        
        self._frame_count += 1
        
        if mode == 'rgb_array':
            self._fig.tight_layout()
            self._fig.canvas.draw()
            # Get RGB buffer
            buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
            return buf[..., :3]  # Return RGB only
        elif mode == 'human':
            return self._fig
            
    def close(self):
        """Clean up renderer resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax3D = None
            self._ax2D = None
        self._trajectories.clear()