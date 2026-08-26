"""UrbanMARL 3D Environment Renderer.

Provides high-performance, vectorized 3D/2D visualization for urban multi-agent
reinforcement learning environments including buildings, UAVs, ground UEs,
base stations, and radio communication links.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    from matplotlib.collections import LineCollection
    
except ImportError:
    raise ImportError(
        "Matplotlib is required for rendering. Install it using `pip install matplotlib`."
    )
    

@dataclass
class UrbanRenderConfig:
    """Configuration settings for urban environment rendering.

    Attributes:
        figsize (Tuple[int, int]): Figure dimensions in pixels (width, height).
        dpi (int): Dots per inch for figure rendering.
        heatmap_color (str): Matplotlib colormap name for 2D top-view heatmap.
        heatmap_alpha (float): Transparency alpha for heatmap.
        building_color (str): Hex color code for building cuboids.
        building_alpha (float): Transparency alpha for buildings.
        uav_color (str): Hex color code for UAV markers and indicators.
        uav_marker (str): Matplotlib marker style for UAVs.
        ue_color (str): Hex color code for User Equipment (UE) markers.
        ue_marker (str): Matplotlib marker style for UEs.
        base_station_color (str): Hex color code for Base Station (BS) markers.
        base_station_marker (str): Matplotlib marker style for Base Stations.
        trajectory_color (str): Hex color code for UAV flight path trajectories.
        show_trajectory (bool): Whether to display historical trajectories.
        trajectory_length (int): Maximum number of historical points per trajectory.
        link_los_color (str): Line color for Line-of-Sight (LoS) links.
        link_nlos_color (str): Line color for Non-Line-of-Sight (NLoS) links.
        camera_elev (float): Initial 3D camera elevation angle in degrees.
        camera_azim (float): Initial 3D camera azimuth angle in degrees.
        show_labels (bool): Whether to display text labels on entities.
    """

    figsize: Tuple[int, int] = (1280, 820)
    dpi: int = 100
    heatmap_color: str = 'gist_yarg'
    heatmap_alpha: float = 0.7
    building_color: str = '#C0C0C0'
    building_alpha: float = 0.2
    uav_color: str = '#FF4444'
    uav_marker: str = "^"
    ue_color: str = '#4444FF'
    ue_marker: str = "."
    base_station_color: str = '#FF4444'
    base_station_marker: str = "v"
    trajectory_color: str = '#FFA500'
    show_trajectory: bool = True
    trajectory_length: int = 100
    link_los_color: str = '#00FF00'
    link_nlos_color: str = '#FF0000'
    camera_elev: float = 45.0
    camera_azim: float = -120.0
    show_labels: bool = True


class Urban3DRenderer:
    """Vectorized 3D & 2D renderer for UrbanMARL environments.

    Renders:
    - 3D buildings as cuboid collections.
    - UAVs as 3D/2D flying markers with historical flight trajectories.
    - Ground User Equipments (UEs) on top-view and 3D maps.
    - Base Stations (BS) with vertical ground-projection indicators.
    - Vectorized communication links classified by LoS/NLoS state.

    Attributes:
        config (UrbanRenderConfig): Renderer configuration parameters.
    """

    def __init__(self, config: Optional[UrbanRenderConfig] = None) -> None:
        """Initializes the Urban3DRenderer.

        Args:
            config (Optional[UrbanRenderConfig]): Rendering configuration.
                If None, uses default settings.
        """
        self.config = config or UrbanRenderConfig()
        self._fig = None
        self._ax3D = None
        self._ax2D = None
        self._trajectories: Dict[str, List[np.ndarray]] = {}
        self._frame_count = 0
        self._plt = None
        self._Poly3DCollection = None
        self._Line3DCollection = None

    def _init_plot(self, volume_size: List[float]) -> None:
        """Initializes matplotlib 3D and 2D plots.

        Args:
            volume_size (List[float]): Environment volume dimensions [x, y, z].

        Raises:
            ImportError: If matplotlib is not installed.
        """
        self._plt = plt
        self._Poly3DCollection = Poly3DCollection

        fig = plt.figure(
            figsize=(
                self.config.figsize[0] / self.config.dpi,
                self.config.figsize[1] / self.config.dpi,
            ),
            dpi=self.config.dpi,
        )
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.set_title('Urban Environment 3D View')
        ax3d.set_xlim(-volume_size[0] / 2, volume_size[0] / 2)
        ax3d.set_ylim(-volume_size[1] / 2, volume_size[1] / 2)
        ax3d.set_zlim(0, volume_size[2])
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.view_init(
            elev=self.config.camera_elev, 
            azim=self.config.camera_azim
        )

        ax2d = fig.add_subplot(122)
        ax2d.set_title('Urban Environment Heatmap (Top View)')
        ax2d.set_xlim(-volume_size[0] / 2, volume_size[0] / 2)
        ax2d.set_ylim(-volume_size[1] / 2, volume_size[1] / 2)
        ax2d.set_xlabel('X (m)')
        ax2d.set_ylabel('Y (m)')

        self._fig = fig
        self._ax3D = ax3d
        self._ax2D = ax2d

    def _setup_axes_limits(self, volume_size: List[float]) -> None:
        """Sets title, limits, labels, and camera perspective on persistent axes.

        Args:
            volume_size (List[float]): Environment volume dimensions [x, y, z].
        """
        self._ax3D.set_title('Urban Environment 3D View')
        self._ax3D.set_xlim(-volume_size[0] / 2, volume_size[0] / 2)
        self._ax3D.set_ylim(-volume_size[1] / 2, volume_size[1] / 2)
        self._ax3D.set_zlim(0, volume_size[2])
        self._ax3D.set_xlabel('X (m)')
        self._ax3D.set_ylabel('Y (m)')
        self._ax3D.set_zlabel('Z (m)')
        self._ax3D.view_init(
            elev=self.config.camera_elev, azim=self.config.camera_azim
        )

        self._ax2D.set_title('Urban Environment Heatmap (Top View)')
        self._ax2D.set_xlim(-volume_size[0] / 2, volume_size[0] / 2)
        self._ax2D.set_ylim(-volume_size[1] / 2, volume_size[1] / 2)
        self._ax2D.set_xlabel('X (m)')
        self._ax2D.set_ylabel('Y (m)')

    def _draw_heatmap(
        self, heatmap: Union[np.ndarray, torch.Tensor], volume_size: List[float]
    ) -> None:
        """Draws a top-view heatmap on the 2D subplot.

        Args:
            heatmap (Union[np.ndarray, torch.Tensor]): 2D grid matrix of signal/coverage values.
            volume_size (List[float]): Environment volume size [x, y, z].
        """
        if self._ax2D is None:
            return
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.detach().cpu().numpy()
        else:
            heatmap_np = np.asarray(heatmap)

        extent = [
            -volume_size[0] / 2,
            volume_size[0] / 2,
            -volume_size[1] / 2,
            volume_size[1] / 2,
        ]
        tpc = self._ax2D.imshow(
            heatmap_np.T,
            extent=extent,
            origin='lower',
            cmap=self.config.heatmap_color,
            alpha=self.config.heatmap_alpha,
        )
        self._fig.colorbar(
            tpc,
            orientation='horizontal',
            ax=self._ax2D,
            fraction=0.046,
            pad=0.04,
        )

    def _draw_buildings(self, buildings: List[Dict]) -> None:
        """Draws buildings as 3D cuboid Poly3DCollections in a single batch.

        Args:
            buildings (List[Dict]): List of building dictionary objects, each with
                keys 'position' [x, y, z] and 'size' [w, l, h].
        """
        all_faces = []
        for b in buildings:
            x, y, z = b['position']
            w, l, h = b['size']

            vertices = np.array([
                [x, y, z],
                [x + w, y, z],
                [x + w, y + l, z],
                [x, y + l, z],
                [x, y, z + h],
                [x + w, y, z + h],
                [x + w, y + l, z + h],
                [x, y + l, z + h],
            ])

            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[3], vertices[0], vertices[4], vertices[7]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
            ]
            all_faces.extend(faces)

        if all_faces:
            poly3d = self._Poly3DCollection(
                all_faces,
                alpha=self.config.building_alpha,
                facecolor=self.config.building_color,
                edgecolor='black',
                linewidth=0.5,
            )
            self._ax3D.add_collection3d(poly3d)

    def _draw_uavs(
        self,
        uav_positions: Union[np.ndarray, torch.Tensor],
        uav_names: Optional[List[str]] = None,
    ) -> None:
        """Draws UAV markers and historical flight trajectories in a vectorized manner.

        Args:
            uav_positions (Union[np.ndarray, torch.Tensor]): UAV 3D coordinates array (N, 3).
            uav_names (Optional[List[str]]): Optional list of UAV display names.
        """
        if isinstance(uav_positions, torch.Tensor):
            uav_positions = uav_positions.detach().cpu().numpy()
        uav_positions = np.asarray(uav_positions)
        if uav_positions.size == 0:
            return
        if uav_positions.ndim == 1:
            uav_positions = uav_positions.reshape(1, -1)

        n_uavs = len(uav_positions)
        names = uav_names or [f'UAV-{i}' for i in range(n_uavs)]

        for i in range(n_uavs):
            pos = uav_positions[i].flatten()
            name = names[i]
            if name not in self._trajectories:
                self._trajectories[name] = []
            if (
                len(self._trajectories[name]) == 0
                or not np.array_equal(pos, self._trajectories[name][-1])
            ):
                self._trajectories[name].append(pos.copy())
            if len(self._trajectories[name]) > self.config.trajectory_length:
                self._trajectories[name].pop(0)

        # Vectorized scatter rendering
        self._ax3D.scatter(
            uav_positions[:, 0],
            uav_positions[:, 1],
            uav_positions[:, 2],
            c=self.config.uav_color,
            s=100,
            marker=self.config.uav_marker,
            edgecolors='black',
            linewidths=1,
            alpha=0.9,
        )
        self._ax2D.scatter(
            uav_positions[:, 0],
            uav_positions[:, 1],
            c=self.config.uav_color,
            s=100,
            marker=self.config.uav_marker,
            edgecolors='black',
            linewidths=1,
            alpha=0.9,
            label='UAV',
        )

        if self.config.show_trajectory:
            for name in names:
                if len(self._trajectories[name]) > 1:
                    traj = np.array(self._trajectories[name])
                    self._ax3D.plot(
                        traj[:, 0],
                        traj[:, 1],
                        traj[:, 2],
                        color=self.config.trajectory_color,
                        alpha=0.8,
                        linewidth=1,
                    )
                    self._ax2D.plot(
                        traj[:, 0],
                        traj[:, 1],
                        color=self.config.trajectory_color,
                        alpha=0.8,
                        linewidth=1,
                    )

        if self.config.show_labels:
            for i, name in enumerate(names):
                pos = uav_positions[i]
                self._ax3D.text(pos[0], pos[1], pos[2] + 5, name, fontsize=8)
                self._ax2D.text(pos[0], pos[1] + 5, name, fontsize=8)

    def _draw_ues(
        self,
        ue_positions: Union[np.ndarray, torch.Tensor],
        ue_names: Optional[List[str]] = None,
    ) -> None:
        """Draws ground UEs as markers on 3D and 2D ground planes in a vectorized manner.

        Args:
            ue_positions (Union[np.ndarray, torch.Tensor]): Ground UE locations (N, 2) or (N, 3).
            ue_names (Optional[List[str]]): Optional list of UE names.
        """
        if isinstance(ue_positions, torch.Tensor):
            ue_positions = ue_positions.detach().cpu().numpy()
        ue_positions = np.asarray(ue_positions)
        if ue_positions.size == 0:
            return
        if ue_positions.ndim == 1:
            ue_positions = ue_positions.reshape(1, -1)

        n_ues = len(ue_positions)
        names = ue_names or [f'UE-{i}' for i in range(n_ues)]
        z_vals = (
            ue_positions[:, 2]
            if ue_positions.shape[1] > 2
            else np.zeros(n_ues)
        )

        self._ax3D.scatter(
            ue_positions[:, 0],
            ue_positions[:, 1],
            z_vals,
            c=self.config.ue_color,
            s=50,
            marker=self.config.ue_marker,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.9,
        )
        self._ax2D.scatter(
            ue_positions[:, 0],
            ue_positions[:, 1],
            c=self.config.ue_color,
            s=50,
            marker=self.config.ue_marker,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.9,
            label='UE',
        )

        if self.config.show_labels:
            for i, name in enumerate(names):
                self._ax3D.text(
                    ue_positions[i, 0],
                    ue_positions[i, 1],
                    z_vals[i] + 2,
                    name,
                    fontsize=6,
                )

    def _draw_base_stations(
        self,
        bs_positions: Union[np.ndarray, torch.Tensor],
        bs_names: Optional[List[str]] = None,
    ) -> None:
        """Draws base station towers and vertical ground lines in a vectorized manner.

        Args:
            bs_positions (Union[np.ndarray, torch.Tensor]): Base station positions (N, 3).
            bs_names (Optional[List[str]]): Optional list of base station names.
        """
        if isinstance(bs_positions, torch.Tensor):
            bs_positions = bs_positions.detach().cpu().numpy()
        bs_positions = np.asarray(bs_positions)
        if bs_positions.size == 0:
            return
        if bs_positions.ndim == 1:
            bs_positions = bs_positions.reshape(1, -1)

        n_bs = len(bs_positions)
        names = bs_names or [f'BS-{i}' for i in range(n_bs)]

        self._ax3D.scatter(
            bs_positions[:, 0],
            bs_positions[:, 1],
            bs_positions[:, 2],
            c=self.config.base_station_color,
            s=150,
            marker=self.config.base_station_marker,
            edgecolors='black',
            linewidths=1.5,
        )
        self._ax2D.scatter(
            bs_positions[:, 0],
            bs_positions[:, 1],
            c=self.config.base_station_color,
            s=150,
            marker=self.config.base_station_marker,
            edgecolors='black',
            linewidths=1.5,
            label='BS',
        )

        segments = [
            [[pos[0], pos[1], 0.0], [pos[0], pos[1], pos[2]]]
            for pos in bs_positions
        ]
        line_coll = Line3DCollection(
            segments, colors='g', linestyles='--', alpha=0.5, linewidths=1
        )
        self._ax3D.add_collection3d(line_coll)

        if self.config.show_labels:
            for i, name in enumerate(names):
                pos = bs_positions[i]
                self._ax3D.text(
                    pos[0], pos[1], pos[2] + 8, name, fontsize=8
                )

    def _draw_links(self, links: List[Dict]) -> None:
        """Draws communication links using batched line collections.

        Args:
            links (List[Dict]): List of links with keys 'source', 'target', and optional 'los'.
        """
        if not links:
            return

        los_3d, los_2d = [], []
        nlos_3d, nlos_2d = [], []

        for link in links:
            src = np.asarray(link['source'])
            dst = np.asarray(link['target'])
            is_los = link.get('los', True)
            if is_los:
                los_3d.append([
                    [src[0], src[1], src[2]],
                    [dst[0], dst[1], dst[2]],
                ])
                los_2d.append([[src[0], src[1]], [dst[0], dst[1]]])
            else:
                nlos_3d.append([
                    [src[0], src[1], src[2]],
                    [dst[0], dst[1], dst[2]],
                ])
                nlos_2d.append([[src[0], src[1]], [dst[0], dst[1]]])

        if los_3d:
            coll_3d = Line3DCollection(
                los_3d,
                colors=self.config.link_los_color,
                linestyles='-',
                alpha=0.3,
                linewidths=0.8,
            )
            self._ax3D.add_collection3d(coll_3d)
            coll_2d = LineCollection(
                los_2d,
                colors=self.config.link_los_color,
                linestyles='-',
                alpha=0.3,
                linewidths=0.8,
            )
            self._ax2D.add_collection(coll_2d)

        if nlos_3d:
            coll_3d = Line3DCollection(
                nlos_3d,
                colors=self.config.link_nlos_color,
                linestyles='--',
                alpha=0.15,
                linewidths=0.8,
            )
            self._ax3D.add_collection3d(coll_3d)
            coll_2d = LineCollection(
                nlos_2d,
                colors=self.config.link_nlos_color,
                linestyles='--',
                alpha=0.15,
                linewidths=0.8,
            )
            self._ax2D.add_collection(coll_2d)

    def _building_faces(self, building_faces: List[List[float]]) -> None:
        """Draws building faces as Poly3DCollection in 3D projection.

        Args:
            building_faces (List[List[float]]): List of polygon vertices for building faces.
        """
        poly3d = self._Poly3DCollection(
            building_faces,
            alpha=self.config.building_alpha,
            facecolor=self.config.building_color,
            edgecolor='black',
            linewidth=0.5,
        )
        self._ax3D.add_collection3d(poly3d)

    def _draw_collisions(
        self, collisions: Union[List[np.ndarray], np.ndarray]
    ) -> None:
        """Draws UAV collision points as red cross markers.

        Args:
            collisions (Union[List[np.ndarray], np.ndarray]): Array of collision positions.
        """
        arr = np.asarray(collisions)
        if arr.size == 0 or arr.ndim < 2 or arr.shape[0] == 0:
            return

        self._ax3D.scatter(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            c='red',
            s=250,
            marker='X',
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9,
        )
        self._ax2D.scatter(
            arr[:, 0],
            arr[:, 1],
            c='red',
            s=250,
            marker='X',
            edgecolors='black',
            linewidths=1.5,
            alpha=0.9,
            label='Collision',
        )

    def render(
        self, state: Dict, mode: str = 'rgb_array'
    ) -> Optional[Union[np.ndarray, object]]:
        """Renders the urban environment.

        Args:
            state (Dict): Environment state payload containing:
                - 'volume_size': [x, y, z] environment bounds.
                - 'buildings': list of building dicts.
                - 'uav_positions': (N, 3) UAV locations.
                - 'ue_positions': (M, 3) UE locations.
                - 'base_station_positions': (K, 3) BS locations.
                - 'links': optional list of link dicts.
                - 'heatmap': optional 2D tensor/array for heatmap overlay.
                - 'collisions': optional array of collision points.
            mode (str): Rendering mode: 'rgb_array' returns (H, W, 3) uint8 image array,
                'human' returns the Matplotlib figure handle. Defaults to 'rgb_array'.

        Returns:
            Optional[Union[np.ndarray, object]]: RGB image numpy array if mode=='rgb_array',
                or matplotlib Figure handle if mode=='human'.
        """
        volume_size = state.get('volume_size', [500, 500, 200])

        if self._fig is not None:
            self._plt.close('all')
            self._fig = None
            self._ax3D = None
            self._ax2D = None
        
        if self._fig is None:
            self._init_plot(volume_size)
        
        if 'current_frame' in state and state['current_frame'] == 0:
            self._trajectories.clear()
            self._frame_count = 0

        if 'heatmap' in state:
            self._draw_heatmap(state['heatmap'], volume_size)

        if 'building_faces' in state:
            self._building_faces(state['building_faces'])
        elif 'buildings' in state:
            self._draw_buildings(state['buildings'])

        if 'base_station_positions' in state:
            self._draw_base_stations(state['base_station_positions'])

        if 'ue_positions' in state:
            self._draw_ues(state['ue_positions'])

        if 'uav_positions' in state:
            self._draw_uavs(state['uav_positions'])

        if 'links' in state:
            self._draw_links(state['links'])

        if 'collisions' in state:
            self._draw_collisions(state['collisions'])

        title = state.get(
            'title', f'UrbanMARL 3D Environment - Frame {self._frame_count}'
        )
        self._fig.suptitle(title)

        self._ax2D.legend(loc="upper right")
        self._ax3D.grid(visible=False)

        self._frame_count += 1

        if mode == 'rgb_array':
            self._fig.tight_layout()
            self._fig.canvas.draw()
            buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(self._fig.canvas.get_width_height()[::-1] + (4,))
            return buf[..., :3]
        elif mode == 'human':
            return self._fig

    def close(self) -> None:
        """Cleans up renderer resources and closes figures."""
        if self._fig is not None and self._plt is not None:
            self._plt.close(self._fig)
            self._fig = None
            self._ax3D = None
            self._ax2D = None
        self._trajectories.clear()