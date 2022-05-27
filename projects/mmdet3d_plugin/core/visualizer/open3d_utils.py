from os import path as osp

import numpy as np
import torch

try:
    import open3d as o3d
    from open3d import geometry
except ImportError:
    raise ImportError(
        'Please run "pip install open3d" to install open3d first.')

from mmdet3d.core.visualizer.open3d_vis import Visualizer as Open3dVisualizer
from mmdet3d.core.visualizer.open3d_vis import _draw_points


def _draw_bboxes(bbox3d,
                 vis,
                 points_colors,
                 pcd=None,
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz'):
    """Draw bbox on visualizer and change the color of points inside bbox3d.

    Args:
        bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
            3d bbox (x, y, z, x_size, y_size, z_size, yaw) to visualize.
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        points_colors (numpy.array): color of each points.
        pcd (:obj:`open3d.geometry.PointCloud`, optional): point cloud.
            Default: None.
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points inside bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
    """
    if isinstance(bbox3d, torch.Tensor):
        bbox3d = bbox3d.cpu().numpy()
    bbox3d = bbox3d.copy()

    in_box_color = np.array(points_in_box_color)
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        box3d.color = bbox_color

        line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
        line_set.paint_uniform_color(bbox_color)
        # draw bboxes on visualizer
        vis.add_geometry(line_set)

        # change the color of points which are in box
        if pcd is not None and mode == 'xyz':
            indices = box3d.get_point_indices_within_bounding_box(pcd.points)
            points_colors[indices] = in_box_color

    # update points colors
    if pcd is not None:
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        vis.update_geometry(pcd)


class Visualizer(Open3dVisualizer):
    r"""Online visualizer implemented with Open3d.

    Args:
        points (numpy.array, shape=[N, 3+C]): Points to visualize. The Points
            cloud is in mode of Coord3DMode.DEPTH (please refer to
            core.structures.coord_3d_mode).
        bbox3d (numpy.array, shape=[M, 7], optional): 3D bbox
            (x, y, z, x_size, y_size, z_size, yaw) to visualize.
            The 3D bbox is in mode of Box3DMode.DEPTH with
            gravity_center (please refer to core.structures.box_3d_mode).
            Default: None.
        points_size (int, optional): the size of points to show on visualizer.
            Default: 2.
        point_color (tuple[float], optional): the color of points.
            Default: (0.5, 0.5, 0.5).
        bbox_color (tuple[float], optional): the color of bbox.
            Default: (0, 1, 0).
        points_in_box_color (tuple[float], optional):
            the color of points which are in bbox3d. Default: (1, 0, 0).
        rot_axis (int, optional): rotation axis of bbox. Default: 2.
        center_mode (bool, optional): indicate the center of bbox is
            bottom center or gravity center. available mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
        mode (str, optional):  indicate type of the input points,
            available mode ['xyz', 'xyzrgb']. Default: 'xyz'.
        window_heigh (int): the height of the window.
        window_width (int): the width of the window.
        viewpoint_path (str): the path to save and load viewpoint parameters.
        online (bool): whether to visualize the results online.
            Default: False.
    """

    def __init__(self,
                 points,
                 bbox3d=None,
                 points_size=2,
                 line_width=2,
                 point_color=(0.5, 0.5, 0.5),
                 background_color=(1, 1, 1),
                 bbox_color=(0, 1, 0),
                 points_in_box_color=(1, 0, 0),
                 rot_axis=2,
                 center_mode='lidar_bottom',
                 mode='xyz',
                 window_height=1080,
                 window_width=1920,
                 viewpoint_path=None,
                 online=False):

        # init visualizer
        self.o3d_visualizer = o3d.visualization.Visualizer()
        self.o3d_visualizer.create_window(width=window_width, height=window_height, visible=online)
        self.ctr = self.o3d_visualizer.get_view_control()
        mesh_frame = geometry.TriangleMesh.create_coordinate_frame(
            size=1, origin=[0, 0, 0])  # create coordinate frame
        self.o3d_visualizer.add_geometry(mesh_frame)

        # Set renderer
        opt = self.o3d_visualizer.get_render_option()
        opt.background_color = np.asarray(background_color)
        opt.line_width = line_width

        self.points_size = points_size
        self.point_color = point_color
        self.bbox_color = bbox_color
        self.points_in_box_color = points_in_box_color
        self.rot_axis = rot_axis
        self.center_mode = center_mode
        self.mode = mode
        self.viewpoint_path = viewpoint_path
        self.online = online
        self.pcd = None
        self.seg_num = 0

        # draw points
        if points is not None:
            self.pcd, self.points_colors = _draw_points(
                points, self.o3d_visualizer, points_size, point_color, mode
            )

        # draw boxes
        if bbox3d is not None:
            _draw_bboxes(
                bbox3d, self.o3d_visualizer, self.points_colors,
                self.pcd, bbox_color, points_in_box_color, rot_axis,
                center_mode, mode
            )

    def reset_points(self, points):
        self.o3d_visualizer.remove_geometry(self.pcd)
        self.pcd, self.points_colors = _draw_points(
            points, self.o3d_visualizer, self.points_size, self.point_color, self.mode
        )

    def add_bboxes(self, bbox3d, bbox_color=None, points_in_box_color=None):
        """Add bounding box to visualizer.

        Args:
            bbox3d (numpy.array, shape=[M, 7]):
                3D bbox (x, y, z, x_size, y_size, z_size, yaw)
                to be visualized. The 3d bbox is in mode of
                Box3DMode.DEPTH with gravity_center (please refer to
                core.structures.box_3d_mode).
            bbox_color (tuple[float]): the color of bbox. Default: None.
            points_in_box_color (tuple[float]): the color of points which
                are in bbox3d. Default: None.
        """
        if bbox_color is None:
            bbox_color = self.bbox_color
        if points_in_box_color is None:
            points_in_box_color = self.points_in_box_color
        _draw_bboxes(bbox3d, self.o3d_visualizer, self.points_colors, self.pcd,
                     bbox_color, points_in_box_color, self.rot_axis,
                     self.center_mode, self.mode)

    def show(self, save_viewpoint=True):
        """Visualize the points cloud.

        Args:
            save_path (str, optional): path to save image. Default: None.
        """

        # set viewpoint
        if self.viewpoint_path is not None and osp.exists(self.viewpoint_path):
            viewpoint_param = o3d.io.read_pinhole_camera_parameters(self.viewpoint_path)
            self.ctr.convert_from_pinhole_camera_parameters(viewpoint_param)

        if self.online:
            self.o3d_visualizer.run()
        else:
            self.o3d_visualizer.poll_events()

        img = self.o3d_visualizer.capture_screen_float_buffer(do_render=True)
        img = (np.asarray(img)*255).astype(np.uint8)

        if self.online:
            if save_viewpoint and self.viewpoint_path is not None:
                viewpoint_param = self.ctr.convert_to_pinhole_camera_parameters()
                o3d.io.write_pinhole_camera_parameters(self.viewpoint_path, viewpoint_param)
            self.o3d_visualizer.destroy_window()
            self.o3d_visualizer.close()
            del self.o3d_visualizer
        else:
            self.o3d_visualizer.clear_geometries()

        return img
