import copy

import numpy as np
import torch

from mmdet3d.core.bbox import points_cam2img


def lidar_corners3d_to_img(corners3d, lidar2img_rt):
    """Project the corners of 3D bbox on 2D plane

    Args:
        corners3d (numpy.array, shape=[N, 8, 3]):
            The numpy array of 3d corners in lidar coordinate system
        lidar2img_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
    Returns:
        numpy.array, shape=[N, 8, 2]
    """
    if isinstance(corners3d, torch.Tensor):
        corners3d = corners3d.cpu().numpy()
    num_bbox = corners3d.shape[0]
    pts_4d = np.concatenate(
        [corners3d.reshape(-1, 3),
        np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2img_rt = copy.deepcopy(lidar2img_rt).reshape(4, 4)
    if isinstance(lidar2img_rt, torch.Tensor):
        lidar2img_rt = lidar2img_rt.cpu().numpy()
    pts_2d = pts_4d @ lidar2img_rt.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return imgfov_pts_2d


def lidar_corners3d_to_cam(corners3d, lidar2cam_rt):
    """Convert the coordinate sysmte of 3D bbox corners from lidar to camera

    Args:
        corners3d (numpy.array, shape=[N, 8, 3]):
            The numpy array of 3d corners in lidar coordinate system
        lidar2cam_rt (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera extrinsic parameters.
    Returns:
        numpy.array, shape=[N, 8, 3]
    """
    if isinstance(corners3d, torch.Tensor):
        corners3d = corners3d.cpu().numpy()
    num_bbox = corners3d.shape[0]
    pts_4d = np.concatenate(
        [corners3d.reshape(-1, 3),
        np.ones((num_bbox * 8, 1))], axis=-1
    )
    lidar2cam_rt = copy.deepcopy(lidar2cam_rt).reshape(4, 4)
    if isinstance(lidar2cam_rt, torch.Tensor):
        lidar2cam_rt = lidar2cam_rt.cpu().numpy()
    pts_3d = pts_4d @ lidar2cam_rt

    return pts_3d[..., :3].reshape(num_bbox, 8, 3)


def cam_corners3d_to_img(corners3d, cam2img):
    """Project the corners of 3D bbox on 2D plane

    Args:
        corners3d (torch.Tensor | numpy.array, shape=[N, 8, 3]):
            The numpy array of 3d corners in camera coordinate system
        cam2img (dict): Camera intrinsic matrix,
            denoted as `K` in depth bbox coordinate system.
    Returns:
        numpy.array, shape=[N, 8, 2]
    """
    cam2img = copy.deepcopy(cam2img)
    num_bbox = corners3d.shape[0]
    points_3d = corners3d.reshape(-1, 3)
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.from_numpy(np.array(cam2img))

    assert (cam2img.shape == torch.Size([3, 3])
            or cam2img.shape == torch.Size([4, 4]))
    cam2img = cam2img.float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam2img)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2)
    if isinstance(imgfov_pts_2d, torch.Tensor):
        imgfov_pts_2d = imgfov_pts_2d.numpy()

    return imgfov_pts_2d


def check_box3d_in_image(box_corners, box_corners_in_image, imsize):
    """Check if 3D bbox is in an image

    Args:
        box_corners (numpy.array, shape=[N, 8, 3]):
            The numpy array of 3d corners in camera coordinate system
        box_corners_in_image (numpy.array, shape=[N, 8, 2]):
            The numpy array of 3d corners in image coordinate system
        imsize (tuple[int]): The height and width of an image.
    """
    if isinstance(box_corners, torch.Tensor):
        box_corners = box_corners.cpu().numpy()
    if isinstance(box_corners_in_image, torch.Tensor):
        box_corners_in_image = box_corners_in_image.cpu().numpy()
    imsize = imsize[::-1]  # (height, width) -> (width, height)
    visible = np.logical_and(
        box_corners_in_image[:, :, 0] >= 0,
        box_corners_in_image[:, :, 0] < imsize[0]
    )
    visible = np.logical_and(visible, box_corners_in_image[:, :, 1] >= 0)
    visible = np.logical_and(visible, box_corners_in_image[:, :, 1] < imsize[1])

    # True if all corners are at least 0.1 meter in front of the camera.
    in_front = box_corners[:, :, 2] > 0.1

    return np.logical_and(np.any(visible, axis=1), np.all(in_front, axis=1))
