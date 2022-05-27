import copy
from os import path as osp

import cv2
import math
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
import pyquaternion
import torch

from mmdet3d.core import bbox3d2result, box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.core.bbox import (Box3DMode, Coord3DMode,
                               CameraInstance3DBoxes, LiDARInstance3DBoxes)
from mmdet3d.core.visualizer.image_vis import draw_camera_bbox3d_on_img, draw_lidar_bbox3d_on_img
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.nuscenes_mono_dataset import (output_to_nusc_box,
                                                    cam_nusc_box_to_global,
                                                    nusc_box_to_cam_box3d)

from mmdet3d_plugin.core.visualizer.common_utils import get_color
from mmdet3d_plugin.core.visualizer.image_utils import (lidar_corners3d_to_img,
                                                        lidar_corners3d_to_cam,
                                                        cam_corners3d_to_img,
                                                        check_box3d_in_image)
from mmdet3d_plugin.core.visualizer.open3d_utils import Visualizer


@DATASETS.register_module()
class NuScenesVisDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This class serves as the API for visualization on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        version (str, optional): Dataset version. Defaults to 'v1.0-trainval'.
        results (list[dict]): List of bounding boxes results.
    """
    CAMERA_TYPES = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    CAM_NUM = len(CAMERA_TYPES)

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 version='v1.0-trainval',
                 results=None):
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_root=data_root,
            classes=classes,
            load_interval=load_interval,
            with_velocity=with_velocity,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            eval_version=eval_version,
            use_valid_flag=use_valid_flag)

        self.version = version
        self.results = self.preprocess(results)

        # overwrite modality
        self.modality = dict(
            use_camera=True,
            use_lidar=True,
            use_radar=False,
            use_map=False,
            use_external=False,
        )

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = data['infos']
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def preprocess(self, results=None):
        if results is not None:
            assert self.load_interval == 1
            if len(results) == len(self.data_infos):
                for result, data_info in zip(results, self.data_infos):
                    result['timestamp'] = data_info['timestamp']
            elif len(results) == len(self.data_infos) * __class__.CAM_NUM:
                for i, data_info in enumerate(self.data_infos):
                    for j, cam in enumerate(__class__.CAMERA_TYPES):
                        result = results[i * __class__.CAM_NUM + j]
                        result['timestamp'] = data_info['cams'][cam]['timestamp']
            else:
                raise ValueError("The number of results is invalid.")
            results = list(sorted(results, key=lambda e: e['timestamp']))
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
        # self.data_infos = self.data_infos[::self.load_interval]
        return results

    def import_nusc(self):
        if not hasattr(self, "nusc"):
            self.nusc = NuScenes(
                version=self.version, dataroot=str(self.data_root), verbose=False
            )
        if not hasattr(self, "scene_token_list"):
            self.scene_token_list = [scene["token"] for scene in self.nusc.scene]
            self.scene_name_list = [scene["name"] for scene in self.nusc.scene]

    def get_scene_name(self, token):
        sample = self.nusc.get("sample", token)
        scene_token = sample["scene_token"]
        scene_index = self.scene_token_list.index(scene_token)
        return self.scene_name_list[scene_index]

    def show(
        self,
        out_dir,
        online=False,
        out_format='video',
        pipeline=None,
        show_gt=False,
        nms=True,
        score_th=0.1,
        viewpoint_path=None,
        canvas_type='v2',
    ):
        """Visualization.

        Args:
            out_dir (str): Output directory of visualization result.
            online (bool): Whether to visualize the results online.
                Default: False.
            out_format (str): Output file format (image or video).
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
            show_gt (bool): Whether to visualize GT bboxes.
            score_th (float): score threshold.
            viewpoint_path (str): Output path of viewpoint parameters.
            canvas_type (str): Type of visualization format (v1 or v2).
        """
        mapped_class_names = self.CLASSES
        box_origin = (0.5, 0.5, 0)
        gt_color = (255, 0, 0)  # Red
        if out_format == 'video':
            assert not online
            self.import_nusc()
        video_frequency = 5
        if canvas_type == 'v1':
            window_height=960
            window_width=1280
        elif canvas_type == 'v2':
            window_height=800
            window_width=600
        else:
            raise NotImplementedError()

        assert out_dir is not None, 'Expect out_dir, got none.'
        mmcv.mkdir_or_exist(out_dir)
        mmcv.mkdir_or_exist(osp.dirname(viewpoint_path))

        vis = None
        pipeline = self._get_pipeline(pipeline)
        prev_scene_name, out = None, None
        progress_bar = mmcv.ProgressBar(len(self.data_infos))
        for i, data_info in enumerate(self.data_infos):
            # load GT bboxes
            if show_gt:
                gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor
                gt_lidar_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes,
                    box_dim=gt_bboxes.shape[-1],
                    origin=box_origin
                )

            # load bboxes predicted from lidar
            pred_lidar_bboxes, pred_bboxes, pred_labels = None, None, None
            if self.results is not None and len(self.results) == len(self.data_infos):
                result = self.results[i]
                if 'pts_bbox' in result.keys():
                    result = result['pts_bbox']
                    # remove unconfidensed bboxes
                    inds = result['scores_3d'] > score_th
                    pred_lidar_bboxes = result['boxes_3d'][inds, :]
                    pred_labels = result['labels_3d'][inds]
                    if len(pred_lidar_bboxes) > 0:
                        pred_bboxes = pred_lidar_bboxes.tensor.numpy()
                        pred_labels = pred_labels.numpy()

            # load bboxes predicted from images and identify the type of camera
            boxes_per_frame = []
            attrs_per_frame = []
            imgs = {}
            cam_infos = {}
            for j in range(__class__.CAM_NUM):
                if self.results is not None and \
                        len(self.results) == len(self.data_infos) * __class__.CAM_NUM:
                    result = self.results[i * __class__.CAM_NUM + j]
                    timestamp = result['timestamp']
                    for cam in __class__.CAMERA_TYPES:
                        if data_info['cams'][cam]['timestamp'] == timestamp:
                            break
                    if 'img_bbox' in result.keys():
                        result = result['img_bbox']
                        boxes, attrs = output_to_nusc_box(result)
                        boxes, attrs = cam_nusc_box_to_global(
                            data_info['cams'][cam],
                            boxes, attrs,
                            mapped_class_names,
                            self.eval_detection_configs,
                            self.eval_version
                        )
                        boxes_per_frame.extend(boxes)
                        attrs_per_frame.extend(attrs)
                else:
                    cam = __class__.CAMERA_TYPES[j]

                # load an image
                cam_info = data_info['cams'][cam]
                img = mmcv.imread(cam_info['data_path'])  # (900, 1600, 3)

                # store images and infos
                imgs[cam] = img
                cam_infos[cam] = cam_info

            # postprocess for predicted bboxes
            if len(boxes_per_frame) > 0:
                if nms:
                    # transform the coordinate sysmte from global to front camera
                    boxes_per_frame, attrs_per_frame = global_nusc_box_to_cam(
                        cam_infos['CAM_FRONT'],
                        boxes_per_frame,
                        attrs_per_frame,
                        mapped_class_names,
                        self.eval_detection_configs,
                        self.eval_version,
                        filter=False,
                    )

                    # run nms to remove redundant predictions caused by overlap of images
                    det = nusc_nms(boxes_per_frame, attrs_per_frame)

                    # restore the coordinate system from front camera to global
                    boxes_per_frame, attrs_per_frame = output_to_nusc_box(det)
                    boxes_per_frame, attrs_per_frame = cam_nusc_box_to_global(
                        cam_infos['CAM_FRONT'],
                        boxes_per_frame,
                        attrs_per_frame,
                        mapped_class_names,
                        self.eval_detection_configs,
                        self.eval_version,
                        filter=False,
                    )

                # remove unconfidensed bboxes
                attrs_per_frame = [a for a, b in zip(attrs_per_frame, boxes_per_frame) if b.score > score_th]
                boxes_per_frame = [b for b in boxes_per_frame if b.score > score_th]

                # convert coordinate system from global to lidar
                lidar_bboxes = global_nusc_box_to_lidar(
                    data_info,
                    copy.deepcopy(boxes_per_frame),
                    mapped_class_names,
                    self.eval_detection_configs,
                    self.eval_version,
                    filter=False,
                )
                lidar_bboxes, _, pred_labels = nusc_box_to_lidar_box3d(lidar_bboxes)
                pred_bboxes = lidar_bboxes.tensor.numpy()
                pred_labels = pred_labels.numpy()

            for cam in __class__.CAMERA_TYPES:
                img = imgs[cam]
                cam_info = cam_infos[cam]

                # convert coordinate system of predicted bboxes from global to camera
                pred_cam_boxes3d = None
                if len(boxes_per_frame) > 0:
                    boxes, _ = global_nusc_box_to_cam(
                        cam_info,
                        copy.deepcopy(boxes_per_frame),
                        attrs_per_frame,
                        mapped_class_names,
                        self.eval_detection_configs,
                        self.eval_version,
                    )
                    cam_boxes3d, _, _ = nusc_box_to_cam_box3d(boxes)
                    boxes3d = cam_boxes3d.tensor.cpu()
                    pred_cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)

                # obtain lidar to image transformation matrix
                lidar2cam_rt, intrinsic, lidar2img_rt = get_transform_matrix(cam_info)

                # draw GT bboxes on an image
                if show_gt and len(gt_lidar_bboxes) > 0:
                    corners_3d = gt_lidar_bboxes.corners.numpy()
                    corners_3d_cam = lidar_corners3d_to_cam(corners_3d, lidar2cam_rt)
                    box_corners_in_image = lidar_corners3d_to_img(corners_3d, lidar2img_rt)
                    visible = check_box3d_in_image(corners_3d_cam, box_corners_in_image, img.shape[:2])
                    visible_indx = np.where(visible==True)[0]
                    visible_gt_lidar_bboxes = gt_lidar_bboxes[visible_indx, ...]
                    if len(visible_gt_lidar_bboxes) > 0:
                        img = draw_lidar_bbox3d_on_img(
                            visible_gt_lidar_bboxes, img, lidar2img_rt, None,
                            color=gt_color[::-1], thickness=2,
                        )

                # draw predicted bboxes on an image
                if self.results is not None:
                    tmp_bboxes = None
                    if pred_lidar_bboxes is not None and len(pred_lidar_bboxes) > 0:
                        draw_bbox = draw_lidar_bbox3d_on_img
                        corners_3d = pred_lidar_bboxes.corners.numpy()
                        corners_3d_cam = lidar_corners3d_to_cam(corners_3d, lidar2cam_rt)
                        box_corners_in_image = lidar_corners3d_to_img(corners_3d, lidar2img_rt)
                        tmp_bboxes = pred_lidar_bboxes
                        trans_mat = lidar2img_rt
                    elif pred_cam_boxes3d is not None and len(pred_cam_boxes3d) > 0:
                        draw_bbox = draw_camera_bbox3d_on_img
                        corners_3d_cam = pred_cam_boxes3d.corners.numpy()
                        box_corners_in_image = cam_corners3d_to_img(corners_3d_cam, intrinsic)
                        tmp_bboxes = pred_cam_boxes3d
                        trans_mat = intrinsic
                    if tmp_bboxes is not None and len(tmp_bboxes) > 0:
                        visible = check_box3d_in_image(corners_3d_cam, box_corners_in_image, img.shape[:2])
                        visible_indx = np.where(visible==True)[0]
                        visible_pred_bboxes = tmp_bboxes[visible_indx, ...]
                        visible_pred_labels = pred_labels[visible_indx]
                        for k in np.unique(visible_pred_labels):
                            cls_indx = np.where(visible_pred_labels == k)[0]
                            selected_pred_bboxes = visible_pred_bboxes[cls_indx, ...]
                            if len(selected_pred_bboxes) > 0:
                                img = draw_bbox(
                                    selected_pred_bboxes, img, trans_mat, None,
                                    color=get_color(k)[::-1], thickness=2,
                                )
                imgs[cam] = img

            # prepare lidar point clouds
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            points = Coord3DMode.convert_point(
                points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
            )

            # visualize bboxes on lidar point clouds
            if online or vis is None:
                del vis
                vis = Visualizer(
                    points,
                    points_size=1,
                    line_width=1,
                    point_color=(1, 1, 1),
                    background_color=(0.25, 0.25, 0.25),
                    points_in_box_color=(1, 1, 1),
                    window_height=window_height,
                    window_width=window_width,
                    viewpoint_path=viewpoint_path,
                    online=online,
                )
            else:
                vis.reset_points(points)
            if show_gt and len(gt_bboxes) > 0:
                show_gt_bboxes = Box3DMode.convert(
                    gt_bboxes.numpy(), Box3DMode.LIDAR, Box3DMode.DEPTH
                )
                vis.add_bboxes(
                    bbox3d=show_gt_bboxes,
                    bbox_color=np.asarray(gt_color, dtype=np.float64) / 255,
                    points_in_box_color=None,
                )
            if pred_bboxes is not None:
                show_pred_bboxes = Box3DMode.convert(
                    pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH
                )
                if pred_labels is None:
                    vis.add_bboxes(bbox3d=show_pred_bboxes)
                else:
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        k = int(pred_labels[j])
                        if labelDict.get(k) is None:
                            labelDict[k] = []
                        labelDict[k].append(show_pred_bboxes[j])
                    for k in labelDict:
                        vis.add_bboxes(
                            bbox3d=np.array(labelDict[k]),
                            bbox_color=np.asarray(get_color(k), dtype=np.float64) / 255,
                            points_in_box_color=np.asarray(get_color(k), dtype=np.float64) / 255,
                        )
            pcd_img = vis.show()
            pcd_img = mmcv.image.rgb2bgr(pcd_img)

            # create canvas
            if canvas_type == 'v1':
                canvas = create_canvas(pcd_img, imgs)
            elif canvas_type == 'v2':
                canvas = create_canvas_v2(pcd_img, imgs)
            else:
                raise NotImplementedError()

            # output
            if out_format == 'image':
                result_path = osp.join(out_dir, f'{file_name}.png')
                mmcv.imwrite(canvas, result_path)
            elif out_format == 'video':
                scene_name = self.get_scene_name(data_info['token'])
                if scene_name != prev_scene_name:
                    if prev_scene_name is not None:
                        # close current video
                        print("\nSaved a video into %s" % output_path)
                        out.release()
                    # create new video
                    output_path = osp.join(out_dir, scene_name+".avi")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(output_path, fourcc, video_frequency,
                                            (canvas.shape[1], canvas.shape[0]))
                    prev_scene_name = scene_name
                out.write(canvas[:, :, ::-1])
            else:
                raise NotImplementedError()
            progress_bar.update()


def get_transform_matrix(cam_info):
    """Get transformation matrics of coordinate systems from
       calibration information.

    Args:
        cam_info (dict): Info for a specific camera sample data,
            including the calibration information.

    Returns:
        np.ndarray (4x4): transformation matrix from lidar coordinate to 
            camera coordinate.
        np.ndarray (3x3): transformation matrix from camera coordinate to 
            image coordinate.
        np.ndarray (4x4): transformation matrix from lidar coordinate to 
            image coordinate.
    """
    lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
    lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = cam_info['cam_intrinsic']
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    return lidar2cam_rt, intrinsic, lidar2img_rt


def cam_nusc_box_to_global(info,
                           boxes,
                           attrs,
                           classes,
                           eval_configs,
                           eval_version='detection_cvpr_2019',
                           filter=True):
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (list[int]): List of predicted attributes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
        list: List of attributes
    """
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['sensor2ego_rotation']))
        box.translate(np.array(info['sensor2ego_translation']))
        # filter det in ego.
        if filter:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
        attr_list.append(attr)
    return box_list, attr_list


def global_nusc_box_to_cam(info,
                           boxes,
                           attrs,
                           classes,
                           eval_configs,
                           eval_version='detection_cvpr_2019',
                           filter=True):
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (list[int]): List of predicted attributes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    attr_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.translate(-np.array(info['ego2global_translation']))
        box.rotate(
            pyquaternion.Quaternion(info['ego2global_rotation']).inverse)
        # filter det in ego.
        if filter:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to camera coord system
        box.translate(-np.array(info['sensor2ego_translation']))
        box.rotate(pyquaternion.Quaternion(info['sensor2ego_rotation']).inverse)
        box_list.append(box)
        if attrs is not None:
            attr_list.append(attrs[i])
    return box_list, attr_list


def global_nusc_box_to_lidar(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019',
                             filter=True):
    """Convert the box from global to lidar coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the lidar
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.translate(-np.array(info['ego2global_translation']))
        box.rotate(
            pyquaternion.Quaternion(info['ego2global_rotation']).inverse)
        # filter det in ego.
        if filter:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to camera coord system
        box.translate(-np.array(info['lidar2ego_translation']))
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_lidar_box3d(boxes):
    """Convert boxes from :obj:`NuScenesBox` to :obj:`LiDARInstance3DBoxes`.

    Args:
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.

    Returns:
        tuple (:obj:`LiDARInstance3DBoxes` | torch.Tensor | torch.Tensor):
            Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)

    # convert nusbox to lidarbox convention (wlh->lwh)
    dims = dims[:, [1, 0, 2]]

    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1)
    lidar_boxes3d = LiDARInstance3DBoxes(
        boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes])
    labels = torch.LongTensor([b.label for b in boxes])
    return lidar_boxes3d, scores, labels


def nusc_nms(boxes_per_frame, attrs_per_frame):
    cam_boxes3d, scores, labels = nusc_box_to_cam_box3d(boxes_per_frame)
    nms_cfg = dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_pre=4096,
        nms_thr=0.05,
        score_thr=0.01,
        min_bbox_size=0,
        max_per_frame=500)
    nms_cfg = mmcv.Config(nms_cfg)
    cam_boxes3d_for_nms = xywhr2xyxyr(cam_boxes3d.bev)
    boxes3d = cam_boxes3d.tensor
    attrs = labels.new_tensor([attr for attr in attrs_per_frame])
    boxes3d, scores, labels, attrs = box3d_multiclass_nms(
        boxes3d,
        cam_boxes3d_for_nms,
        scores,
        nms_cfg.score_thr,
        nms_cfg.max_per_frame,
        nms_cfg,
        mlvl_attr_scores=attrs)
    cam_boxes3d = CameraInstance3DBoxes(boxes3d, box_dim=9)
    det = bbox3d2result(cam_boxes3d, scores, labels, attrs)
    return det

def create_canvas(pcd_img, imgs):
    camera_list = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    ]
    pcd_img_h, pcd_img_w = pcd_img.shape[:2]
    new_width = math.floor(pcd_img_w / 3)
    margin = math.floor((pcd_img_w - 3 * new_width) / 2)
    for cam_chan in camera_list:
        img = imgs[cam_chan]
        img_h, img_w = img.shape[:2]
        img_scale = new_width / img_w
        img = mmcv.image.imresize(
            img, (int(img_w * img_scale), int(img_h * img_scale)),
            interpolation='bicubic', backend='cv2',
        )
        if 'BACK' in cam_chan:
            img = img[:, ::-1, :]
        img_h, img_w = img.shape[:2]
        if cam_chan == 'CAM_FRONT_LEFT':
            canvas = np.zeros((pcd_img_h + 2 * img_h, pcd_img_w, 3),
                            dtype=np.uint8)
            h_s, h_e, w_s, w_e = 0, img_h, 0, img_w
        elif cam_chan in ['CAM_FRONT', 'CAM_FRONT_RIGHT']:
            w_s = w_e + margin
            w_e = w_s + img_w
        elif cam_chan == 'CAM_BACK_LEFT':
            h_s, h_e, w_s, w_e = pcd_img_h + img_h, pcd_img_h + 2 * img_h, 0, img_w
        elif cam_chan in ['CAM_BACK', 'CAM_BACK_RIGHT']:
            w_s = w_e + margin
            w_e = w_s + img_w
        canvas[h_s:h_e, w_s:w_e, :] = img
    pcd_scale = float(canvas.shape[0] - 2 * img_h) / pcd_img_h
    resized_pcd_img = mmcv.image.imresize(
        pcd_img, (int(pcd_img_w * pcd_scale), int(pcd_img_h * pcd_scale)),
        interpolation='bicubic', backend='cv2',
    )
    pcd_img_h2, pcd_img_w2 = resized_pcd_img.shape[:2]
    w_start = (canvas.shape[1] - pcd_img_w2) // 2
    canvas[img_h:img_h+pcd_img_h2, w_start:w_start+pcd_img_w2, :] = resized_pcd_img
    return canvas


def create_canvas_v2(pcd_img, imgs):
    camera_list = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    ]
    max_width = 1920
    pcd_img_h, pcd_img_w = pcd_img.shape[:2]
    img_h, img_w = imgs[camera_list[0]].shape[:2]
    new_width = math.floor((max_width - pcd_img_w) / 3)
    if new_width > img_w:
        new_width = img_w
        img_scale = 1.0
    else:
        img_scale = new_width / img_w
    if int(img_h * img_scale) * 2 > pcd_img_h:
        new_height = math.floor(pcd_img_h / 2)
        img_scale = new_height / img_h
    new_height, new_width = int(img_h * img_scale), int(img_w * img_scale)
    canvas = np.zeros(
        (pcd_img_h, pcd_img_w + new_width * 3, 3), dtype=np.uint8
    )
    canvas[:pcd_img_h, :pcd_img_w] = pcd_img
    margin_h = canvas.shape[0] - new_height * 2
    margin_w = canvas.shape[1] - pcd_img_w - new_width * 3
    for cam_chan in camera_list:
        img = imgs[cam_chan]
        if img_scale != 1.0:
            img = mmcv.image.imresize(
                img, (new_width, new_height), interpolation='bicubic', backend='cv2',
            )
        if 'BACK' in cam_chan:
            img = img[:, ::-1, :]
        img_h, img_w = img.shape[:2]
        if cam_chan == 'CAM_FRONT_LEFT':
            h_s = int(margin_h / 2)
            w_s = pcd_img_w + int(margin_w / 2)
            h_e = h_s + img_h
            w_e = w_s + img_w
        elif cam_chan in ['CAM_FRONT', 'CAM_FRONT_RIGHT']:
            w_s = w_e
            w_e = w_s + img_w
        elif cam_chan == 'CAM_BACK_LEFT':
            h_s = h_e
            h_e = h_s + img_h
            w_s = pcd_img_w + int(margin_w / 2)
            w_e = w_s + img_w
        elif cam_chan in ['CAM_BACK', 'CAM_BACK_RIGHT']:
            w_s = w_e
            w_e = w_s + img_w
        canvas[h_s:h_e, w_s:w_e, :] = img
    return canvas
