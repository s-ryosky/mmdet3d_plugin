import argparse
from os import path as osp
import pickle
import tqdm

import numpy as np
from pyquaternion import Quaternion

import mmcv
from nuscenes.nuscenes import NuScenes

from nuscenes_converter_seg import create_nuscenes_infos


def nuscenes_data_prep(root_path,
                       key_prefix,
                       info_prefix,
                       version,
                       num_prev=5,
                       num_sweeps=5):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        key_prefix (str): The prefix of key info filenames.
        info_prefix (str): The prefix of output info filenames.
        version (str): Dataset version.
        num_prev (int, optional): Number of previous key frames.
            Default: 5
        max_sweeps (int, optional): Number of sweep frames between two key frames.
            Default: 5
    """
    create_sweep_infos(
        root_path, key_prefix, info_prefix, version=version, num_prev=num_prev, num_sweeps=num_sweeps)

    if version == 'v1.0-trainval':
        create_nuscenes_infos(root_path, info_prefix, max_sweeps=num_prev)


def create_sweep_infos(root_path,
                       key_prefix,
                       info_prefix,
                       version,
                       num_prev,
                       num_sweeps):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    if 'test' in version:
        nusc_infos = _fill_trainval_infos(
            nusc, root_path, key_prefix, 'test', num_prev=num_prev, num_sweeps=num_sweeps)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(nusc_infos, info_path)
    else:
        nusc_infos = _fill_trainval_infos(
            nusc, root_path, key_prefix, 'train', num_prev=num_prev, num_sweeps=num_sweeps)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(nusc_infos, info_path)
        nusc_infos = _fill_trainval_infos(
            nusc, root_path, key_prefix, 'val', num_prev=num_prev, num_sweeps=num_sweeps)
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(nusc_infos, info_val_path)


def _fill_trainval_infos(nusc, root_path, key_prefix, split, num_prev=5, num_sweeps=5):
    sensors = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    key_infos = pickle.load(open(osp.join(root_path,'{}_infos_{}.pkl'.format(key_prefix, split)), 'rb'))
    for current_id in tqdm.tqdm(range(len(key_infos['infos']))):
        # parameters of current key frame
        e2g_t = key_infos['infos'][current_id]['ego2global_translation']
        e2g_r = key_infos['infos'][current_id]['ego2global_rotation']
        l2e_t = key_infos['infos'][current_id]['lidar2ego_translation']
        l2e_r = key_infos['infos'][current_id]['lidar2ego_rotation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sample = nusc.get('sample', key_infos['infos'][current_id]['token'])
        current_cams = dict()
        for cam in sensors:
            current_cams[cam] = nusc.get('sample_data', sample['data'][cam])

        sweep_lists = []
        for i in range(num_prev):
            if sample['prev'] == '':
                break
            # add sweep frame between two key frame
            for j in range(num_sweeps):
                sweep_cams = dict()
                for cam in sensors:
                    if current_cams[cam]['prev'] == '':
                        sweep_cams = sweep_lists[-1]
                        break
                    sample_data = nusc.get('sample_data', current_cams[cam]['prev'])
                    sweep_cam = add_frame(
                        nusc, root_path, sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat)
                    current_cams[cam] = sample_data
                    sweep_cams[cam] = sweep_cam
                sweep_lists.append(sweep_cams)
            # add previous key frame
            sample = nusc.get('sample', sample['prev'])
            sweep_cams = dict()
            for cam in sensors:
                sample_data = nusc.get('sample_data', sample['data'][cam])
                sweep_cam = add_frame(
                    nusc, root_path, sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat)
                current_cams[cam] = sample_data
                sweep_cams[cam] = sweep_cam
            sweep_lists.append(sweep_cams)
        key_infos['infos'][current_id]['sweeps'] = sweep_lists
    return key_infos


def add_frame(nusc, data_root, sample_data, e2g_t, l2e_t, l2e_r_mat, e2g_r_mat):
    sweep_cam = dict()
    sweep_cam['is_key_frame'] = sample_data['is_key_frame']
    sweep_cam['data_path'] = osp.join(data_root, sample_data['filename'])
    sweep_cam['type'] = 'camera'
    sweep_cam['timestamp'] = sample_data['timestamp']
    sweep_cam['sample_data_token'] = sample_data['sample_token']
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    calibrated_sensor_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    sweep_cam['ego2global_translation']  = pose_record['translation']
    sweep_cam['ego2global_rotation']  = pose_record['rotation']
    sweep_cam['sensor2ego_translation']  = calibrated_sensor_record['translation']
    sweep_cam['sensor2ego_rotation']  = calibrated_sensor_record['rotation']
    sweep_cam['cam_intrinsic'] = calibrated_sensor_record['camera_intrinsic']

    l2e_r_s = sweep_cam['sensor2ego_rotation']
    l2e_t_s = sweep_cam['sensor2ego_translation']
    e2g_r_s = sweep_cam['ego2global_rotation']
    e2g_t_s = sweep_cam['ego2global_translation']

    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                    ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep_cam['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep_cam['sensor2lidar_translation'] = T

    lidar2cam_r = np.linalg.inv(sweep_cam['sensor2lidar_rotation'])
    lidar2cam_t = sweep_cam['sensor2lidar_translation'] @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    intrinsic = np.array(sweep_cam['cam_intrinsic'])
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
    sweep_cam['intrinsics'] = viewpad.astype(np.float32)
    sweep_cam['extrinsics'] = lidar2cam_rt.astype(np.float32)
    sweep_cam['lidar2img'] = lidar2img_rt.astype(np.float32)

    pop_keys = ['ego2global_translation', 'ego2global_rotation', 'sensor2ego_translation', 'sensor2ego_rotation', 'cam_intrinsic']
    [sweep_cam.pop(k) for k in pop_keys]

    return sweep_cam


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version')
parser.add_argument(
    '--num-prev',
    type=int,
    default=5,
    required=False,
    help='specify the number of previous key frames')
parser.add_argument(
    '--num-sweeps',
    type=int,
    default=5,
    required=False,
    help='specify the number of sweep frames between two key frames')
parser.add_argument('--extra-tag-key', type=str, default='nuscenes')
parser.add_argument('--extra-tag', type=str, default='nuscenes_petr')
args = parser.parse_args()

if __name__ == '__main__':
    if args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            key_prefix=args.extra_tag_key,
            info_prefix=args.extra_tag,
            version=train_version,
            num_prev=args.num_prev,
            num_sweeps=args.num_sweeps)
        test_version = f'{args.version}-test'
        nuscenes_data_prep(
            root_path=args.root_path,
            key_prefix=args.extra_tag_key,
            info_prefix=args.extra_tag,
            version=test_version,
            num_prev=args.num_prev,
            num_sweeps=args.num_sweeps)
    else:
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            key_prefix=args.extra_tag_key,
            info_prefix=args.extra_tag,
            version=train_version,
            num_prev=args.num_prev,
            num_sweeps=args.num_sweeps)
