import argparse

from mmcv import Config

from mmdet3d_plugin.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D_Plugin visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved'
    )
    parser.add_argument('--online', action='store_true', help='Whether to visualize online')
    parser.add_argument('--out_format', type=str, default='video', choices=['image', 'video'])
    parser.add_argument(
        '--skip_pipeline', type=str, nargs='+', default=['Normalize'],
        help='skip some useless pipeline'
    )
    parser.add_argument(
        '--viewpoint_path', type=str, help='Path to file of Open3D viewpoint params (.json)'
    )
    parser.add_argument(
        '--draw_points_in_imgs', action='store_true', help='Whether to visualize point cloud drawn on the images'
    )
    parser.add_argument(
        '--print_timestamp', action='store_true', help='Whether to print timestamp for each data'
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # build the dataset
    if cfg.data.test.type == "NuScenesDataset":
        cfg.data.test.type = "NuScenesVisDataset"
    else:
        raise ValueError("Specified dataset is not supported.")
    dataset = build_dataset(cfg.data.test)

    # data loading pipeline for showing
    eval_pipeline = cfg.get('eval_pipeline', {})
    eval_pipeline = [x for x in eval_pipeline if x['type'] not in args.skip_pipeline]
    dataset.show_img(
        args.show_dir,
        online=args.online,
        out_format=args.out_format,
        pipeline=eval_pipeline if eval_pipeline else None,
        viewpoint_path=args.viewpoint_path,
        draw_points_in_imgs=args.draw_points_in_imgs,
        print_timestamp=args.print_timestamp,
    )


if __name__ == '__main__':
    main()
