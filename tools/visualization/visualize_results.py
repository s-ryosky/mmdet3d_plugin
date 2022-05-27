import argparse

import mmcv
from mmcv import Config

from mmdet3d_plugin.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D_Plugin visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved'
    )
    parser.add_argument('--online', action='store_true', help='Whether to visualize online')
    parser.add_argument('--out_format', type=str, default='video', choices=['image', 'video'])
    parser.add_argument(
        '--show-gt', action='store_true', help='Whether to visualize GT boxes'
    )
    parser.add_argument(
        '--score_thresh', type=float, default=0.1,
        help='score threshold to visualize predictions'
    )
    parser.add_argument(
        '--skip_pipeline', type=str, nargs='+', default=['Normalize'],
        help='skip some useless pipeline'
    )
    parser.add_argument(
        '--viewpoint_path', type=str, help='Path to file of Open3D viewpoint params (.json)'
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    if args.result:
        results = mmcv.load(args.result)
    else:
        results = None

    # build the dataset
    if cfg.data.test.type == "NuScenesDataset":
        cfg.data.test.type = "NuScenesVisDataset"
    else:
        raise ValueError("Specified dataset is not supported.")
    dataset = build_dataset(cfg.data.test, dict(results=results))

    # data loading pipeline for showing
    eval_pipeline = cfg.get('eval_pipeline', {})
    eval_pipeline = [x for x in eval_pipeline if x['type'] not in args.skip_pipeline]
    if eval_pipeline:
        dataset.show(
            args.show_dir,
            online=args.online,
            out_format=args.out_format,
            pipeline=eval_pipeline,
            show_gt=args.show_gt,
            score_th=args.score_thresh,
            viewpoint_path=args.viewpoint_path,
        )
    else:
        dataset.show(
            args.show_dir,
            online=args.online,
            out_format=args.out_format,
            show_gt=args.show_gt,
            score_th=args.score_thresh,
            viewpoint_path=args.viewpoint_path,
        )  # use default pipeline


if __name__ == '__main__':
    main()
