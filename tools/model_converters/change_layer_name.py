import argparse
import tempfile

import torch
from mmcv import Config
from mmcv.runner import load_state_dict

from mmdet3d_plugin.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D upgrade model version(before v0.6.0) of VoteNet')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--src_name', '-s', help='module name before changing')
    parser.add_argument('--dst_name', '-d', help='module name after changing')
    parser.add_argument('--out', '-o', help='path of the output checkpoint file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    assert 'state_dict' in checkpoint
    orig_ckpt = checkpoint['state_dict']
    converted_ckpt = orig_ckpt.copy()

    # Rename keys with specific prefix
    RENAME_KEYS = dict()
    for old_key in converted_ckpt.keys():
        if args.src_name in old_key:
            new_key = old_key.replace(args.src_name, args.dst_name)
            RENAME_KEYS[new_key] = old_key
    if len(RENAME_KEYS) == 0:
        print('No changed.')
    for new_key, old_key in RENAME_KEYS.items():
        print('Changed layer name from %s into %s' % (old_key, new_key))
        converted_ckpt[new_key] = converted_ckpt.pop(old_key)

    # Save checkpoint
    checkpoint['state_dict'] = converted_ckpt
    torch.save(checkpoint, args.out)


if __name__ == '__main__':
    main()
