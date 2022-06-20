import argparse
import io as sysio
import os


CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone',
    'barrier']
ErrNames = [
    'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='dir path storing eval_result.txt')
    args = parser.parse_args()
    return args


def print_str(value, *arg, sstream=None):
    if sstream is None:
        sstream = sysio.StringIO()
    sstream.truncate(0)
    sstream.seek(0)
    print(value, *arg, file=sstream)
    return sstream.getvalue()


def main():
    args = parse_args()

    input_file = os.path.join(args.result_dir, 'eval_result.txt')
    output_file = os.path.join(args.result_dir, 'eval_result_summary.txt')

    eval_results = {}
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            sep = line.strip().split()
            eval_type = sep[0].split('/')[0]
            breakdown, val = sep[0].split('/')[1].replace(':', ''), float(sep[1])
            eval_results[breakdown] = val

    result = ''
    result += print_str(f"Evaluating bboxes of {eval_type}")
    result += print_str(f"mAP: {eval_results['mAP']:.4f}")
    for err_name in ErrNames:
        result += print_str(f"{err_name}: {eval_results[err_name]:.4f}")
    result += print_str(f"NDS: {eval_results['NDS']:.4f}\n")
    result += print_str(f"Per-class results:")
    result += print_str(f"Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE")
    for cls_name in CLASSES:
        ap = 0.0
        for key, val in eval_results.items():
            if cls_name + '_AP_dist' in key:
                ap += val
        ap /= 4
        ate = eval_results[cls_name + '_trans_err']
        ase = eval_results[cls_name + '_scale_err']
        aoe = eval_results[cls_name + '_orient_err']
        ave = eval_results[cls_name + '_vel_err']
        aae = eval_results[cls_name + '_attr_err']
        result += print_str(
            f"{cls_name}\t{ap:.3f}\t{ate:.3f}\t{ase:.3f}\t{aoe:.3f}\t{ave:.3f}\t{aae:.3f}"
        )

    print(result)
    with open(output_file, 'w') as fw:
        fw.write(result)


if __name__ == '__main__':
    main()
