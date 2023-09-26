import argparse
import logging
import sys

caffe_root = '/media/manu/kingstop/workspace/caffe/python'
sys.path.insert(0, caffe_root)

from modelComparator import compareCaffeAndCaffe


def run(args):
    compareCaffeAndCaffe(args.path_in_ptt_a, args.path_in_cfm_a, args.path_in_ptt_b, args.path_in_cfm_b)


def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


def parse_ars():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_in_ptt_a', default='/home/manu/tmp/bhv.prototxt', type=str)
    parser.add_argument('--path_in_ptt_b', default='/home/manu/tmp/bhv_eft.prototxt', type=str)
    parser.add_argument('--path_in_cfm_a', default='/home/manu/tmp/bhv.caffemodel', type=str)
    parser.add_argument('--path_in_cfm_b', default='/home/manu/tmp/bhv.caffemodel', type=str)
    return parser.parse_args()


def main():
    set_logging()
    args = parse_ars()
    logging.info(args)
    run(args)


if __name__ == '__main__':
    main()
