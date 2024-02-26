import argparse
from detector import Detector
from os import path, getcwd


def parse_argument():
    parser = argparse.ArgumentParser(description='This script provide object detection applications.')
    parser.add_argument('--source', type=str, default='0', help='input mp4file, rtsp url, camera')
    parser.add_argument('--app_type', type=str, default='detection', help='type of application(detection, linecross, roi, segmentation)')
    parser.add_argument('--save', action='store_true', help='save detection to file')
    parser.add_argument('--output', type=str, default=path.join(getcwd(), 'output.mp4'), help='file path to save output')
    parser.add_argument('--view', action='store_true', help='show streaming window.')
    args = parser.parse_args()
    # print(vars(args))
    return args


def main():
    args = parse_argument()
    if args.app_type == 'detection':
        object_detector = Detector()
        Detector.view_detection(args.source)
    else:
        print(f'application type ({args.app_type}) haven\'t support')


if __name__ == '__main__':
    main()

# 'https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=11470'
