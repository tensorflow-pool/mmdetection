import os
from argparse import ArgumentParser

from pycocotools.coco import COCO


def main():
    parser = ArgumentParser(description='COCO rsync')
    parser.add_argument('--ann', help='annotation file path', default=os.path.expanduser("~/datasets/coco_detectron/annotations/instances_val2017.json"))
    parser.add_argument('--img_dir', help='', default=os.path.expanduser("~/datasets/coco_detectron/images/val2017"))
    args = parser.parse_args()

    coco = COCO(args.ann)
    print("ann len {}".format(len(coco.imgs)))
    # coco.download(args.img_dir)


if __name__ == '__main__':
    main()
