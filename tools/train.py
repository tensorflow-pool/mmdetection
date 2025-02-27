from __future__ import division

import argparse
import os

import torch
from mmcv import Config
from mmdet import __version__
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.datasets import get_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')

    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py")))
    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/faster_rcnn_r50_caffe_c4_1x.py")))
    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/faster_rcnn_r50_fpn_1x.py")))

    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/mask_rcnn_r50_fpn_1x.py")))

    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/retinanet_r50_fpn_1x.py")))

    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/guided_anchoring/ga_retinanet_r50_caffe_fpn_1x.py")))
    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/guided_anchoring/ga_retinanet_x101_32x4d_fpn_1x.py")))

    parser.add_argument('--config', help='train config file path',
                        default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py")))

    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        default=True,
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_dataset = get_dataset(cfg.data.train)
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=train_dataset.CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = train_dataset.CLASSES
    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger)


if __name__ == '__main__':
    main()
