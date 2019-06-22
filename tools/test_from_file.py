import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import get_dataset


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    # parser.add_argument('--config', help='test config file path', default="configs/fast_mask_rcnn_r101_fpn_1x.py")
    # parser.add_argument('--checkpoint', help='checkpoint file', default="checkpoints/mask_rcnn_r101_fpn_1x_20181129-34ad1961.pth")

    parser.add_argument('--config', help='train config file path',
                        default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py")))
    parser.add_argument('--checkpoint', help='checkpoint file',
                        default="https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu_20190516-9f253a93.pth")

    # parser.add_argument('--config', help='train config file path',
    #                     default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../configs/retinanet_r50_fpn_1x.py")))
    # parser.add_argument('--checkpoint', help='checkpoint file',
    #                     default="https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth")

    parser.add_argument('--out', help='output result file',
                        default=os.path.expanduser(os.path.join(os.path.dirname(__file__), "../outs/result.pkl")))
    parser.add_argument(
        '--eval',
        type=str,
        default=["bbox"],
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
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

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.test)
    outputs = mmcv.load(args.out)
    eval_types = args.eval
    if eval_types:
        print('Starting evaluate {}'.format(' and '.join(eval_types)))
        if eval_types == ['proposal_fast']:
            result_file = args.out
            coco_eval(result_file, eval_types, dataset.coco)
        else:
            if not isinstance(outputs[0], dict):
                result_files = results2json(dataset, outputs, args.out)
                coco_eval(result_files, eval_types, dataset.coco)
            else:
                for name in outputs[0]:
                    print('\nEvaluating {}'.format(name))
                    outputs_ = [out[name] for out in outputs]
                    result_file = args.out + '.{}'.format(name)
                    result_files = results2json(dataset, outputs_,
                                                result_file)
                    coco_eval(result_files, eval_types, dataset.coco)


if __name__ == '__main__':
    main()
