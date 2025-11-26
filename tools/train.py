# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')  #这是一个位置参数，必须提供，用于指定训练配置文件的路径
    parser.add_argument('--work-dir', help='the dir to save logs and models') #这是一个可选参数，用于指定保存日志和模型的目录
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically') #这是一个标志（flag）参数，当设置为 True 时，表示自动从 work_dir 中最新的检查点（checkpoint）恢复训练。默认值为 False
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training') #：这是一个标志参数，用于启用自动混合精度（automatic-mixed-precision）训练。默认值为 False
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.') #这是一个可以接收多个值的参数，用于覆盖配置文件中的一些设置
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher') #这是一个带有选择项的参数，用于指定作业启动器，可选值包括 'none'、'pytorch'、'slurm' 和 'mpi'。默认值为 'none'
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0) #这是两个指向同一个参数的别名，用于接收分布式训练时的本地排名（local rank）
    args = parser.parse_args() #使用 parser.parse_args() 解析命令行输入的参数，并将结果存储在 args 变量中
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():

    args = parse_args() #通过调用parse_args()函数来获取用户指定的配置文件路径、工作目录、是否恢复训练、是否启用自动混合精度训练等参数

    # load config     #使用Config.fromfile(args.config)从用户指定的配置文件中加载配置信息到cfg变量中。然后，根据命令行参数更新配置，比如设置作业启动器cfg.launcher = args.launcher，以及如果提供了--cfg-options，则将这些选项合并到配置中
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('F:/E_sanduao/work_dirs',
                                '202508002_deeplabv3plus_消融_基金' + osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
      #如果用户指定了--amp选项，则检查当前配置中优化器包装器的类型。如果已经是AmpOptimWrapper（表示已经启用了AMP），则打印警告信息；
      # 如果不是OptimWrapper，则断言失败（因为AMP目前只支持OptimWrapper），并将优化器包装器类型更改为AmpOptimWrapper，并设置损失缩放为动态。
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training  根据命令行参数args.resume设置配置中的cfg.resume，以决定是否从最近的检查点恢复训练
    cfg.resume = args.resume

    # build the runner from config
      #根据配置中的runner_type（如果有的话）从注册表中构建自定义的Runner，或者如果没有指定runner_type，则构建默认的Runner。
      #Runner是负责管理训练过程的对象，包括模型的构建、数据的加载、训练的循环等。
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
