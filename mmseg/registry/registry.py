# Copyright (c) OpenMMLab. All rights reserved.
"""MMSegmentation provides 21 registry nodes to support using modules across
projects. Each node is a child of the root registry in MMEngine.

More details can be found at
https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html.
"""

from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import INFERENCERS as MMENGINE_INFERENCERS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=MMENGINE_RUNNERS) #管理不同类型的运行器，如基于迭代的运行器（IterBasedRunner）和基于周期的运行器（EpochBasedRunner）。这些运行器负责控制训练过程的整体流程。

# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=MMENGINE_RUNNER_CONSTRUCTORS) #管理用于初始化运行器的构造函数。这些构造函数定义了如何根据配置创建运行器实例。

# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=MMENGINE_LOOPS) #管理不同类型的循环，如训练循环（EpochBasedTrainLoop）。这些循环定义了训练过程中的具体迭代逻辑。

# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['mmseg.engine.hooks']) #管理各种钩子，如检查点钩子（CheckpointHook）。这些钩子在训练过程中的特定时间点执行，以支持如日志记录、模型保存、验证等额外功能。

# manage data-related modules
  #DATASETS、DATA_SAMPLERS、TRANSFORMS：分别管理数据集、数据采样器和数据变换。这些数据相关的模块是训练过程中数据处理的关键部分，包括数据的加载、预处理和增强。
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['mmseg.datasets'])
DATA_SAMPLERS = Registry('data sampler', parent=MMENGINE_DATA_SAMPLERS)
TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['mmseg.datasets.transforms'])

# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry('model', parent=MMENGINE_MODELS, locations=['mmseg.models']) #管理所有继承自nn.Module的模型。这些模型定义了用于训练任务的神经网络架构

# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['mmseg.models']) #管理模型包装器，如MMDistributedDataParallel。这些包装器提供了额外的功能，如模型并行和分布式训练支持

# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['mmseg.models']) #管理权重初始化模块，如Uniform。这些模块定义了模型参数在训练开始前的初始值。

#OPTIMIZERS、OPTIM_WRAPPERS、OPTIM_WRAPPER_CONSTRUCTORS：分别管理优化器、优化器包装器以及优化器包装器构造函数。这些组件负责在训练过程中更新模型的参数。
# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    locations=['mmseg.engine.optimizers'])
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim_wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    locations=['mmseg.engine.optimizers'])
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    locations=['mmseg.engine.optimizers'])

# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    locations=['mmseg.engine.schedulers']) #管理参数调度器，如MultiStepLR。这些调度器负责在训练过程中根据预定策略调整学习率等超参数。

# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['mmseg.evaluation']) #这是一个用于管理各种度量指标的注册表。度量指标用于评估模型性能，如准确率、召回率等。
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['mmseg.evaluation']) #评估器用于在训练或测试过程中评估模型性能。

# manage task-specific modules like ohem pixel sampler
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['mmseg.models']) #这个注册表用于管理特定于任务的工具或实用程序，例如在线难例挖掘（OHEM）像素采样器，这在图像分割任务中可能很有用。

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['mmseg.visualization']) #可视化器用于将模型预测或中间结果以图形化的方式展示出来，帮助开发者理解模型行为。

# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['mmseg.visualization']) #可视化后端指的是实现可视化功能的底层库或框架，如matplotlib、PIL等。

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    locations=['mmseg.visualization']) #日志处理器用于处理和分析训练或测试过程中的日志信息，提取有用的统计数据或进行可视化。

# manage inferencer
INFERENCERS = Registry('inferencer', parent=MMENGINE_INFERENCERS) #推理器用于在模型训练完成后，对新的数据进行预测或推理。
