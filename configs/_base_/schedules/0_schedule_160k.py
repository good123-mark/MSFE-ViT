# optimizer  优化器
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
## type: 优化器类型,这里是随机梯度下降（SGD）。lr: 学习率,控制参数更新的步长。
# momentum: 动量因子,动量帮助加速SGD在相关方向上的收敛，并抑制震荡。weight_decay: 权重衰减，这是一种正则化方法，用于减少模型的过拟合。
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
#优化器包装器（OptimWrapper）,type: 指定包装器的类型，这里是OptimWrapper。它可能用于对优化器进行额外的封装，比如梯度裁剪。
# optimizer: 传入之前定义的优化器配置。lip_grad: 梯度裁剪的阈值，设置为None，意味着不进行梯度裁剪。梯度裁剪是一种防止梯度爆炸的技术。

# learning policy  学习率调度策略
param_scheduler = [
    dict(
        type='PolyLR',                           #指定学习率调度器的类型，这里是PolyLR，即多项式学习率衰减。
        eta_min=1e-4,                            # 学习率衰减到的最小值，设置为1e-4。
        power=0.9,                               #多项式衰减的幂指数，设置为0.9。
        begin=0,                                 #学习率调度开始的迭代次数，设置为0。
        end=160000,                              #学习率调度结束的迭代次数，设置为160000。
       # end=80000,
        by_epoch=False)                          #指定是否按epoch进行调度，这里设置为False，意味着按迭代次数进行调度。
]
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)            #指定训练循环的类型，这里是基于迭代次数的循环（IterBasedTrainLoop）。
val_cfg = dict(type='ValLoop')                                                  #验证配置（ValCfg）
test_cfg = dict(type='TestLoop')                                                #测试配置（TestCfg）
default_hooks = dict(                                                           #默认钩子
    timer=dict(type='IterTimerHook'),                                           #timer: 迭代计时钩子，用于记录每次迭代的耗时。
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),     #logger: 日志记录钩子，每50次迭代记录一次日志，不按epoch记录指标。
    param_scheduler=dict(type='ParamSchedulerHook'),                            #param_scheduler: 参数调度钩子，用于根据学习率调度策略更新学习率。
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),     #checkpoint: 检查点钩子，每16000次迭代保存一次模型，不按epoch保存。
    sampler_seed=dict(type='DistSamplerSeedHook'),                              #sampler_seed: 分布式采样种子钩子，用于设置分布式训练时的采样种子。
    visualization=dict(type='SegVisualizationHook'))                            #visualization: 可视化钩子，用于在训练过程中可视化分割结果。




