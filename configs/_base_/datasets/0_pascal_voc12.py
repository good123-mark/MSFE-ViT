# dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),                                   #从文件系统中加载图像。这是数据预处理的第一步，它读取指定路径的图像文件。
    dict(type='LoadAnnotations'),                                     #加载与图像对应的标注信息。在图像分割任务中，这通常包括每个像素的类别标签，形成一个与图像相同大小的分割图
    dict(
        type='RandomResize',                                          #随机调整图像的大小。
        scale=(2048, 512),                                            #表示缩放尺寸的最小值和最大值
        ratio_range=(0.5, 2.0),                                       #指定了缩放比例的范围
        keep_ratio=True),                                             #表示在缩放时保持图像的宽高比
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), #随机裁剪图像和对应的标注。crop_size是裁剪后的大小，
    dict(type='RandomFlip', prob=0.5),                                #以一定的概率（prob=0.5）随机翻转图像和对应的标注。
    dict(type='PhotoMetricDistortion'),                               #对图像进行一系列的光度度量失真变换，如亮度、对比度、饱和度、色调的随机变化。这有助于模型对光照条件的变化更加鲁棒
    #dict(type='EdgeMap', edge_type='label_boundary', edge_width=1),          #新增：为边界头生成对应的边界真值。通常可以在数据管道中通过边缘提取方法（如 Sobel 算子）生成边界掩码。
    dict(type='PackSegInputs')                                        #将处理后的图像和标注打包成模型训练所需的格式。这可能包括将图像、标注和其他可能需要的元数据（如图像尺寸、缩放比例等）组合成一个字典或张量列表。
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),          #调整图像的大小,保持图像的原始宽高比。
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),                                     #加载与图像对应的标注数据。
    dict(type='PackSegInputs')                                        #将图像和标注数据打包成模型训练或推理所需的格式。
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),          #调整图像的大小,保持图像的原始宽高比。
    #dict(type='SegSlidingWindowCrop', window_size=(1000, 1000), stride=(1000, 1000)),  # 滑动窗口裁剪
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    #dict(type='LoadAnnotations'),                                     #加载与图像对应的标注数据。
    #dict(type='SlidingWindow', window_size=(512, 512), stride=(256, 256)),  # 滑窗配置
    dict(type='PackSegInputs'),                                       #将图像和标注数据打包成模型训练或推理所需的格式。
    #dict(type='SegSlidingWindowMerge')  # 预测后拼接结果
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]                       #是一个包含不同缩放比例的列表，用于测试时增强（Test-Time Augmentation, TTA）
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),               #通常涉及读取磁盘上的图像文件（如JPEG、PNG等），并将其解码为计算机可以处理的像素数据。backend_args=None表明没有为加载图像指定任何后端参数。
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,                                                  # 每批处理的图像数量
    num_workers=4,                                                 # 用于数据加载的子进程数
    persistent_workers=True,                                       # 指示是否使用持久化worker，这有助于在数据加载时减少CPU开销
    sampler=dict(type='InfiniteSampler', shuffle=True),            # 指定采样器类型，这里使用InfiniteSampler以无限循环方式遍历数据集，并启用打乱
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline))
val_dataloader = dict(                                             #数据加载器
    batch_size=1,                                                  #这表示每次从数据集中加载一个样本（图像）到内存中
    num_workers=4,                                                 #这指定了用于数据加载的子进程数。使用多个子进程可以并行加载数据，从而加快数据加载速度
    persistent_workers=True,                                       #这表示数据加载器的工作进程将在数据集迭代之间保持活动状态，从而避免在每次迭代时都重新初始化进程，这可以提高数据加载的效率
    sampler=dict(type='DefaultSampler', shuffle=False),            #这指定了如何从数据集中选择样本。这里使用的是DefaultSampler，并且shuffle=False表示不随机打乱样本顺序。在验证和测试阶段，通常不需要打乱样本顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=val_pipeline))                                  #数据预处理流程（pipeline）
#test_dataloader = val_dataloader
test_dataloader = dict(                                             #数据加载器
    batch_size=1,                                                  #这表示每次从数据集中加载一个样本（图像）到内存中
    num_workers=4,                                                 #这指定了用于数据加载的子进程数。使用多个子进程可以并行加载数据，从而加快数据加载速度
    persistent_workers=True,                                       #这表示数据加载器的工作进程将在数据集迭代之间保持活动状态，从而避免在每次迭代时都重新初始化进程，这可以提高数据加载的效率
    sampler=dict(type='DefaultSampler', shuffle=False),            #这指定了如何从数据集中选择样本。这里使用的是DefaultSampler，并且shuffle=False表示不随机打乱样本顺序。在验证和测试阶段，通常不需要打乱样本顺序
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/test.txt',
        pipeline=test_pipeline

    )

)


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU','mDice','mFscore'], classwise=True)      #IoUMetric评估器，它基于交并比（Intersection over Union, IoU），mIoU是多个类别IoU的平均值 'mDice','mFscore'# 开启按类别输出指标
#test_evaluator = val_evaluator
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU',],
                      format_only=True,                          #测试数据没有标注时设置为true
                      output_dir='my_output/work_dirs/format_results',
                      filename_pattern='{}',
                      )