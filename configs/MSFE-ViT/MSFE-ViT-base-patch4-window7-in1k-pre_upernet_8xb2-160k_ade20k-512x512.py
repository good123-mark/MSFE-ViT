_base_ = [
    './swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 16, 2],
        num_heads=[4, 8, 16, 32]),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4
    ),

    decode_head=dict(
                     in_channels=[256, 256,256, 256],
                     type='MSFEHead',
                     pool_scales=(1, 2, 3, 6, 12),
                     num_classes=3,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_weight=1.0),
            dict(type='DiceLoss', loss_weight=0.4),
                     ]),

    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=3,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)



        ),
    ]


)

