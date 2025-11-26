import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.builder import HEADS
from mmseg.models.builder import build_loss
import matplotlib.pyplot as plt
import os

@HEADS.register_module()
class BoundaryHead(BaseDecodeHead):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 num_convs=2,
                 loss_decode=None,
                 **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)

        self.num_classes = num_classes
        print(self.num_classes)
        self.loss_decode = nn.CrossEntropyLoss(ignore_index=255)  # 直接使用CrossEntropyLoss，设置 ignore_index 为 255

        self.convs = nn.Sequential(*[  # 定义卷积层
            nn.Sequential(
                nn.Conv2d(
                    in_channels if i == 0 else channels,
                    channels,
                    kernel_size=3,
                    padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )
            for i in range(num_convs)
        ])

        self.edge_pred = nn.Conv2d(channels, num_classes, kernel_size=1)  # 最后一层输出类别数

    def forward(self, inputs):
        selected_feature = inputs[0]  # 选择第一个特征图
        x = self.convs(selected_feature)
        boundary_logits = self.edge_pred(x)  # 得到边界预测结果
        print("boundary_logits min:", boundary_logits.min().item())
        print("boundary_logits max:", boundary_logits.max().item())

        return boundary_logits

    def loss_by_feat(self, boundary_logits, batch_data_samples) -> dict:
        # 获取标签
        edge_maps = torch.stack([sample.gt_edge_map.data for sample in batch_data_samples], dim=0)
        edge_maps = edge_maps.float()

        # 对标签进行插值，使其大小与输出的 logits 相匹配
        edge_maps_resized = F.interpolate(
            edge_maps, size=boundary_logits.shape[2:], mode='bilinear', align_corners=self.align_corners
        ).squeeze(1).long()  # 确保是 Long 类型并去掉多余的通道维度

        # 打印信息
        print("edge_maps unique values:", torch.unique(edge_maps))
        print("edge_maps_resized unique values:", torch.unique(edge_maps_resized))
        print("boundary_logits shape:", boundary_logits.shape)
        print("edge_maps_resized shape:", edge_maps_resized.shape)

        # 使用整个 boundary_logits 而不是只取第一个通道
        loss = self.loss_decode(boundary_logits, edge_maps_resized)  # 使用整个 logits 计算损失

        total_loss = dict()
        total_loss['loss_boundary'] = loss

        return total_loss


    def visualize_and_save(self, boundary_logits, edge_maps_resized, output_dir='visualizations', prefix='comparison'):
        """
        可视化和保存预测边界与真实边界的对比结果。
        Args:
            boundary_logits (torch.Tensor): 预测的边界信息，形状为 (N, num_classes, H, W)。
            edge_maps_resized (torch.Tensor): 调整尺寸后的真实边界，形状为 (N, num_classes, H, W)。
            output_dir (str): 保存结果的目录。
            prefix (str): 文件名前缀。
        """
        os.makedirs(output_dir, exist_ok=True)

        # 转换预测结果为概率
        boundary_probs = torch.sigmoid(boundary_logits)

        # 二值化
        boundary_preds = (boundary_probs > 0.5).float()

        for i in range(boundary_preds.size(0)):  # 遍历 batch
            for c in range(boundary_preds.size(1)):  # 遍历类别
                pred = boundary_preds[i, c].cpu().numpy()
                gt = edge_maps_resized[i, c].cpu().numpy()

                # 创建对比图
                plt.figure(figsize=(10, 5))

                # 预测图
                plt.subplot(1, 2, 1)
                plt.imshow(pred, cmap='gray')
                plt.title(f'Prediction - Sample {i} Class {c}')
                plt.axis('off')

                # 真实边界图
                plt.subplot(1, 2, 2)
                plt.imshow(gt, cmap='gray')
                plt.title(f'Ground Truth - Sample {i} Class {c}')
                plt.axis('off')

                # 保存图像
                save_path = os.path.join(output_dir, f'{prefix}_sample_{i}_class_{c}.png')
                plt.savefig(save_path)
                plt.close()

