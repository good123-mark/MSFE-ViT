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
                 in_index=0,  # 新增参数：用于指定选择的特征图索引
                 num_convs=2,
                 loss_decode=None,
                 **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)

        self.in_index = in_index  # 保存索引
        self.num_classes = num_classes
        print(self.num_classes)
        self.loss_decode = build_loss(loss_decode)

        # 用于边界提取的卷积层
        self.convs = nn.Sequential(*[
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

        # 用于边界预测的最终卷积
        self.edge_pred = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        """
        前向传播，用于生成边界预测。
        Args:
            inputs (list[torch.Tensor]): 来自 `neck` 的特征图列表。
        Returns:
            torch.Tensor: 边界预测，形状为 (N, num_classes, H, W)。
        """
        # 假设使用第一个特征图
        selected_feature = inputs[self.in_index]  # 选择融合完整尺度特征图
        # 通过卷积提取边界特征
        x = self.convs(selected_feature)
        # 生成边界预测
        boundary_logits = self.edge_pred(x)

        return boundary_logits

    def loss_by_feat(self, boundary_logits, batch_data_samples) -> dict:
        """
        根据边界预测计算损失。
        Args:
            boundary_logits (torch.Tensor): 预测的边界信息，形状为 (N, num_classes, H, W)。
            batch_data_samples (list[dict]): 每个样本包含的真实边界信息等元数据。
        Returns:
            torch.Tensor: 边界损失。
        """

        """真实边界降采样为预测边界大小
        # 提取真实的边界信息 (N, num_classes, H, W)
        edge_maps = torch.stack([sample.gt_edge_map.data for sample in batch_data_samples], dim=0)

        # 转换为浮点类型
        edge_maps = edge_maps.float()

        # print(f"boundary_logits.shape: {boundary_logits.shape}")
        # print(f"edge_maps.shape: {edge_maps.shape}")
        # print(f"edge_maps dtype: {edge_maps.dtype}")

        # 调整尺寸以匹配预测
        edge_maps_resized = F.interpolate(
            edge_maps, size=boundary_logits.shape[2:],
            mode='nearest'      #mode = 'bilinear', align_corners = self.align_corners
        )

        # 阈值化处理，确保二值性
        edge_maps_resized = (edge_maps_resized > 0.5).float()
        #edge_maps_resized = edge_maps_resized.clamp(0, self.num_classes - 1)
        # 转换为 Long 类型，确保损失函数能够正确处理
        edge_maps_resized = edge_maps_resized.long()
        #print("edge_maps unique values:", torch.unique(edge_maps))
        #print("edge_maps_resized unique values:", torch.unique(edge_maps_resized))
        # 打印形状信息
        # print(f"boundary_logits.shape: {boundary_logits.shape}")
        # print(f"edge_maps_resized.shape: {edge_maps_resized.shape}")

        # 可视化和保存
        #self.visualize_and_save(boundary_logits, edge_maps_resized, output_dir='visualizations', prefix='comparison')


        # 逐类别计算边界损失
        loss = []
        # for c in range(self.num_classes):
        #     print(f"boundary_logits.shape: {boundary_logits.shape}")
        #     loss.append(self.loss_decode(boundary_logits[:, c, :, :], edge_maps_resized[:, c, :, :]))
        for c in range(self.num_classes):
            pred = boundary_logits[:, c, :, :].unsqueeze(1)  # 恢复通道维度
            target = edge_maps_resized[:, c, :, :].contiguous().unsqueeze(1)   # 保证连续性
            loss.append(self.loss_decode(pred, target))
        """
        edge_maps = torch.stack([sample.gt_edge_map.data for sample in batch_data_samples], dim=0)
        boundary_logits = boundary_logits.float()
        print(f"boundary_logits.shape: {boundary_logits.shape}")

        # 调整尺寸以匹配预测
        boundary_logits_resized = F.interpolate(
            boundary_logits, size=edge_maps.shape[2:],  # 与 edge_maps 保持一致
            mode='bilinear', align_corners=False  # 使用双线性插值
        )
        boundary_logits_resized = (boundary_logits_resized > 0.5).float()
        # boundary_logits_resized = boundary_logits_resized.long()

        # 可视化和保存
        self.visualize_and_save(boundary_logits_resized, edge_maps, output_dir='visualizations', prefix='comparison')

        # 逐类别计算边界损失
        loss = []
        # for c in range(self.num_classes):
        #     print(f"boundary_logits.shape: {boundary_logits.shape}")
        #     loss.append(self.loss_decode(boundary_logits[:, c, :, :], edge_maps_resized[:, c, :, :]))
        for c in range(self.num_classes):
            pred = boundary_logits_resized[:, c, :, :].unsqueeze(1)  # 恢复通道维度
            target = boundary_logits_resized[:, c, :, :].contiguous().unsqueeze(1)  # 保证连续性

            print(f"boundary_logits_resized.shape: {pred.shape}")
            print(f"edge_maps.shape: {target.shape}")

            loss.append(self.loss_decode(pred, target))

        # 计算平均边界损失
        total_loss = dict()
        total_loss['loss_boundary'] = sum(loss) / self.num_classes

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

