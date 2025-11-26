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
                 upsample_scale=2,  # 新增参数：上采样倍数
                 dropout_ratio=0.1,  # 新增 dropout_ratio 参数
                 loss_decode=None,
                 **kwargs):
        super().__init__(in_channels=in_channels, channels=channels, num_classes=num_classes, **kwargs)

        self.in_index = in_index  # 保存索引
        self.num_classes = num_classes
        self.loss_decode = build_loss(loss_decode)
        self.dropout_ratio = dropout_ratio  # 保存 dropout_ratio

        # 用于边界特征提取的卷积层
        self.convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(
                    in_channels if i == 0 else channels,
                    channels,
                    kernel_size=3,
                    padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=self.dropout_ratio)  # 添加 Dropout
            )
            for i in range(num_convs)
        ])

        # 用于上采样的反卷积层（反卷积上采样）
        self.upsample = nn.ConvTranspose2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=upsample_scale * 2,
            stride=upsample_scale,
            padding=upsample_scale // 2,
            output_padding=upsample_scale % 2
        )

        # 用于边界预测的最终卷积
        self.edge_pred = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs, output_size=[512,  512]):
        """
        前向传播，用于生成边界预测。
        Args:
            inputs (list[torch.Tensor]): 来自 `neck` 的特征图列表。
            output_size (tuple[int, int], optional): 输出图像的目标分辨率 (H, W)。
        Returns:
            torch.Tensor: 上采样后的边界预测，形状为 (N, num_classes, H, W)。
        """
        # 获取指定索引的特征图
        selected_feature = inputs[self.in_index]

        # 提取边界特征
        x = self.convs(selected_feature)

        # 上采样至目标分辨率
        #x = self.upsample(x)
        #print(f"boundary_logits.shape(upsample): {x.shape}")

        # 生成边界预测
        boundary_logits = self.edge_pred(x)

        # 如果指定了 `output_size`，再进一步调整分辨率（使用插值）
        if output_size is not None:
            boundary_logits = F.interpolate(
                boundary_logits, size=output_size, mode='bilinear', align_corners=False
            )

        #print(f"boundary_logits.shape: {boundary_logits.shape}")

        return boundary_logits

    def loss_by_feat(self, boundary_logits, batch_data_samples):
        """
        计算损失，逐类别对边界图进行监督。
        Args:
            boundary_logits (torch.Tensor): 预测的边界图，形状为 (N, num_classes, H, W)。
            batch_data_samples (list[DataSample]): 包含标注边界信息的样本。
        Returns:
            dict: 包含边界损失的字典。
        """
        # 提取标注边界图
        edge_maps = torch.stack([sample.gt_edge_map.data for sample in batch_data_samples], dim=0)



        # # 确保预测结果和标注图一致分辨率
        # boundary_logits_resized = F.interpolate(
        #     boundary_logits, size=edge_maps.shape[2:],  # 调整到标注图尺寸
        #     mode='bilinear', align_corners=False
        # )
        #boundary_logits_resized = (boundary_logits > 0.05).float()
        #print(f"edge_maps.shape: {edge_maps.shape}")
        #print(f"boundary_logits_resized: {boundary_logits.shape}")
        # 可视化和保存
        #self.visualize_and_save(boundary_logits, edge_maps, output_dir='visualizations', prefix='comparison')

        # 逐类别计算边界损失
        loss = []
        for c in range(self.num_classes):
            pred = boundary_logits[:, c, :, :].unsqueeze(1)  # 恢复通道维度
            target = edge_maps[:, c, :, :].contiguous().unsqueeze(1)  # 保证维度一致

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
        boundary_preds = torch.sigmoid(boundary_logits)

        # 二值化
        #boundary_preds = (boundary_probs > 0.5).float()

        for i in range(boundary_preds.size(0)):  # 遍历 batch
            for c in range(boundary_preds.size(1)):  # 遍历类别
                pred = boundary_preds[i, c].detach().cpu().numpy()
                gt = edge_maps_resized[i, c].detach().cpu().numpy()

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

