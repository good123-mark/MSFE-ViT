import numpy as np
from scipy.ndimage import sobel
from skimage.feature import canny
from mmengine.registry import TRANSFORMS
from mmcv.image import imresize
#from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class EdgeMap:
    """从分割标签或图像中提取边界信息。

    Args:
        edge_type (str): 边界提取类型，可选 ['sobel', 'canny', 'label_boundary']。
        edge_width (int): 边界宽度，仅对 'label_boundary' 生效。
        sigma (float): 用于 'canny' 方法的高斯平滑参数。
    """

    def __init__(self, edge_type='sobel', edge_width=1, sigma=1.0):
        assert edge_type in ['sobel', 'canny', 'label_boundary'], \
            f"Unsupported edge_type {edge_type}"
        self.edge_type = edge_type
        self.edge_width = edge_width
        self.sigma = sigma

    def _sobel_edge(self, mask):
        """通过 Sobel 算子提取边界。"""
        edge_x = sobel(mask, axis=0)
        edge_y = sobel(mask, axis=1)
        edge_map = np.sqrt(edge_x ** 2 + edge_y ** 2) > 0
        return edge_map.astype(np.uint8)

    def _canny_edge(self, image):
        """通过 Canny 算法提取边界。"""
        edge_map = canny(image, sigma=self.sigma)
        return edge_map.astype(np.uint8)

    def _label_boundary(self, mask, num_classes=3):
        """
        从分割标签中提取每个类别的边界。

        Args:
            mask (np.ndarray): 分割标签图，形状为 (H, W)。
            fixed_num_classes (int, optional): 固定的类别数量。如果提供，会根据此值进行填充。

        Returns:
            np.ndarray: 边界图，形状为 (fixed_num_classes, H, W) 或 (num_classes, H, W)。
        """
        #num_classes = mask.max() + 1  # 获取实际类别数量
        edge_maps = []

        # 提取每个类别的边界
        for c in range(num_classes):
            class_mask = (mask == c)
            dilated = np.pad(class_mask, pad_width=1, mode='constant')
            diff = (
                    (dilated[:-2, 1:-1] != class_mask) |
                    (dilated[2:, 1:-1] != class_mask) |
                    (dilated[1:-1, :-2] != class_mask) |
                    (dilated[1:-1, 2:] != class_mask)
            )
            edge = diff.astype(np.uint8)

            # 应用边界扩展 (可选，提升边界宽度)
            if self.edge_width > 1:
                from scipy.ndimage import binary_dilation
                for _ in range(self.edge_width - 1):
                    edge = binary_dilation(edge)
            edge_maps.append(edge)

        edge_maps = np.stack(edge_maps, axis=0)  # 形状为 (num_classes, H, W)

        # # 如果需要固定类别数，进行填充
        # if fixed_num_classes is not None:
        #     if edge_maps.shape[0] < fixed_num_classes:
        #         pad_shape = (fixed_num_classes - edge_maps.shape[0], mask.shape[0], mask.shape[1])
        #         pad_map = np.zeros(pad_shape, dtype=edge_maps.dtype)
        #         edge_maps = np.concatenate([edge_maps, pad_map], axis=0)
        #     elif edge_maps.shape[0] > fixed_num_classes:
        #         edge_maps = edge_maps[:fixed_num_classes]  # 截断到固定类别数

        return edge_maps


    def __call__(self, results):
        """
        处理数据并添加边界图。

        Args:
            results (dict): 包含原始数据（如 'gt_seg_map', 'img' 等）的字典。

        Returns:
            dict: 更新后的结果字典，增加了 `gt_edge_map`。
        """
        mask = results['gt_seg_map']
        if self.edge_type == 'sobel':
            edge_map = self._sobel_edge(mask)  # (H, W)
            edge_map = np.expand_dims(edge_map, axis=0)  # (1, H, W)
        elif self.edge_type == 'canny':
            image = results['img']
            if len(image.shape) == 3:  # 转为灰度图
                image = np.mean(image, axis=2)
            edge_map = self._canny_edge(image)  # (H, W)
            edge_map = np.expand_dims(edge_map, axis=0)  # (1, H, W)
        elif self.edge_type == 'label_boundary':
            edge_map = self._label_boundary(mask)  # (num_classes, H, W)
            # fixed_num_classes = results.get('num_classes', 2)  # 固定类别数量
            # # 如果实际类别数少于固定值，则补充
            # if edge_map.shape[0] < fixed_num_classes:
            #     pad_shape = (fixed_num_classes - edge_map.shape[0], mask.shape[0], mask.shape[1])
            #     pad_map = np.zeros(pad_shape, dtype=edge_map.dtype)
            #     edge_map = np.concatenate([edge_map, pad_map], axis=0)
        else:
            raise NotImplementedError(f"Edge type {self.edge_type} not implemented.")

        #print(f"EdgeMap output shape: {edge_map.shape}")  # 打印维度

        edge_map_resized = [
            imresize(edge, size=mask.shape, interpolation='nearest')
            for edge in edge_map
        ]
        edge_map = np.stack(edge_map_resized, axis=0)  # 重新堆叠
        #
        # print(f"EdgeMap output shape: {edge_map.shape}")  # 打印维度

        # 添加边界图 (C, H, W)
        results['gt_edge_map'] = edge_map

        return results

# @TRANSFORMS.register_module()
# class SlidingWindowCrop(BaseTransform):
#     """使用滑动窗口方式裁剪输入图像，保证大尺寸图像不会降采样"""
#
#     def __init__(self, crop_size=(1024, 1024), stride=(512, 512)):
#         self.crop_size = crop_size
#         self.stride = stride
#
#     def __call__(self, results):
#         """执行滑动窗口裁剪"""
#         img = results['img']  # 获取输入图像
#         h, w, _ = img.shape  # 获取图像尺寸
#         crop_h, crop_w = self.crop_size
#         stride_h, stride_w = self.stride
#
#         crops = []
#         crop_infos = []
#
#         for i in range(0, h - crop_h + 1, stride_h):
#             for j in range(0, w - crop_w + 1, stride_w):
#                 crop = img[i:i + crop_h, j:j + crop_w]
#                 crops.append(crop)
#                 crop_infos.append((i, j))
#
#         results['img'] = crops  # 存储所有裁剪结果
#         results['crop_infos'] = crop_infos  # 存储裁剪坐标信息
#         return results
#
#
# @TRANSFORMS.register_module()
# class SlidingWindowMerge(BaseTransform):
#     """将滑动窗口裁剪的结果重新拼接成完整图像"""
#
#     def __call__(self, results):
#         crops = results['img']  # 读取所有裁剪的图像
#         crop_infos = results['crop_infos']  # 获取位置信息
#         merged_result = self.merge_sliding_window_results(crops, crop_infos, results['ori_shape'])
#         results['img'] = merged_result
#         return results
#
#     def merge_sliding_window_results(self, crops, crop_infos, ori_shape):
#         """合并多个滑动窗口结果"""
#         h, w, _ = ori_shape  # 原始图像尺寸
#         merged_img = np.zeros((h, w, 3), dtype=np.uint8)  # 创建空白图像
#         count_map = np.zeros((h, w, 1), dtype=np.uint8)  # 计数掩膜，避免重叠区域被多次叠加
#
#         for crop, (i, j) in zip(crops, crop_infos):
#             crop_h, crop_w, _ = crop.shape
#             merged_img[i:i + crop_h, j:j + crop_w] += crop
#             count_map[i:i + crop_h, j:j + crop_w] += 1
#
#         # 避免除零错误
#         count_map[count_map == 0] = 1
#         merged_img = merged_img / count_map  # 归一化，处理重叠区域
#         return merged_img.astype(np.uint8)
#
#
