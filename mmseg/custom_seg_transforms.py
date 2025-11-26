import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SegSlidingWindowCrop(BaseTransform):
    """对大图进行滑动窗口裁剪（仅用于推理阶段）"""

    def __init__(self, window_size=(512, 512), stride=(256, 256)):
        self.window_size = window_size
        self.stride = stride

    def transform(self, results):
        img = results['img']
        h, w = img.shape[:2]
        crop_h, crop_w = self.window_size
        stride_h, stride_w = self.stride

        crops = []
        crop_infos = []
        for i in range(0, h - crop_h + 1, stride_h):
            for j in range(0, w - crop_w + 1, stride_w):
                crop = img[i:i + crop_h, j:j + crop_w]
                crops.append(crop)
                crop_infos.append((i, j))

        results['crops'] = crops
        results['crop_infos'] = crop_infos
        # 同时保存原始尺寸信息，便于后续合并
        results['ori_shape'] = img.shape
        return results


@TRANSFORMS.register_module()
class SegSlidingWindowMerge(BaseTransform):
    """将滑动窗口裁剪后的预测结果合并为完整预测图"""

    def transform(self, results):
        # 这里假设模型预测结果存放在 'preds' 字段中
        preds = results['preds']
        crop_infos = results['crop_infos']
        ori_h, ori_w = results['ori_shape'][:2]

        merged_pred = np.zeros((ori_h, ori_w), dtype=np.float32)
        count_map = np.zeros((ori_h, ori_w), dtype=np.float32)

        for pred, (i, j) in zip(preds, crop_infos):
            crop_h, crop_w = pred.shape
            merged_pred[i:i + crop_h, j:j + crop_w] += pred
            count_map[i:i + crop_h, j:j + crop_w] += 1

        count_map[count_map == 0] = 1
        merged_pred = merged_pred / count_map
        results['merged_pred'] = merged_pred.astype(np.uint8)
        return results