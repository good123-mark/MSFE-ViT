# mmseg/models/decode_heads/msfe_head.py

from mmseg.registry import MODELS
from .uper_head import UPerHead


@MODELS.register_module()
class MSFEHead(UPerHead):
    """MSFE Decoder Head."""

    def __init__(self,
                 pool_scales=(1, 2, 3, 6, 12),
                 **kwargs):
        super().__init__(pool_scales=pool_scales, **kwargs)
