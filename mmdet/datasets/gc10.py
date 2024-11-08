# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module()
class GC10Dataset(XMLDataset):
    """Dataset for PASCAL VOC."""


    METAINFO = {
        'classes':
            ('1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen',
             '9_zhehen', '10_yaozhe'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (109, 63, 54),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157)]
    }

