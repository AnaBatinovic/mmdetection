_base_ = './htc-without-semantic_r50_fpn_1x_taco.py'
model = dict(
    data_preprocessor=dict(pad_seg=True),
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            seg_scale_factor=1 / 8,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))

