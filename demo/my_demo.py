from mmdet.apis import init_detector, inference_detector
import mmcv
from mmdet.registry import VISUALIZERS

config_file = '/mmdetection/configs/mask_rcnn/mask-rcnn_r101_fpn_1x_taco.py'
checkpoint_file = '/mmdetection/work_dirs/mask-rcnn_r50_fpn_albu-1x_taco/epoch_1.pth'

# config_file = '/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
# checkpoint_file = '/mmdetection/demo/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'

model = init_detector(config_file, checkpoint_file, device='cpu')

img = '/mmdetection/demo/pipe2.jpg'  # Replace with your image path
result = inference_detector(model, img)


# show the results

img = mmcv.imread(img)
img = mmcv.imconvert(img, 'bgr', 'rgb')
out_file = 'output.jpg'

# init visualizer
visualizer_cfg = dict(type='DetLocalVisualizer', name='visualizer')
visualizer = VISUALIZERS.build(visualizer_cfg)

visualizer.add_datasample(
    'result',
    img,
    data_sample=result,
    draw_gt=False,
    out_file=out_file,
    pred_score_thr=0.5)
