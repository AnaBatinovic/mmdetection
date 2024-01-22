WEIGHTS_NAME=$1
docker cp mmdetection_cont:/mmdetection/work_dirs/rtmdet-ins_s_8xb32-300e_taco/$WEIGHTS_NAME.pth .
