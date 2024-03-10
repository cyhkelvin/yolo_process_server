# model inference
  - get trained model in onnx fromat (can be export from yolo)
  - run with application.py
  - e.g. `python demo.py --weight yolov8s.onnx --config coco.yaml`

## object detection (yolo)
  - detection in given ROI(region of interest)

## object detection (yolo) with tracking (deep sort)
  - track in given ROI
  - track and count (people flow or car flow counting)
  - track and cross-line judgement

## segmentation (yolo)
  - segmentation in given ROI