# yolo_process_server
yolo series model inferenced with opencv backend and maintain service docker image
  - [installation](#installation)
  - [demo](#demo)
  - [data preparation](#data-preparation)
  - [model training](#model-training)
  - [reference](#reference)

## installation
### step-by-step build up environment on windows
  - opencv (with GPU support) preparation:
    - on windows: https://thinkinfi.com/install-opencv-gpu-with-cuda-for-windows-10/
    - on linux: https://medium.com/geekculture/setup-opencv-dnn-module-with-cuda-backend-support-for-linux-1677c627f3bd
    - pre-built whl files: https://github.com/cudawarped/opencv-python-cuda-wheels/releases
### build up docker images
  - docker file for yolov5
  - docker file for opencv-onnx

## demo
  - ####todo####

## data preparation

### labelImg for object detection data labeling
  - https://github.com/HumanSignal/labelImg
  - former version of label studio

### PixelAnnotationTool for segmentation data labeling
  - https://github.com/abreheret/PixelAnnotationTool

### fiftyone for well-prepared datasets(coco) downloading
  - https://github.com/voxel51/fiftyone
  - download datasets(coco) with correct format(yolo)

## model training
  - yolov5
  - yolov8

## reference
  - yolo model
    - training: https://github.com/ultralytics/yolov5  (AGPL-3.0 License/Ultralytics Licensing)
    - onnx-inference: https://gitee.com//ppov-nuc/yolov5_infer  (MIT License)
  - deepsort:
    - https://github.com/ZQPei/deep_sort_pytorch
  - opencv (with GPU) whl:
    - https://github.com/cudawarped/opencv-python-cuda-wheels
