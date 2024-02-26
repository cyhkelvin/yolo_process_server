import cv2
import numpy as np
from os import path
import time
import yaml


from constants import ROOT
from libs.yolo_preprocess import detect, wrap_detection, format_yolov5
INPUT_WIDTH = 640
INPUT_HEIGHT = 640


class Detector():

    def __init__(self,
        config_path=path.join(ROOT, 'models', 'coco.yaml'),
        model_path=path.join(ROOT, 'models', 'yolov5s.onnx')):
        with open(config_path,'r', encoding='utf-8') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        self.class_list = config['names']
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def view_detection(self, streaming_str=None): 
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

        if streaming_str is None:
            cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(streaming_str)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            start = time.time()
            _, frame = cap.read()
            if frame is None:
                print("End of stream")
                break
            inputImage = format_yolov5(frame)
            outs = detect(inputImage, self.net)
            class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

            for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                color = colors[int(classid) % len(colors)]
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, self.class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
            end = time.time()

            fps = 1 / (end - start)
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(fps_label+ "; Detections: " + str(len(class_ids)))
            cv2.imshow("output", frame)

            if cv2.waitKey(1) > -1:
                print("finished by user")
                break
