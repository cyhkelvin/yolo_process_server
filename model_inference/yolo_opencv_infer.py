import cv2
import numpy as np
from os import path
import time
import yaml


ROOT = path.dirname(__file__)
INPUT_WIDTH = 640
INPUT_HEIGHT = 640


class detector():

    def __init__(self,
        config_path=path.join(ROOT, 'coco.yaml'),
        model_path=path.join(ROOT, 'yolov5s.onnx')):
        with open(config_path,'r', encoding='utf-8') as f:
            config = yaml.load(f.read(),Loader=yaml.FullLoader)
        self.class_list = config['names']
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def start_detection(self, streaming_str=None): 
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


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    #print(output_data.shape)
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


if __name__ == '__main__':
    object_detector = detector('https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=11470')