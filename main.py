import cv2
import numpy as np
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)



file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip("\n").split("\n")

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

def get_color_by_class(class_id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    return colors[class_id % len(colors)]

img = cv2.imread('traffic.jpg')
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.50)

font = cv2.FONT_HERSHEY_SIMPLEX
img_height, img_width = img.shape[:2]

font_scale = img_width / 300.0
thickness = int(font_scale * 1.50)

if isinstance(ClassIndex, np.ndarray):
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        if 0 < ClassInd <= len(classLabels):
            color = get_color_by_class(ClassInd)
            cv2.rectangle(img, boxes, color, thickness)
            cv2.putText(img, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=color, thickness=thickness)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# webcam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Something wrong with webcam connection")

font_scale=3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    print(ClassIndex)
    if isinstance(ClassIndex, np.ndarray):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if 0 < ClassInd <= len(classLabels):
                color = get_color_by_class(ClassInd)
                print(f'Label: {classLabels[ClassInd - 1]}, Confidence: {conf}')
                cv2.rectangle(frame, boxes, color, 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(0, 255, 0), thickness=2)

    cv2.imshow('objectdetection', frame)

    if cv2.waitKey(2) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()