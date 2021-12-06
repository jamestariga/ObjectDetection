import cv2
thres = 0.6

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #captureDevice = camera
cap.set(3, 640)
cap.set(4, 480)

classNames = []

# Windows version, comment line below if running in raspberry pi
classFile = 'coco.names'

# Raspberry Pi version, remove '#' on the next line
# classFile = '/home/pi/Desktop/ObjectDetection/coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Windows version, comment line below if running in raspberry pi
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Raspberry Pi version, remove '#' on the next line
# configPath = '/home/pi/Desktop/ObjectDetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

# Windows version, comment line below if running in raspberry pi
weightsPath = 'frozen_inference_graph.pb'

# Raspberry Pi version, remove '#' on the next line
# weightsPath = '/home/pi/Desktop/ObjectDetection/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.2)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Object Detection", img)
    cv2.waitKey(1)
