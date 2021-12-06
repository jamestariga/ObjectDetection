import cv2
thres = 0.6

classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, draw=True):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=0.5)

    objectInfo = []

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            objectInfo.append([box, classNames])
            if (draw):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

if __name__ == "__main__":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # captureDevice = camera
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img, False)
        print(objectInfo)
        cv2.imshow("Object Detection", img)
        cv2.waitKey(1)