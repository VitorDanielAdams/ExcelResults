#vid = cv2.VideoCapture(0)
import cv2
import numpy as np
from rmn import RMN
from obj_tracking import CentroidTracker
m = RMN()

vid = cv2.VideoCapture(0)
#vid.open("rtsp://admin:labdeia123@192.168.1.64:554/h264/ch1/sub")

ct = CentroidTracker()
auxId = -1

while True:
    ret, frame = vid.read()
    if frame is None or ret is not True:
        continue

    try:
        frame = np.fliplr(frame).astype(np.uint8)
        
        boxes = []

        results = m.detect_emotion_for_single_frame(frame)
        print(results)
        
        if(len(results) != 0):
            i = 0
            for i in range(len(results)):
                boxes.append([results[i]['xmin'], results[i]['ymin'], results[i]['xmax'], results[i]['ymax']])
        
        objects = ct.update(boxes)

        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            if(objectID > auxId):
                print('ID DIFERENTE')

            auxId = objectID

        print(boxes)
        frame = m.draw(frame, results)

        cv2.rectangle(frame, (1, 1), (220, 25), (223, 128, 255), cv2.FILLED)
        cv2.putText(frame, f"press q to exit", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imshow("disp", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    except Exception as err:
        print(err)
        continue

cv2.destroyAllWindows()