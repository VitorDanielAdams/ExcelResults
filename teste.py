#vid = cv2.VideoCapture(0)
import cv2
import numpy as np
from rmn import RMN
#from obj_tracking import CentroidTracker
 
m = RMN()
 
def finish(arquivo, arquivo70, cont, happy):
    res = happy/cont
    arquivo70.write("A % de felicidade foi de " + str(format((res*100),'.1f')) 
    + "% em " + str(cont) + " frames - " + str(happy) + "/" + str(cont))
 
    arquivo.close()
    arquivo70.close()
    cv2.destroyAllWindows()
 
vid = cv2.VideoCapture("video.mp4")
#vid.open("rtsp://admin:labdeia123@192.168.1.64:554/h264/ch1/sub")
 
#ct = CentroidTracker()
auxId = -1
cont = 0
y = 0
happy = 0
 
arquivo = open('resultados.xls','w') 
arquivo70 = open('resultados70.txt' , 'w')
 
while True:
    ret, frame = vid.read()
    if frame is None or ret is not True:
        finish(arquivo, arquivo70, cont, happy)
        continue
 
    try:
        frame = np.fliplr(frame).astype(np.uint8)
        
        results = m.detect_emotion_for_single_frame(frame)
 
        for result in results:
            emolabel = result['emo_label']
            emoproba = result['emo_proba']
            
            #tabela excel
            if(emoproba > 0.7):
                cont = cont + 1
                
                if(emolabel == 'happy'):
                    happy += 1
                    y = 3
                elif(emolabel == 'surprise'):
                    y = 2
                elif(emolabel == 'neutral'):
                    y = 0
                elif(emolabel == 'fear'):
                    y = -1
                elif(emolabel == 'angry'):
                    y = -2
                elif(emolabel == 'sad'):
                    y = -3
                elif(emolabel == 'disgust'):
                    y = -4
                
                arquivo.write(str(y) + "\n")
 
        #print(boxes)
        frame = m.draw(frame, results)
 
        cv2.rectangle(frame, (1, 1), (220, 25), (223, 128, 255), cv2.FILLED)
        cv2.putText(frame, f"press q to exit", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imshow("disp", frame)
 
        if cv2.waitKey(1) == ord("q"):
            finish(arquivo, arquivo70, cont, happy)
            break
    
    except Exception as err:
        print(err)
        continue
 
 