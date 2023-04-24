# Importing Libraries
import time
import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model


model = load_model(f"Final\Assignment-1\mymodel.h5",compile=False)

with open(f'Final\Assignment-1\category2label.pkl', 'rb') as cat:
    catrgory_2_label = pickle.load(cat)


print("Categories: ",catrgory_2_label)

img_size = (100, 100)

box_colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)}

face_cascade = cv2.CascadeClassifier(f'Final\Assignment-1\haarcascade_frontalface_default.xml')

# video from webcam

cap = cv2.VideoCapture(0)
begin_time = time.time()
frame_cnt = 0

while cap.isOpened():
    rate, frame = cap.read()

    if not rate:
        break

    frame_cnt += 1
    frame = cv2.flip(frame, 1)
    face = face_cascade.detectMultiScale(frame, 1.3, 5)

    for (x, y, w, h) in face:
        # prediction

        roi = frame[y:y+h, x:x+w]
        data = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), img_size)
        data = data / 255.
        data = data.reshape((1,) + data.shape)

        pred = model.predict(data)

        target = np.argmax(pred, axis=1)[0]

        # drawing box
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h),color=box_colors[target], thickness=2)
        texts = "{}: {:.2f}".format(catrgory_2_label[target], pred[0][target])
        cv2.putText(frame, text=texts, org=(x, y-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(255, 255, 255), thickness=1)

    endtime = time.time()-begin_time
    fps = frame_cnt / endtime

    cv2.putText(img=frame, text="FPS: "+str(round(fps, 2)), org=(10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  fontScale=0.5, color=(255, 255, 255), thickness=1)
    # Show the frame
    cv2.imshow('Face Mask Detection', frame)
    # for terminationg ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
