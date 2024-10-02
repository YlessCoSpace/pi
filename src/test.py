import logging
import cv2
from ultralytics import YOLO
from lib_find_network_cam import find_network_cam

logging.getLogger("ultralytics").setLevel(logging.WARNING)
model = YOLO("yolo11n.pt")
cap = find_network_cam(username='admin', password='admin')

if __name__ == '__main__':
    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Unable to read frame from stream')
            break

        results = model.predict(frame)
        annot_frame = results[0].plot()
        cv2.imshow('Detection', annot_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
