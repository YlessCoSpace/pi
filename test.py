import logging
import cv2 as cv
from torch import classes
from ultralytics import YOLO
from scan_ip_port import scan_ip_port


def find_network_cam(username: str, password: str) -> cv.VideoCapture:
    cam_ip = scan_ip_port(8554, max_workers=255)
    if cam_ip:
        return cv.VideoCapture(f'rtsp://{username}:{password}@{cam_ip[0]}:8554/live')
    else:
        raise FileNotFoundError('No video devices on the network')


logging.getLogger("ultralytics").setLevel(logging.WARNING)
model = YOLO("yolov10n.pt")
cap = find_network_cam(username='admin', password='admin')
target_classes = [0, 60]

if __name__ == '__main__':
    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Unable to read frame from stream')
            break

        results = model.predict(frame, classes=target_classes)
        annot_frame = results[0].plot()
        cv.imshow('YOLOv10-N Detection', annot_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
