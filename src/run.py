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
model = YOLO("yolo11n.pt")

# target_classes = [0, 60]

if __name__ == '__main__':
    try:
        cap = find_network_cam(username='admin', password='admin')
    except FileNotFoundError:
        print(f'Error: No video stream available in the network')
        exit()

    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()
