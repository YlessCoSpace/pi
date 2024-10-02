import cv2
from lib_scan_ip_port import scan_ip_port


def find_network_cam(username: str, password: str) -> cv2.VideoCapture:
    cam_ip = scan_ip_port(8554, max_workers=255)
    if cam_ip:
        return cv2.VideoCapture(f'rtsp://{username}:{password}@{cam_ip[0]}:8554/live')
    else:
        raise FileNotFoundError('No video devices on the network')
