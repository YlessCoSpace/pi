import cv2
from lib_network import *

cap = find_network_cam(username='admin', password='admin')

if __name__ == '__main__':
    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Unable to read frame from stream')
            break

        cv2.imshow('Image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
