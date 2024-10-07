import cv2
from lib_network import *

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

        cv2.imshow('Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
