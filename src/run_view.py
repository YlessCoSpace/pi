import math

import cv2
from lib_network import *

cap = find_network_cam(username='admin', password='admin')
target_fps = 2
target_fps = math.floor(target_fps)
frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) // target_fps

if __name__ == '__main__':
    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    stopped = False

    try:

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error: Unable to read frame from stream')
                break

            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopped = True

            # for _ in range(1000 // target_fps):
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         stopped = True
            #         break

            if stopped:
                break

            # Skip frames
            # for _ in range(frame_interval - 1):
            #     cap.grab()
    except Exception:
        pass
    finally:
        print("releasing...")
        cap.release()
        cv2.destroyAllWindows()
