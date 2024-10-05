import math
import time

import cv2
import argparse
from lib_find_network_cam import find_network_cam
from lib_transform import *

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

points = []
stopped = False


def draw_overlays(img):
    global points, COLORS

    for i, p in enumerate(points):
        cv2.circle(img, p, 16, COLORS[(i // 2) % len(COLORS)], -1)

    for i in range(0, (len(points) // 2) * 2, 2):
        tl = points[i]
        br = points[i + 1]
        bl = points[i][0], points[i + 1][1]
        tr = points[i + 1][0], points[i][1]
        cv2.line(img, tl, tr, COLORS[(i // 2) % len(COLORS)], 3)
        cv2.line(img, tr, br, COLORS[(i // 2) % len(COLORS)], 3)
        cv2.line(img, br, bl, COLORS[(i // 2) % len(COLORS)], 3)
        cv2.line(img, bl, tl, COLORS[(i // 2) % len(COLORS)], 3)


def cb_click(_event, _x, _y, _flags, _param):
    if _event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((_x, _y))

    draw_overlays(frame)
    cv2.imshow('Image', frame)


cap = cv2.VideoCapture('videos/IMG_4174.MOV')
target_fps = 10
target_fps = math.floor(target_fps)
frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) // target_fps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mark points on an image.")
    parser.add_argument("tables", type=int, help="Number of tables to mark on the image.")
    args = parser.parse_args()
    num_points = args.tables * 2

    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Image', cb_click)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Unable to read frame from stream')
            exit(1)

        draw_overlays(frame)
        cv2.imshow('Image', frame)

        for _ in range(1000 // target_fps):
            if len(points) >= num_points:
                stopped = True
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopped = True
                break

        if stopped:
            break

        # Skip frames
        for _ in range(frame_interval - 1):
            cap.grab()

    cap.release()
    o = [tuple(points[i:i + 2]) for i in range(0, len(points), 2)]
    print(o)
    save_obj(o, 'tables.pkl')
    print('Saved points!')
    time.sleep(3)
    cv2.destroyAllWindows()
