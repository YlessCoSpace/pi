import cv2
import argparse
from lib_transform import *


def cb_click(_event, _x, _y, _flags, _param):
    if _event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((_x, _y))
    cv2.circle(frame, points[-1], 16, (0, 0, 255), -1)


image = cv2.imread('videos/out/ffmpeg_37.bmp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mark points on an image.")
    parser.add_argument("num_points", type=int, help="Number of points to mark on the image.")
    args = parser.parse_args()
    num_points = args.num_points

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Image', cb_click)

    points = []
    stopped = False

    while True:
        frame = image
        for _ in range(10):
            if len(points) >= num_points:
                stopped = True
                break

            cv2.imshow('Image', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        if stopped:
            break

    save_obj(tuple(points), 'tables.pkl')
    cv2.destroyAllWindows()
