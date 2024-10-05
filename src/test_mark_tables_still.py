import cv2
import argparse
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


image = cv2.imread('videos/out/ffmpeg_1.bmp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mark points on an image.")
    parser.add_argument("tables", type=int, help="Number of tables to mark on the image.")
    args = parser.parse_args()
    num_points = args.tables * 2

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Image', cb_click)

    while True:
        frame = image

        draw_overlays(frame)
        cv2.imshow('Image', frame)

        for _ in range(33):
            if len(points) >= num_points:
                stopped = True
                break

            if cv2.waitKey(30) & 0xFF == ord('q'):
                stopped = True
                break

        if stopped:
            break

    o = [tuple(points[i:i + 2]) for i in range(0, len(points), 2)]
    print(o)
    save_obj(o, 'tables.pkl')
    print('Saved points!')
    cv2.destroyAllWindows()
