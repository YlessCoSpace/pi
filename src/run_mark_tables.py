import queue
import threading
import time
import cv2
import argparse
from lib_network import *
from lib_transform import *

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

points = []
frame = None


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
    global frame, stopped
    if _event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((_x, _y))

    draw_overlays(frame)
    cv2.imshow('Image', frame)

    if len(points) >= num_points:
        stopped.set()


cap = find_network_cam(username='admin', password='admin')
target_fps = 30
q = queue.Queue()
stopped = threading.Event()

def receive():
    global frame
    prev = 0
    try:
        while not stopped.is_set():
            time_elapsed = time.time() - prev
            ret, frame = cap.read()
            if not ret:
                break

            if time_elapsed > 1. / target_fps:
                prev = time.time()
                q.put(frame)
    except Exception as e:
        print(f"Error in receive thread: {e}")
    finally:
        print("Releasing camera...")
        cap.release()


def display():
    out_video = cv2.VideoWriter('videos/out_mark.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Image', cb_click)
    print("Started displaying")

    while not stopped.is_set():
        try:
            if not q.empty():
                fr = q.get(timeout=1)
                draw_overlays(fr)
                cv2.imshow("Image", fr)
                out_video.write(fr)
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stopped.set()
            break

    draw_overlays(fr)
    for i in range(60):
        out_video.write(fr)

    print('Saving...')
    out_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mark points on an image.")
    parser.add_argument("tables", type=int, help="Number of tables to mark on the image.")
    args = parser.parse_args()
    num_points = args.tables * 2

    try:
        if not cap.isOpened():
            print(f'Error: Unable to open video stream')
            exit()

        p1 = threading.Thread(target=receive)
        p2 = threading.Thread(target=display)
        p1.start()
        p2.start()

        p1.join()
        p2.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping threads...")
        stopped.set()

    finally:
        p1.join()
        p2.join()

        o = [tuple(points[i:i + 2]) for i in range(0, len(points), 2)]
        print(o)
        save_obj(o, 'tables.pkl')
        print('Saved points!')

        print("Program terminated.")
