import queue
import threading
import time
import cv2
from lib_network import *
from lib_transform import *

frame = None
points = []
tf = None


def draw_overlays(img):
    global points

    for p in points:
        cv2.circle(img, p, 16, (0, 0, 255), -1)

    if len(points) >= 8:
        cv2.line(img, points[6], points[7], (255, 255, 0), 3)
    if len(points) >= 6:
        cv2.line(img, points[4], points[5], (255, 0, 0), 3)
    if len(points) >= 4:
        cv2.line(img, points[2], points[3], (0, 255, 0), 3)
    if len(points) >= 2:
        cv2.line(img, points[0], points[1], (0, 0, 255), 3)


def cb_click(_event, _x, _y, _flags, _param):
    global stopped, points, tf, frame

    if _event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((_x, _y))

    if len(points) == 8:
        ow, oh = (1000, 1000)

        _l1 = points[0], points[1]
        _l2 = points[2], points[3]
        _l3 = points[4], points[5]
        _l4 = points[6], points[7]

        _tl = tup_int(line_intersection(*_l1, *_l3))
        _tr = tup_int(line_intersection(*_l3, *_l4))
        _br = tup_int(line_intersection(*_l2, *_l4))
        _bl = tup_int(line_intersection(*_l1, *_l2))

        mat = get_perspective_tf_mat(_tl, _tr, _br, _bl, (ow, oh))
        tf = perspective_tf_image(frame, mat, (ow, oh))

        save_obj(mat, 'perspective_matrix.pkl')
        print('Saved weights!')
        stopped.set()

    draw_overlays(frame)
    cv2.imshow('Image', frame)


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
    out_video = cv2.VideoWriter('videos/out_cal.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1080))

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
        print("Program terminated.")
