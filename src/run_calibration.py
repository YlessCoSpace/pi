import cv2
from lib_find_network_cam import find_network_cam
from lib_transform import *


def cb_click(_event, _x, _y, _flags, _param):
    if _event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((_x, _y))


cap = find_network_cam(username='admin', password='admin')
target_fps = 1
frame_interval = int(cap.get(cv2.CAP_PROP_FPS)) // target_fps

if __name__ == '__main__':
    if not cap.isOpened():
        print(f'Error: Unable to open video stream')
        exit()

    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Image', cb_click)

    points = []

    i = 0
    stopped = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Unable to read frame from stream')
            exit(1)

        for _ in range(10):
            # Drawing Logic
            for point in points:
                cv2.circle(frame, point, 16, (0, 0, 255), -1)

            if len(points) == 8:
                cv2.line(frame, points[6], points[7], (255, 255, 0), 3)

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
                _tf = perspective_tf_image(frame, mat, (ow, oh))

                save_mat(mat, 'perspective_matrix.pkl')

                cv2.imshow('Image', frame)
                stopped = True
                break

            if len(points) >= 6:
                cv2.line(frame, points[4], points[5], (255, 0, 0), 3)
            if len(points) >= 4:
                cv2.line(frame, points[2], points[3], (0, 255, 0), 3)
            if len(points) >= 2:
                cv2.line(frame, points[0], points[1], (0, 0, 255), 3)

            cv2.imshow('Image', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        if stopped:
            break

        for _ in range(frame_interval - 1):
            cap.grab()

    cv2.namedWindow('Transformed')
    cv2.imshow('Transformed', _tf)
    cv2.waitKey(5000)

    cv2.destroyAllWindows()
