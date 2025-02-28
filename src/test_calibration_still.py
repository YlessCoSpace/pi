import cv2
from lib_transform import *

points = []
stopped = False
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
    global stopped, points, tf

    if _event != cv2.EVENT_LBUTTONDOWN:
        return

    points.append((_x, _y))
    cv2.circle(frame, points[-1], 16, (0, 0, 255), -1)

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
        tf = perspective_tf_image(frame, mat, (ow, oh))

        save_obj(mat, 'perspective_matrix.pkl')
        print('Saved weights!')
        stopped = True

    if len(points) >= 6:
        cv2.line(frame, points[4], points[5], (255, 0, 0), 3)
    if len(points) >= 4:
        cv2.line(frame, points[2], points[3], (0, 255, 0), 3)
    if len(points) >= 2:
        cv2.line(frame, points[0], points[1], (0, 0, 255), 3)

    draw_overlays(frame)
    cv2.imshow('Image', frame)


image = cv2.imread('videos/out/ffmpeg_1.bmp')

if __name__ == '__main__':
    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('Image', cb_click)

    while True:
        frame = image

        draw_overlays(frame)
        cv2.imshow('Image', frame)

        for _ in range(33):
            if len(points) >= 8:
                stopped = True
                break

            if cv2.waitKey(30) & 0xFF == ord('q'):
                stopped = True
                break

        if stopped:
            break

    cv2.namedWindow('Transformed')
    cv2.setWindowProperty('Transformed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Transformed', tf)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
