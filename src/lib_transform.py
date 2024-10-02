import numpy as np
import cv2


def tup_int(tup: tuple):
    return int(tup[0]), int(tup[1])


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if det == 0:
        return float('inf'), float('inf')

    x_i = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    y_i = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return x_i, y_i


def get_perspective_tf_mat(tl, tr, br, bl, dsize):
    w, h = dsize

    input_points = np.array([tl, tr, br, bl], dtype=np.float32)
    output_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    return cv2.getPerspectiveTransform(input_points, output_points)


def perspective_tf_image(image, mat, dsize):
    return cv2.warpPerspective(image, mat, dsize)


def perspective_tf_points(points, mat):
    _points = np.array(points, dtype=np.float32)
    _points = np.array([_points])
    return cv2.perspectiveTransform(_points, mat)


__all__ = [
    'tup_int',
    'line_intersection',
    'get_perspective_tf_mat',
    'perspective_tf_image',
    'perspective_tf_points'
]
