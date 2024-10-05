import numpy as np
import cv2
import pickle

from typing import TypedDict

EntityEntry = TypedDict('EntityEntry', {
    'conf': float,
    'pos1': tuple[float, float],
    'pos2': tuple[float, float]
})

SubEntityTransformed = TypedDict('SubEntityTransformed', {
    'tl': tuple[float, float],
    'tr': tuple[float, float],
    'bl': tuple[float, float],
    'br': tuple[float, float],
})

EntityTransformed = TypedDict('EntityTransformed', {
    'all': SubEntityTransformed,
    'real': SubEntityTransformed,
})

TableEntry = TypedDict('TableEntry', {
    'x': float,
    'y': float,
    'people': list[tuple[float, float]],
    'items': list[tuple[float, float]]
})


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


def y_intercept(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return y1 - x1 * (y2 - y1) / (x2 - x1)


def pad_image(image, t, b, l, r):
    return cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value=(0, 0, 0))


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
    return cv2.perspectiveTransform(_points, mat)[0]


def draw_bb(image, bbox, color=(0, 255, 0), thickness=2):
    x, y, w, h = bbox
    top_left = (x, y)
    bottom_right = (x + w, y + h)
    image_with_box = cv2.rectangle(image, top_left, bottom_right, color, thickness)
    return image_with_box


def draw_poly(image, points, color=(0, 255, 0), thickness=2):
    pts = np.array(points, dtype=int).reshape((-1, 1, 2))
    return cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)


def save_obj(obj, filename: str):
    with open(filename, mode='wb') as f:
        pickle.dump(obj, f)


def load_obj(filename: str):
    with open(filename, mode='rb') as f:
        return pickle.load(f)


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0


def coverage(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    if x1_ >= x1 and y1_ >= y1 and x2_ <= x2 and y2_ <= y2:
        return True

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box2_area = (x2_ - x1_) * (y2_ - y1_)

    return inter_area / box2_area


def merge_boxes(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    merged_x1 = min(x1, x1_)
    merged_y1 = min(y1, y1_)
    merged_x2 = max(x2, x2_)
    merged_y2 = max(y2, y2_)

    return [merged_x1, merged_y1, merged_x2, merged_y2]


def merge_overlapped(bboxes, threshold=0.8):
    merged_boxes = []

    bboxes = sorted(bboxes, key=lambda x: (x['pos1'][0], x['pos1'][1]))

    while bboxes:
        box = bboxes.pop(0)
        bbox = (*box['pos1'], *box['pos2'])
        merged = False

        for i, merged_box in enumerate(merged_boxes):
            mbox = (*merged_box['pos1'], *merged_box['pos2'])
            if (coverage(mbox, bbox) > threshold
                    or coverage(bbox, mbox) > threshold):
                obox = merge_boxes(bbox, mbox)
                merged_boxes[i] = {
                    **{k: box[k] for k in box},
                    'pos1': tuple(obox[0:2]),
                    'pos2': tuple(obox[2:4]),
                }
                merged = True
                break

        if not merged:
            merged_boxes.append(box)

    return merged_boxes


def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def clamp_normalize(tup):
    x = max(0, min(1, tup[0]))
    y = max(0, min(1, tup[1]))
    return x, y


def scale_tup(tup, scale_x, scale_y):
    return tup[0] * scale_x, tup[1] * scale_y

def offset_tup(tup, offset_x, offset_y):
    return tup[0] + offset_x, tup[1] + offset_y

def add_tup(tup1, tup2):
    return tup1[0] + tup2[0], tup1[1] + tup2[1]

def p_center(tl, tr, bl, br):
    return line_intersection(tl, br, bl, tr)

__all__ = [
    'save_obj',
    'load_obj',
    'tup_int',
    'pad_image',
    'line_intersection',
    'get_perspective_tf_mat',
    'perspective_tf_image',
    'perspective_tf_points',
    'draw_bb',
    'draw_poly',
    'iou',
    'coverage',
    'merge_boxes',
    'merge_overlapped',
    'EntityEntry',
    'EntityTransformed',
    'TableEntry',
    'euclidean_distance',
    'clamp_normalize',
    'scale_tup',
    'offset_tup',
    'add_tup',
    'p_center'
]
