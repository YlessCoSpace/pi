import pprint

import cv2
import numpy as np
from ultralytics import YOLO
from lib_transform import *
from src.lib_transform import *

# https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
ITEM_CLASSES = [
    15, 16, 24, 25, 26, 27, 28, 32,
    38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55,
    58, 63, 64, 65, 66, 67, 68, 73, 75, 76, 77, 78, 79
]
PERSON_CLASS = 0
TABLE_CLASS = 60
COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def two_stage_det(image, model: YOLO):
    def _to_result(__results):
        __ret = {}
        for __res in __results:
            for det in __res.boxes:
                x1, y1, x2, y2 = det.xyxy[0].float().tolist()
                conf = det.conf.item()
                cls = det.cls.item()
                cls_name = model.names[int(cls)]

                if cls_name not in __ret:
                    __ret[cls_name] = []
                __ret[cls_name].append({
                    'pos1': (x1 / _w, y1 / _h),
                    'pos2': (x2 / _w, y2 / _h),
                    'conf': conf
                })
        return __ret

    _h, _w = image.shape[:2]
    _results = [model.predict(image, classes=[PERSON_CLASS], conf=0.25)[0],
                model.predict(image, classes=[TABLE_CLASS], conf=0.05)[0],
                model.predict(image, classes=ITEM_CLASSES, conf=0.1)[0]]

    _ret = _to_result(_results)
    _ret['dining table'] = merge_overlapped(_ret['dining table'], threshold=0.5)

    for _instance in _ret['dining table']:
        _x1, _y1 = int(_instance['pos1'][0] * _w), int(_instance['pos1'][1] * _h)
        _x2, _y2 = int(_instance['pos2'][0] * _w), int(_instance['pos2'][1] * _h)
        cropped = image[_y1:_y2, _x1:_x2]
        __results = model.predict(cropped, classes=ITEM_CLASSES, conf=0.1)[0]
        _ir = _to_result(__results)
        for k in _ir:
            for il in range(len(_ir[k])):
                _ir[k][il]['pos1'] = (_ir[k][il]['pos1'][0] + _x1) / _w, (_ir[k][il]['pos1'][0] + _y1) / _h
                _ir[k][il]['pos2'] = (_ir[k][il]['pos2'][0] + _x1) / _w, (_ir[k][il]['pos2'][0] + _y1) / _h
            if k not in _ret:
                _ret[k] = []
            _ret[k].extend(_ir[k])

    _out = {
        'dining table': _ret['dining table'],
        'person': _ret['person'],
        'items': []
    }

    for _c in _ret:
        if _c == 'dining table' or _c == 'person':
            continue
        _out['items'].extend(_ret[_c])

    _out['items'] = merge_overlapped(_out['items'], threshold=0.5)

    return _out


if __name__ == '__main__':
    IM_PATH = 'videos/out/ffmpeg_1.bmp'
    MODEL = YOLO('yolo11n.pt')

    mat = load_mat('perspective_matrix.pkl')
    img = cv2.imread(IM_PATH)
    h, w = img.shape[:2]

    res = two_stage_det(img, MODEL)
    out = perspective_tf_image(img, mat, (1000, 1000))

    for i, c in enumerate(res):
        print(c)
        for j, instance in enumerate(res[c]):
            x1, y1 = w * instance['pos1'][0], h * instance['pos1'][1]
            x2, y2 = w * instance['pos2'][0], h * instance['pos2'][1]

            tl = perspective_tf_points([(x1, y1)], mat)[0]
            tr = perspective_tf_points([(x2, y1)], mat)[0]
            bl = perspective_tf_points([(x1, y2)], mat)[0]
            br = perspective_tf_points([(x2, y2)], mat)[0]

            print(tl, tr, br, bl)
            draw_poly(img, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], color=COLORS[i % len(COLORS)])
            draw_poly(out, [tl, tr, br, bl], color=COLORS[i % len(COLORS)])

    # results.show()
    cv2.imwrite('in.png', img)
    cv2.imwrite('out.png', out)
    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
