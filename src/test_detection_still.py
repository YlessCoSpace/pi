import json

import cv2
import numpy as np
from ultralytics import YOLO
from src.lib_transform import *
from src.lib_transform import scale_tup

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


def two_stage_det(image, model: YOLO) -> dict[str, list[EntityEntry]]:
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

    _out: dict[str, list[EntityEntry]] = {
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


def transform_normalize_sort(data: dict[str, list[EntityEntry]], mat) -> dict[str, list[EntityTransformed]]:
    _out: dict[str, list[EntityTransformed]] = {}

    _min_x, _min_y = float('inf'), float('inf')
    _max_x, _max_y = float('-inf'), float('-inf')

    _loc_min_x, _loc_min_y = float('inf'), float('inf')
    _loc_max_x, _loc_max_y = float('-inf'), float('-inf')

    for c in data:
        for instance in data[c]:
            # Pos
            x1, y1 = w * instance['pos1'][0], h * instance['pos1'][1]
            x2, y2 = w * instance['pos2'][0], h * instance['pos2'][1]

            # Warp transform
            tl = tuple(perspective_tf_points([(x1, y1)], mat)[0])
            tr = tuple(perspective_tf_points([(x2, y1)], mat)[0])
            bl = tuple(perspective_tf_points([(x1, y2)], mat)[0])
            br = tuple(perspective_tf_points([(x2, y2)], mat)[0])

            # To normalize
            _min_x = min(_min_x, tl[0], tr[0], bl[0], br[0])
            _max_x = max(_max_x, tl[0], tr[0], bl[0], br[0])
            _min_y = min(_min_y, tl[1], tr[1], bl[1], br[1])
            _max_y = max(_max_x, tl[1], tr[1], bl[1], br[1])

            _loc_min_x = min(_loc_min_x, (bl[0] + br[0]) / 2)
            _loc_max_x = max(_loc_max_x, (bl[0] + br[0]) / 2)
            _loc_min_y = min(_loc_min_y, (bl[1] + br[1]) / 2)
            _loc_max_y = max(_loc_max_x, (bl[1] + br[1]) / 2)

            if c not in _out:
                _out[c] = []
            _out[c].append({
                'all': {'tl': tl, 'tr': tr, 'bl': bl, 'br': br},
                'real': {'tl': tl, 'tr': tr, 'bl': bl, 'br': br}
            })

    _wx = _max_x - _min_x
    _wy = _max_y - _min_y

    _loc_wx = _loc_max_x - _loc_min_x
    _loc_wy = _loc_max_y - _loc_min_y

    for c in _out:
        for j, instance in enumerate(_out[c]):
            tl = scale_tup(offset_tup(instance['all']['tl'], -_min_x, -_min_y), 1 / _wx, 1 / _wy)
            tr = scale_tup(offset_tup(instance['all']['tr'], -_min_x, -_min_y), 1 / _wx, 1 / _wy)
            bl = scale_tup(offset_tup(instance['all']['bl'], -_min_x, -_min_y), 1 / _wx, 1 / _wy)
            br = scale_tup(offset_tup(instance['all']['br'], -_min_x, -_min_y), 1 / _wx, 1 / _wy)

            _out[c][j]['all'] = {
                'tl': clamp_normalize(tl),
                'tr': clamp_normalize(tr),
                'bl': clamp_normalize(bl),
                'br': clamp_normalize(br),
            }

            tl = scale_tup(offset_tup(instance['real']['tl'], -_loc_min_x, -_loc_min_y), 1 / _loc_wx, 1 / _loc_wy)
            tr = scale_tup(offset_tup(instance['real']['tr'], -_loc_min_x, -_loc_min_y), 1 / _loc_wx, 1 / _loc_wy)
            bl = scale_tup(offset_tup(instance['real']['bl'], -_loc_min_x, -_loc_min_y), 1 / _loc_wx, 1 / _loc_wy)
            br = scale_tup(offset_tup(instance['real']['br'], -_loc_min_x, -_loc_min_y), 1 / _loc_wx, 1 / _loc_wy)

            _out[c][j]['real'] = {
                'tl': clamp_normalize(tl),
                'tr': clamp_normalize(tr),
                'bl': clamp_normalize(bl),
                'br': clamp_normalize(br),
            }
        _out[c].sort(
            key=lambda x: p_center(x['real']['tl'], x['real']['tr'], x['real']['bl'], x['real']['br'])
        )

    return _out


def assign_to_table(data: dict[str, list[EntityTransformed]]) -> dict[int, TableEntry]:
    _tables = data['dining table']
    _people = data['person']
    _items = data['items']

    _out: dict[int, TableEntry] = {}

    for index, table in enumerate(_tables):
        try:
            _cen = p_center(**table['real'])
            tup_int(_cen)
            _out[index] = {
                'x': _cen[0],
                'y': _cen[1],
                'people': [],
                'items': []
            }
        except OverflowError:
            pass

    table_pos = [p_center(**table['real']) for table in _tables]

    # Compare center bottom of people vs table
    for index_person, person in enumerate(_people):
        p1 = p_center(**person['real'])
        min_dist = float('inf')
        min_idx = None
        for index, table in enumerate(_tables):
            p2 = table_pos[index]
            d = euclidean_distance(p1, p2)
            if d < min_dist:
                min_dist = d
                min_idx = index
        if min_idx in _out:
            try:
                tup_int((p1[0], p1[1]))
                _out[min_idx]['people'].append(p1)
            except OverflowError:
                pass
    for index_item, item in enumerate(_items):
        p1 = p_center(**item['real'])
        min_dist = float('inf')
        min_idx = None
        for index, table in enumerate(_tables):
            p2 = table_pos[index]
            d = euclidean_distance(p1, p2)
            if d < min_dist:
                min_dist = d
                min_idx = index
        if min_idx in _out:
            try:
                tup_int((p1[0], p1[1]))
                _out[min_idx]['items'].append(p1)
            except OverflowError:
                pass

    return _out


def make_payload(data: dict[int, TableEntry]) -> str:
    return json.dumps({'tables': [
        {
            'id': k,
            'x': v['x'],
            'y': v['y'],
            'people': len(v['people']),
            'item': len(v['items']),
            'time': 0
        } for k, v in data.items()
    ]})


if __name__ == '__main__':
    from pprint import pprint

    IM_PATH = 'videos/out/ffmpeg_37.bmp'
    MODEL = YOLO('yolo11n.pt')

    mat = load_obj('perspective_matrix.pkl')
    img = cv2.imread(IM_PATH)
    h, w = img.shape[:2]

    res = two_stage_det(img, MODEL)
    # out = perspective_tf_image(img, mat, (1000, 1000))
    warped = transform_normalize_sort(res, mat)
    assigned = assign_to_table(warped)
    payload = make_payload(assigned)

    out = np.zeros((1000, 1000, 3), dtype=np.uint8)

    for i, c in enumerate(res):
        print(c)
        for j, instance in enumerate(res[c]):
            x1, y1 = scale_tup(instance['pos1'], w, h)
            x2, y2 = scale_tup(instance['pos2'], w, h)
            draw_poly(img, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], color=COLORS[i % len(COLORS)])

    for i, instance in assigned.items():
        try:
            p1 = (instance['x'], instance['y'])
            cen1 = tup_int(scale_tup(p1, 1000, 1000))

            for p2 in instance['items']:
                cen2 = tup_int(scale_tup(p2, 1000, 1000))
                cv2.circle(out, cen2, 8, COLORS[1], -1)
                cv2.line(out, cen1, cen2, COLORS[1], 2)

            for p2 in instance['people']:
                cen2 = tup_int(scale_tup(p2, 1000, 1000))
                cv2.circle(out, cen2, 8, COLORS[2], -1)
                cv2.line(out, cen1, cen2, COLORS[2], 2)

            cv2.circle(out, cen1, 16, COLORS[0], -1)

        except OverflowError:
            pass

    print(payload)

    # results.show()
    cv2.imwrite('in.png', img)
    cv2.imwrite('out.png', out)
    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
