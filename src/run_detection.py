import json
import logging
import math
import queue
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
from lib_transform import *
from lib_network import *
from pprint import pprint

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
BORDER_EPSILON = 0.05


def two_stage_det(image, model: YOLO, more_tables: list[EntityEntry] | None = None) -> dict[str, list[EntityEntry]]:
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

                _p1 = scale_tup((x1, y1), 1 / _w, 1 / _h)
                _p2 = scale_tup((x2, y2), 1 / _w, 1 / _h)
                _cen = scale_tup(add_tup(_p1, _p2), 0.5, 0.5)

                if (BORDER_EPSILON <= _cen[0] <= (1 - BORDER_EPSILON) and
                        BORDER_EPSILON <= _cen[1] <= (1 - BORDER_EPSILON)):
                    __ret[cls_name].append({
                        'pos1': _p1,
                        'pos2': _p2,
                        'conf': conf
                    })
        return __ret

    _h, _w = image.shape[:2]
    _results = [model.predict(image, classes=[PERSON_CLASS, TABLE_CLASS, *ITEM_CLASSES], conf=0.25)[0]]

    _ret = _to_result(_results)

    for k in ['dining table', 'person']:
        if k not in _ret:
            _ret[k] = []

    if more_tables:
        _ret['dining table'].extend(more_tables)
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
        if _c != 'dining table' and _c != 'person':
            _out['items'].extend(_ret[_c])

    _out['items'] = merge_overlapped(_out['items'], threshold=0.5)

    # Filter out small BBs
    for _c in _out:
        _out[_c] = list(filter(lambda x: area((*x['pos1'], *x['pos2'])) >= 0.005 * 0.005, _out[_c]))

    return _out


def add_bb_from_file(image, bb: list[tuple[tuple[int, int], tuple[int, int]]]) -> list[EntityEntry]:
    _h, _w = image.shape[:2]
    _out = []

    for b in bb:
        _out.append({
            'pos1': scale_tup(b[0], 1 / _w, 1 / _h),
            'pos2': scale_tup(b[1], 1 / _w, 1 / _h),
            'conf': 1.0
        })

    return _out


def transform_normalize_sort(data: dict[str, list[EntityEntry]], mat, w, h) -> dict[str, list[EntityTransformed]]:
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
    _tables = data['dining table'] if 'dining table' in data else []
    _people = data['person'] if 'person' in data else []
    _items = data['items'] if 'items' in data else []

    _out: dict[int, TableEntry] = {}

    for index, p_table in enumerate(_tables):
        try:
            _cen = p_center(**p_table['real'])
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
        p1 = p_bottom(**person['real'])
        min_dist = float('inf')
        min_idx = None
        for index, p_table in enumerate(table_pos):
            p2 = p_table
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
        p1 = p_bottom(**item['real'])
        min_dist = float('inf')
        min_idx = None
        for index, p_table in enumerate(table_pos):
            p2 = p_table
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


def make_payload(data: dict[int, TableEntry], max_x=10.0, max_y=10.0, offset=2.5) -> str:
    return json.dumps(
        {
            'tables': [
                {
                    'id': k,
                    'x': v['x'] * max_x + offset,
                    'y': v['y'] * max_y + offset,
                    'people': len(v['people']),
                    'item': bool(len(v['items'])),
                    'time': 0,
                    'startTime': 0
                } for k, v in data.items()
            ],
            'max_x': max_x + 2 * offset,
            'max_y': max_y + 2 * offset
        }
    )


REALTIME = True
cap = find_network_cam(username='admin', password='admin')
target_fps = 0.5
q_input = queue.Queue()
q_output = queue.Queue()

stopped = threading.Event()

MODEL = YOLO('yolo11n.pt')


# logging.getLogger("ultralytics").setLevel(logging.WARNING)


def receive():
    prev = 0
    try:
        while not stopped.is_set():
            time_elapsed = time.time() - prev
            ret, frame = cap.read()
            if not ret:
                break

            if time_elapsed > 1. / target_fps:
                prev = time.time()
                q_input.put(frame)
    except Exception as e:
        print(f"Error in receive thread: {e}")
    finally:
        print("Releasing camera...")
        cap.release()


def process():
    print("Started processing")

    pub = MQTTPublisher('8eaeae364d3c44be8113419a0b9bf948.s1.eu.hivemq.cloud',
                        username='admin',
                        password='Admin123',
                        port=8883)

    # Load weights
    mat = load_obj('perspective_matrix.pkl')
    bb = load_obj('tables.pkl')

    while not stopped.is_set():
        try:
            if not q_input.empty():
                fr = q_input.get(timeout=1)
                # Always get latest frame possible
                while not q_input.empty():
                    fr = q_input.get(timeout=1)
                print('Got frame!')
                # Load image properties
                h, w = fr.shape[:2]

                # Manually add table in addition to automatic table detection
                more = add_bb_from_file(fr, bb)

                # Auto object detection
                res = two_stage_det(cv2.resize(fr, (0, 0), fx=0.5, fy=0.5), MODEL, more_tables=more)

                # Transform coordinates
                warped = transform_normalize_sort(res, mat, w, h)

                # Assign people and items to table
                assigned = assign_to_table(warped)

                # Construct outgoing payload
                payload = make_payload(assigned)
                pub.publish('seatmap', payload)

                # Draw input frame overlays
                for i, c in enumerate(res):
                    color = COLORS[i % len(COLORS)]
                    for j, instance in enumerate(res[c]):
                        x1, y1 = tup_int(scale_tup(instance['pos1'], w, h))
                        x2, y2 = tup_int(scale_tup(instance['pos2'], w, h))
                        draw_poly(fr, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], color=color)

                        text = f"{c}: {j}"
                        text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + 20
                        font_scale = 0.6
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                              font_scale,
                                                                              thickness)
                        cv2.rectangle(fr, (text_x, text_y - text_height - baseline),
                                      (text_x + text_width, text_y + baseline),
                                      color, thickness=cv2.FILLED)
                        cv2.putText(fr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (255 - color[0], 255 - color[1], 255 - color[2]), thickness,
                                    cv2.LINE_AA)

                # Create an empty output frame
                out = np.zeros((1000, 1000, 3), dtype=np.uint8)

                # Draw output frame overlays
                for i, instance in assigned.items():
                    try:
                        p1 = (instance['x'], instance['y'])
                        cen1 = tup_int(scale_tup(p1, 1000, 1000))

                        for p2 in instance['items']:
                            cen2 = tup_int(scale_tup(p2, 1000, 1000))
                            cv2.circle(out, cen2, 8, COLORS[2], -1)
                            cv2.line(out, cen1, cen2, COLORS[2], 2)

                        for p2 in instance['people']:
                            cen2 = tup_int(scale_tup(p2, 1000, 1000))
                            cv2.circle(out, cen2, 8, COLORS[1], -1)
                            cv2.line(out, cen1, cen2, COLORS[1], 2)

                        cv2.circle(out, cen1, 16, COLORS[0], -1)

                    except OverflowError:
                        pass

                scaled_out = cv2.resize(out, (500, 500))
                for c in range(0, 3):
                    fr[0:500, 1420:1920, c] = scaled_out[:, :, c]
                q_output.put(fr)

        except queue.Empty:
            pass

    pub.disconnect()


def display():
    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Started displaying")

    while not stopped.is_set():
        try:
            if not q_output.empty():
                fr = q_output.get(timeout=1)
                cv2.imshow("Image", fr)
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stopped.set()
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':

    try:
        if not cap.isOpened():
            print(f'Error: Unable to open video stream')
            exit()

        p1 = threading.Thread(target=receive)
        p2 = threading.Thread(target=process)
        p3 = threading.Thread(target=display)
        p1.start()
        p2.start()
        p3.start()

        p1.join()
        p2.join()
        p3.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, stopping threads...")
        stopped.set()

    finally:
        p1.join()
        p2.join()
        p3.join()

        print("Program terminated.")
