import cv2
import numpy as np
from ultralytics import YOLO
from lib_transform import *
from src.lib_transform import tup_int

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def get_objects(image, model: YOLO):
    _h, _w = image.shape[:2]
    _res = model.predict(image)[0]
    _ret = {}

    for det in _res.boxes:
        x1, y1, x2, y2 = det.xyxy[0].float().tolist()
        conf = det.conf.item()
        cls = det.cls.item()
        cls_name = model.names[int(cls)]

        if cls_name not in _ret:
            _ret[cls_name] = []
        _ret[cls_name].append({
            'pos1': (x1 / _w, y1 / _h),
            'pos2': (x2 / _w, y2 / _h),
            'conf': conf
        })

    return _ret, _res


if __name__ == '__main__':
    IM_PATH = 'videos/out/ffmpeg_1.bmp'
    MODEL = YOLO('yolo11n.pt')

    mat = load_mat('perspective_matrix.pkl')
    img = cv2.imread(IM_PATH)
    h, w = img.shape[:2]

    res, results = get_objects(img, MODEL)
    out = perspective_tf_image(img, mat, (1000, 1000))

    for i, c in enumerate(res):
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
