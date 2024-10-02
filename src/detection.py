import cv2
import numpy as np
from ultralytics import YOLO

# todo: tracking model required????????

def detect_room_floor(image):
    # todo: NOT WORKING
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            return np.squeeze(approx)
    return None


def get_objects(image, model: YOLO):
    results = model.predict(image)[0]
    ret = {}

    for det in results.boxes:
        x1, y1, x2, y2 = det.xyxy[0].float().tolist()
        conf = det.conf.item()
        cls = det.cls.item()
        cls_name = model.names[int(cls)]

        if cls_name not in ret:
            ret[cls_name] = []
        ret[cls_name].append(((x1, y1, x2, y2), conf))

    return ret, results


def draw_bb(image, points):
    pts = points.reshape((-1, 1, 2))
    return cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)


if __name__ == '__main__':
    IM_PATH = '../videos/out/ffmpeg_1.bmp'
    MODEL = YOLO('yolo11n.pt')
    # print(model)
    img = cv2.imread(IM_PATH)
    # img = cv2.resize(img, (0, 0), fx=2, fy=2)

    # points = detect_room_floor(img)
    # print(points)
    #
    # img = draw_bb(img, points)
    #
    # cv2.imshow("Output", img)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    res, results = get_objects(img, MODEL)
    for k in res:
        print(f'Class: {k}')
        for instance in res[k]:
            print(f'    {instance}')
    results.show()
