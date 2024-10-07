import math
import queue
import threading
import time
import cv2
from lib_network import *

cap = find_network_cam(username='admin', password='admin')
target_fps = 1
q = queue.Queue()
stopped = threading.Event()


def receive():
    prev = 0

    try:
        while not stopped.is_set():
            time_elapsed = time.time() - prev
            ret, frame = cap.read()
            if not ret:
                print('Error: Unable to read frame from stream')
                break

            if time_elapsed > 1. / target_fps:
                prev = time.time()
                q.put(frame)
                print('Put')
    except Exception as e:
        print(f"Error in receive thread: {e}")
    finally:
        print("Releasing camera...")
        cap.release()


def display():
    cv2.namedWindow('Image')
    cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Start Displaying")
    while not stopped.is_set():
        try:
            if not q.empty():
                fr = q.get(timeout=1)
                print('Get')
                cv2.imshow("Image", fr)
        except queue.Empty:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stopped.set()  # Stop both threads if 'q' is pressed
            break

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
        stopped.set()  # Signal threads to stop

    finally:
        p1.join()
        p2.join()
        print("Program terminated gracefully.")
