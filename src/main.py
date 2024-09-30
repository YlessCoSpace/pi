import argparse
import logging
import os
import cv2 as cv
from ultralytics import YOLO
from camera import IPCamera
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging from ultralytics
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def parse_arguments():
    parser = argparse.ArgumentParser(description="IPCamera YOLO Detection Module")
    parser.add_argument("--url", help="RTSP URL of the IP camera")
    parser.add_argument("--model", help="Path to YOLO model")
    parser.add_argument("--classes", nargs="+", type=int, help="Target classes for detection")
    parser.add_argument("--fps", type=int, help="Target frame rate")
    parser.add_argument("--cam-test", action="store_true", help="Run in camera test mode (no YOLO detection)")
    return parser.parse_args()

def get_config(args):
    config = {
        "url": args.url or os.getenv("CAMERA_URL") or "http://10.10.4.34:4747/video",
        "model": args.model or os.getenv("YOLO_MODEL") or "yolov8n.pt",
        "classes": args.classes or [int(c) for c in os.getenv("TARGET_CLASSES", "0 60").split()] if os.getenv("TARGET_CLASSES") else [0, 60],
        "fps": args.fps or int(os.getenv("TARGET_FPS", "30")),
        "cam_test": args.cam_test
    }
    return config

def camera_test(camera: IPCamera, fps: int):
    logger.info("Running in camera test mode")
    camera.start()
    camera.set_frame_rate(fps)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.error("Unable to read frame from stream")
                break

            cv.imshow('Camera Test', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
    finally:
        camera.stop()
        cv.destroyAllWindows()

def yolo_detection(camera: IPCamera, model: YOLO, target_classes: list, fps: int):
    logger.info("Running YOLO detection")
    camera.start()
    camera.set_frame_rate(fps)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.error("Unable to read frame from stream")
                break

            results = model.predict(frame, classes=target_classes)
            annotated_frame = results[0].plot()
            cv.imshow('YOLO Detection', annotated_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
    finally:
        camera.stop()
        cv.destroyAllWindows()

def main():
    args = parse_arguments()
    config = get_config(args)
    
    logger.info(f"Configuration: {config}")
    
    camera = IPCamera(config["url"])

    try:
        if config["cam_test"]:
            camera_test(camera, config["fps"])
        else:
            model = YOLO("../model/"+config["model"])
            yolo_detection(camera, model, config["classes"], config["fps"])
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        logger.info("Program finished")

if __name__ == '__main__':
    main()