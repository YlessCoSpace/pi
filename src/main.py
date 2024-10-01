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
    parser = argparse.ArgumentParser(description="IPCamera/Video YOLO Detection Module")
    parser.add_argument("--url", help="RTSP URL of the IP camera")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--model", help="Path to YOLO model")
    parser.add_argument("--classes", nargs="+", type=int, help="Target classes for detection")
    parser.add_argument("--fps", type=int, help="Target frame rate")
    parser.add_argument("--cam-test", action="store_true", help="Run in camera test mode (no YOLO detection)")
    return parser.parse_args()

def get_config(args):
    config = {
        "url": args.url or os.getenv("CAMERA_URL") or "http://10.10.4.34:4747/video",
        "video": args.video or os.getenv("VIDEO_PATH"),
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

def yolo_detection(frame, model: YOLO, target_classes: list):
    results = model.predict(frame, classes=target_classes)
    return results[0]

def process_camera(url: str, model: YOLO, target_classes: list, fps: int):
    logger.info("Processing camera feed")
    camera = IPCamera(url)
    camera.start()
    camera.set_frame_rate(fps)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                logger.error("Unable to read frame from camera")
                break

            result = yolo_detection(frame, model, target_classes)
            annotated_frame = result.plot()
            cv.imshow('YOLO Detection - Camera', annotated_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
    finally:
        camera.stop()
        cv.destroyAllWindows()

def process_video(video_path: str, model: YOLO, target_classes: list):
    logger.info("Processing video file")
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error opening video file")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video file")
                break

            result = yolo_detection(frame, model, target_classes)
            annotated_frame = result.plot()
            cv.imshow('YOLO Detection - Video', annotated_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

def main():
    args = parse_arguments()
    config = get_config(args)
    
    logger.info(f"Configuration: {config}")
    
    if config["cam_test"]:
        if config["video"]:
            logger.warning("Camera test mode is not applicable for video files.")
            return
        camera = IPCamera(config["url"])
        camera_test(camera, config["fps"])
    else:
        model = YOLO("../model/"+config["model"])
        if config["video"]:
            process_video(config["video"], model, config["classes"])
        elif config["url"]:
            process_camera(config["url"], model, config["classes"], config["fps"])
        else:
            logger.error("Neither video file nor camera URL provided.")

if __name__ == '__main__':
    main()