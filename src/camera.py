import cv2
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class IPCamera:
    # Hardcoded connection parameters
    MAX_RETRIES = 5
    RETRY_DELAY = 5
    WARMUP_TIME = 3

    def __init__(self, url: str):
        self.url = url
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_rate = 30.0

    def connect(self) -> bool:
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Attempting to connect to camera (Attempt {attempt + 1}/{self.MAX_RETRIES})")
                self.capture = cv2.VideoCapture(self.url)
                
                if not self.capture.isOpened():
                    raise Exception("Failed to open video capture")
                
                # Wait for the camera to warm up
                logger.info(f"Camera connected. Warming up for {self.WARMUP_TIME} seconds...")
                time.sleep(self.WARMUP_TIME)
                
                # Test if we can get a frame
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    raise Exception("Connected but unable to retrieve a frame")
                
                logger.info("Successfully connected to the camera and received a frame")
                self.is_running = True
                return True
            
            except Exception as e:
                logger.error(f"Failed to connect to camera: {str(e)}")
                if self.capture:
                    self.capture.release()
                    self.capture = None
                
                if attempt < self.MAX_RETRIES - 1:
                    logger.info(f"Retrying in {self.RETRY_DELAY} seconds...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.error("Max retries reached. Unable to connect to the camera.")
        
        return False

    def start(self) -> bool:
        if not self.is_running:
            return self.connect()
        return True

    def stop(self) -> None:
        if self.is_running:
            self.is_running = False
            if self.capture:
                self.capture.release()
                self.capture = None

    def get_frame(self) -> Optional[cv2.Mat]:
        if not self.is_running or self.capture is None:
            return None
        
        ret, frame = self.capture.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None
        
        return frame

    def set_frame_rate(self, frame_rate: float) -> None:
        self.frame_rate = frame_rate
        if self.capture:
            self.capture.set(cv2.CAP_PROP_FPS, self.frame_rate)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()