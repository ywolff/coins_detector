from .circular_hough_detector import CircularHoughDetector
from .yolo_detector import YoloDetector

coins_detectors = {
    'circular_hough': CircularHoughDetector,
    'yolo': YoloDetector,
}
