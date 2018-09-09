from .circular_hough_detector import CircularHoughDetector
from .yolo_detector import YoloDetector
from .yolo_circular_hough_detector import YoloCircularHoughDetector

coins_detectors = {
    'circular_hough': CircularHoughDetector,
    'yolo': YoloDetector,
    'yolo_circular_hough': YoloCircularHoughDetector,
}
