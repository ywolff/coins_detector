from .circular_hough_detector import CircularHoughDetector
from .yolo_detector import YoloDetector
from .yolo_circular_hough_detector import YoloCircularHoughDetector
from .yolo_circular_hough_detector_v2 import YoloCircularHoughDetectorV2

coins_detectors = {
    'circular_hough': CircularHoughDetector,
    'yolo': YoloDetector,
    'yolo_circular_hough': YoloCircularHoughDetector,
    'yolo_circular_hough_v2': YoloCircularHoughDetectorV2,
}
