import os

from src.lib.yolo import YOLO
from src.utils.overlapped_detections import filter_overlapping_detections
from .coins_detector_abc import CoinsDetector


class YoloDetector(CoinsDetector):
    name = 'YOLO v3'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = YOLO(
            anchors_path='src/constants/yolo_anchors.txt',
            weights_path=os.path.join('src/weights/', self.parameters['yolo_weights_file']),
            classes_path=os.path.join('src/weights/', self.parameters['yolo_classes_file']),
        )
        self.class_names = self.model._get_class()

    def detect(self, image_path, biggest_radius_coin_value=None):
        detections_boxes, detections_scores, detections_classes = self.model.detect(image_path)

        coins = self.get_coins_from_detections(detections_boxes, detections_scores, detections_classes)
        filtered_coins = filter_overlapping_detections(coins)

        return filtered_coins

    def get_coins_from_detections(self, detections_boxes, detections_scores, detections_classes):
        coins = []
        for bounding_box, score, class_id in zip(detections_boxes, detections_scores, detections_classes):
            coins.append({
                'value': float(self.class_names[class_id]),
                'center_x': (bounding_box[3] + bounding_box[1]) / 2,
                'center_y': (bounding_box[2] + bounding_box[0]) / 2,
                'width': bounding_box[3] - bounding_box[1],
                'height': bounding_box[2] - bounding_box[0],
                'confidence_score': score,
            })

        return coins
