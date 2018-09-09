import cv2
import numpy as np

from src.constants.coins import COINS_DIAMETERS
from .yolo_detector import YoloDetector


class YoloCircularHoughDetector(YoloDetector):
    name = 'YOLO v3 + Circular Hough'

    def detect(self, image_path, biggest_radius_coin_value=None):
        assert biggest_radius_coin_value is not None

        yolo_detected_coins = super().detect(image_path)

        all_circles = []

        for yolo_coin in yolo_detected_coins:
            cropped_image, x1, y1 = self.get_cropped_and_preprocessed_coin_image(
                image_path, yolo_coin)
            circles_in_cropped_image = self.compute_circles(cropped_image)
            if len(circles_in_cropped_image) > 0:
                main_circle_in_cropped_image = circles_in_cropped_image[0]
                circle_in_initial_image = self.compute_circle_in_initial_image(
                    main_circle_in_cropped_image, x1, y1)
                all_circles.append(circle_in_initial_image)

        coins = self.get_coins_from_circles(
            all_circles, biggest_radius_coin_value)

        return coins

    def get_cropped_and_preprocessed_coin_image(self, image_path, coin):
        image = cv2.imread(image_path)

        x_margin = self.parameters['crops_margin_ratio'] * coin['width']
        y_margin = self.parameters['crops_margin_ratio'] * coin['height']

        x1 = max(
            int(round(coin['center_x'] - coin['width'] / 2 - x_margin)), 0)
        x2 = int(round(coin['center_x'] + coin['width'] / 2 + x_margin))
        y1 = max(
            int(round(coin['center_y'] - coin['height'] / 2 - y_margin)), 0)
        y2 = int(round(coin['center_y'] + coin['height'] / 2 + y_margin))

        cropped_image = image[y1:y2, x1:x2]

        gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        blurred_cropped_image = cv2.medianBlur(
            gray_cropped_image, self.parameters['yolo_hough_median_blur_aperture_size'])

        return blurred_cropped_image, x1, y1

    def compute_circles(self, image):
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=self.parameters['yolo_hough_circles_dp'],
            minDist=self.parameters['yolo_hough_circles_min_dist'],
            param1=self.parameters['yolo_hough_circles_param1'],
            param2=self.parameters['yolo_hough_circles_param2'],
            minRadius=self.parameters['yolo_hough_circles_min_radius'],
            maxRadius=self.parameters['yolo_hough_circles_max_radius']
        )

        return np.uint16(np.around(circles))[0] if circles is not None else []

    @staticmethod
    def compute_circle_in_initial_image(circle, x1, y1):
        center_x, center_y, radius = circle

        return x1 + center_x, y1 + center_y, radius

    @classmethod
    def get_coins_from_circles(cls, circles, biggest_radius_coin_value):
        coins = []
        biggest_radius_in_px = 0

        for center_x, center_y, radius in circles:
            if radius > biggest_radius_in_px:
                biggest_radius_in_px = radius
            coins.append({
                'center_x': center_x,
                'center_y': center_y,
                'radius': radius,
            })

        biggest_radius_in_mm = COINS_DIAMETERS[biggest_radius_coin_value] / 2
        px_per_mm = biggest_radius_in_px / biggest_radius_in_mm

        for coin in coins:
            coin['value'] = cls.get_coin_value_from_circle_radius_in_mm(
                coin['radius'] / px_per_mm)

        return coins

    @staticmethod
    def get_coin_value_from_circle_radius_in_mm(radius_in_mm):
        min_radius_diff = None
        for coin_value in COINS_DIAMETERS:
            radius_diff = abs(COINS_DIAMETERS[coin_value] / 2 - radius_in_mm)
            if min_radius_diff is None or radius_diff < min_radius_diff:
                min_radius_diff = radius_diff
                best_coin_value = coin_value
        return best_coin_value
