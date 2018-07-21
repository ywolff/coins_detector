import cv2
import numpy as np

from src.constants.coins import COINS_DIAMETERS
from .coins_detector_abc import CoinsDetector


class CircularHoughDetector(CoinsDetector):
    name = 'Circular Hough Transform'

    def detect(self, image_path, biggest_radius_coin_value=None):
        assert biggest_radius_coin_value is not None
        preprocessed_image = self.preprocess(image_path)
        circles = self.compute_circles(preprocessed_image, self.parameters)
        coins = self.get_coins_from_circles(circles, biggest_radius_coin_value)

        return coins

    def preprocess(self, image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.medianBlur(gray_image, self.parameters['hough_median_blur_aperture_size'])

        return blurred_image

    @staticmethod
    def compute_circles(image, parameters):
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp=parameters['hough_circles_dp'],
            minDist=parameters['hough_circles_min_dist'],
            param1=parameters['hough_circles_param1'],
            param2=parameters['hough_circles_param2'],
            minRadius=parameters['hough_circles_min_radius'],
            maxRadius=parameters['hough_circles_max_radius']
        )

        return np.uint16(np.around(circles)) if circles is not None else None

    @classmethod
    def get_coins_from_circles(cls, circles, biggest_radius_coin_value):
        coins = []
        biggest_radius_in_px = 0

        if circles is not None:
            for center_x, center_y, radius in circles[0]:
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
            coin['value'] = cls.get_coin_value_from_circle_radius_in_mm(coin['radius'] / px_per_mm)

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
