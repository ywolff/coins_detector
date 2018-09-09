import cv2
import numpy as np
import itertools

from scipy.optimize import minimize

from src.constants.coins import COINS_DIAMETERS, SIMILAR_COINS
from .yolo_detector import YoloDetector


class YoloCircularHoughDetectorV2(YoloDetector):
    name = 'YOLO + Circular Hough V2'

    def detect(self, image_path, biggest_radius_coin_value=None):
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
            else:
                all_circles.append(None)

        coins = self.get_coins_from_yolo_coins_and_circles(
            yolo_detected_coins,
            all_circles,
        )

        return coins

    def get_cropped_and_preprocessed_coin_image(self, image_path, coin):
        image = cv2.imread(image_path)

        x_margin = self.parameters['crops_margin_ratio'] * coin['width']
        y_margin = self.parameters['crops_margin_ratio'] * coin['height']

        x1 = max(
            int(round(coin['center_x'] - coin['width'] / 3 - x_margin)), 0)
        x2 = int(round(coin['center_x'] + coin['width'] / 3 + x_margin))
        y1 = max(
            int(round(coin['center_y'] - coin['height'] / 3 - y_margin)), 0)
        y2 = int(round(coin['center_y'] + coin['height'] / 3 + y_margin))

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
    def get_coins_from_yolo_coins_and_circles(cls, yolo_detected_coins, circles):
        coins = []
        possible_values = []

        for yolo_coin, circle in zip(yolo_detected_coins, circles):
            if circle is None:
                coins.append(yolo_coin)
                possible_values.append([yolo_coin['value']])
            else:
                center_x, center_y, radius = circle
                coins.append({
                    'center_x': center_x,
                    'center_y': center_y,
                    'radius': radius,
                })
                possible_values.append(SIMILAR_COINS[yolo_coin['value']])

        best_minimal_error = np.Infinity

        for values_combination in itertools.product(*possible_values):
            def error(px_per_mm):
                return sum([
                    (coin['radius'] - px_per_mm * COINS_DIAMETERS[value] / 2)**2
                    for value, coin in zip(values_combination, coins)
                    if 'radius' in coin
                ])

            minimal_error = minimize(error, 0, method='SLSQP').fun

            if minimal_error < best_minimal_error:
                best_values_combination = values_combination
                best_minimal_error = minimal_error

        for coin, value in zip(coins, best_values_combination):
            coin['value'] = value

        return coins

    @staticmethod
    def get_coin_value_from_circle_radius_in_mm(radius_in_mm, possible_values):
        min_radius_diff = None
        for coin_value in possible_values:
            radius_diff = abs(COINS_DIAMETERS[coin_value] / 2 - radius_in_mm)
            if min_radius_diff is None or radius_diff < min_radius_diff:
                min_radius_diff = radius_diff
                best_coin_value = coin_value
        return best_coin_value
