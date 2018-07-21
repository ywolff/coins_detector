import abc

import cv2

from src.parameters import PARAMETERS
from src.constants.coins import COINS_LEGEND_COLORS
from src.utils.formatters import RGB_to_BGR
from src.utils.file_utils import remove_file_if_exists


class CoinsDetector(abc.ABC):
    def __init__(self, parameters=PARAMETERS):
        self.parameters = parameters

    @abc.abstractproperty
    def name():
        pass

    @abc.abstractmethod
    def detect(self, image_path, biggest_radius_coin_value=None):
        """ Detect coins on an image

        # Arguments
        - image_path: Path to input image

        # Returns
            A list of detected coins, of shape:
            [
                {
                    'value': float (ex: 0.1 for 10 cents),
                    'center_x': int (in pixels),
                    'center_y': int (in pixels),
                    'radius': int (in pixels, optional),
                    'height': int (in pixels, optional),
                    'width': int (in pixels, optional),
                }
            ]
        """
        pass

    def detect_and_visualize(self, image_path, output_image_path, biggest_radius_coin_value=None):
        """ Same as detect method while creating an image with visual result

        # Arguments
        - image_path: Path to input image
        - output_image_path: Path to output image which will be created

        # Returns
            A list of detected coins
        """

        output_image = cv2.imread(image_path)
        detected_coins = self.detect(image_path, biggest_radius_coin_value=biggest_radius_coin_value)
        for coin in detected_coins:
            coin_value = coin['value']
            coin_legend_color = COINS_LEGEND_COLORS[coin_value]
            center_x = coin['center_x']
            center_y = coin['center_y']

            if 'radius' in coin:
                # draw the outer circle
                cv2.circle(
                    output_image,
                    (center_x, center_y),
                    coin['radius'],
                    RGB_to_BGR(coin_legend_color),
                    3,
                )
                # draw the center of the circle
                cv2.circle(output_image, (center_x, center_y), 2, RGB_to_BGR(coin_legend_color), 3)

            elif 'width' in coin and 'height' in coin:
                x1 = center_x - coin['width'] / 2
                x2 = x1 + coin['width']
                y1 = center_y - coin['height'] / 2
                y2 = y1 + coin['height']
                cv2.rectangle(
                    output_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    RGB_to_BGR(coin_legend_color),
                    3
                )

            else:
                cv2.rectangle(
                    output_image,
                    (center_x - 15, center_y - 15),
                    (center_x + 15, center_y + 15),
                    RGB_to_BGR(coin_legend_color),
                    -5
                )
                cv2.rectangle(
                    output_image,
                    (center_x - 100, center_y - 100),
                    (center_x + 100, center_y + 100),
                    RGB_to_BGR(coin_legend_color),
                    3
                )

        remove_file_if_exists(output_image_path)
        cv2.imwrite(output_image_path, output_image)

        return detected_coins
