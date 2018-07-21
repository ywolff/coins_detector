import os
import json
import sys
sys.path.append(".")

import pandas as pd
import numpy as np

from src.constants.coins import COINS


def parse_annotations(annotations_dir):
    coins = []

    for image_annotation_file in sorted(os.listdir(annotations_dir)):
        _, extension = os.path.splitext(image_annotation_file)
        if extension != '.json':
            continue
        image_annotation = json.load(open(os.path.join(annotations_dir, image_annotation_file), 'r'))['annotation']
        image_name = image_annotation['filename'].split('.')[0].upper()
        for coin in image_annotation['object']:
            coin_value = float(coin['name'])
            assert coin_value in COINS
            bounding_box = coin['polygon']['pt']
            center_x, center_y = get_center_from_bounding_box(bounding_box)
            annotated_coin = {
                'image_name': image_name,
                'value': coin_value,
                'center_x': center_x,
                'center_y': center_y,
            }
            coins.append(annotated_coin)

    ordered_columns = ['image_name', 'value', 'center_x', 'center_y']

    df = pd.DataFrame(coins)[ordered_columns].sort_values(ordered_columns)

    return df


def get_center_from_bounding_box(bounding_box):
    center_x = np.average([int(pt['x']) for pt in bounding_box])
    center_y = np.average([int(pt['y']) for pt in bounding_box])

    return np.array([center_x, center_y], dtype="float32")


def get_bounding_box(point_list):
    """ Return the bounding box that contain all the polygon points """
    min_x = np.min([int(point['x']) for point in point_list])
    min_y = np.min([int(point['y']) for point in point_list])
    max_x = np.max([int(point['x']) for point in point_list])
    max_y = np.max([int(point['y']) for point in point_list])

    return min_x, min_y, max_x, max_y
