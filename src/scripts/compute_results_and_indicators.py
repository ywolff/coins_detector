import os
import datetime
import sys
sys.path.append(".")

import click
import pandas as pd
import numpy as np
import tqdm

from src.utils.annotations_utils import parse_annotations
from src.utils.coins_matching import match_two_coins_lists
from src.constants.coins import COINS
from src.parameters import PARAMETERS
from src.coins_detectors import coins_detectors


@click.command()
@click.option(
    '--detector_key',
    required=True,
    help='Detector you want to measure',
    type=click.Choice(coins_detectors)
)
@click.option('--images_folder', required=True, help='Folder of input images', type=click.Path(exists=True))
@click.option(
    '--annotations_folder',
    required=True,
    help='Folder of annotations as json files for dataset',
    type=click.Path(exists=True)
)
@click.option(
    '--output_folder',
    default='results',
    help='Folder to save results and indicators files',
    type=click.Path(exists=True)
)
@click.option(
    '--update_evolution_file',
    default=True,
    help='Should update business_indicators_evolution.csv',
    type=bool
)
def compute_results_and_indicators_command(
    detector_key,
    images_folder,
    annotations_folder,
    output_folder,
    update_evolution_file
):
    """A script to compute the results of detections on all images in a folder"""
    compute_results_and_indicators(
        detector_key,
        images_folder,
        annotations_folder,
        output_folder,
        update_evolution_file=update_evolution_file
    )


def compute_results_and_indicators(
    detector_key,
    images_folder,
    annotations_folder,
    output_folder,
    parameters=PARAMETERS,
    update_evolution_file=True,
):
    now = datetime.datetime.now()

    images_names = get_images_names(images_folder)

    DetectorClass = coins_detectors[detector_key]

    detector = DetectorClass(parameters=parameters)

    results = compute_results(
        detector,
        images_names,
        annotations_folder,
        images_folder,
        parameters,
    )
    results_df = build_results_df(results)
    results_df.to_csv(
        get_output_csv_path(output_folder, 'results', now, detector_key),
        index=False
    )

    indicators = compute_indicators_from_results_df(results_df)
    indicators_df = pd.DataFrame(indicators).T
    indicators_df.to_csv(get_output_csv_path(output_folder, 'indicators', now, detector_key))

    business_indicators = compute_business_indicators_from_results_df(results_df)
    if update_evolution_file:
        update_business_indicators_evolution_file(
            output_folder,
            business_indicators,
            now,
            detector_key,
        )

    print('Created files:', flush=True)
    print(get_output_csv_path(output_folder, 'results', now, detector_key), flush=True)
    print(get_output_csv_path(output_folder, 'indicators', now, detector_key), flush=True)
    print(os.path.join(output_folder, 'business_indicators_evolution.csv'), flush=True)


def get_images_names(images_folder):
    images_names = [
        image_name.upper() for image_name, extension
        in [os.path.splitext(file_name) for file_name in sorted(os.listdir(images_folder))]
        if extension == '.jpg'
    ]

    return images_names


def get_output_csv_path(output_folder, name, date, detector_key):

    return os.path.join(output_folder, name + '_' + detector_key + '_' + date.strftime("%Y-%m-%d_%H-%M") + '.csv')


def get_image_path(image_folder, image_name):

    return os.path.join(image_folder, f'{image_name}.jpg').lower()


def compute_results(detector, images_names, annotations_folder, images_folder, parameters, use_warp=False):
    annotations_df = parse_annotations(annotations_folder)

    results = []

    for image_name in tqdm.tqdm(images_names):
        image_path = get_image_path(images_folder, image_name)
        image_annotations_df = annotations_df[annotations_df.image_name == image_name]

        biggest_radius_coin_value = image_annotations_df.value.max()

        detected_coins = detector.detect(image_path, biggest_radius_coin_value=biggest_radius_coin_value)

        annotated_coins = list(image_annotations_df.T.to_dict().values())

        detection_vs_annotation_matching = match_two_coins_lists(annotated_coins, detected_coins)

        image_results = []

        for annotated_coin, predicted_coin in detection_vs_annotation_matching['matched']:
            image_results.append({
                **annotated_coin,
                'predicted_value': predicted_coin['value'],
                'predicted_center_x': predicted_coin['center_x'],
                'predicted_center_y': predicted_coin['center_y'],
            })
        for isolated_annotated_coin in detection_vs_annotation_matching['first_list_isolated']:
            image_results.append(isolated_annotated_coin)
        for isolated_predicted_coin in detection_vs_annotation_matching['second_list_isolated']:
            image_results.append({
                'image_name': image_name,
                'predicted_value': isolated_predicted_coin['value'],
                'predicted_center_x': isolated_predicted_coin['center_x'],
                'predicted_center_y': isolated_predicted_coin['center_y'],
            })

        results.extend(image_results)

    return results


def build_results_df(results):
    results_ordered_columns = [
        'image_name',
        'value',
        'center_x',
        'center_y',
        'predicted_value',
        'predicted_center_x',
        'predicted_center_y',
    ]
    results_df = pd.DataFrame(results)[results_ordered_columns].sort_values(results_ordered_columns)

    return results_df


def compute_indicators_from_results_df(results_df):
    indicators = {}

    for coin_value in COINS:
        number_of_real_coins = len(results_df[results_df.value == coin_value])
        number_of_coins_found = len(results_df[results_df.predicted_value == coin_value])
        number_of_real_coins_found = len(
            results_df[(results_df.value == coin_value) & (results_df.predicted_value == coin_value)]
        )

        precision = (
            100 * number_of_real_coins_found / number_of_coins_found
            if number_of_coins_found else None
        )
        recall = (
            100 * number_of_real_coins_found / number_of_real_coins
            if number_of_real_coins else None
        )

        indicators[coin_value] = {'precision': precision, 'recall': recall}

    return indicators


def compute_business_indicators_from_results_df(results_df):
    grouped_by_image = results_df.groupby('image_name')
    total = len(grouped_by_image)

    grouped_by_image_counts = grouped_by_image.count()
    grouped_by_image_sums = grouped_by_image.sum()

    nb_of_coins_number_errors = len(
        grouped_by_image_counts[grouped_by_image_counts.predicted_value != grouped_by_image_counts.value]
    )
    nb_of_coins_number_or_types_errors = len(
        results_df[results_df.value != results_df.predicted_value].groupby('image_name')
    )
    nb_of_coins_types_errors = nb_of_coins_number_or_types_errors - nb_of_coins_number_errors
    nb_of_sum_errors = len(
        grouped_by_image_sums[grouped_by_image_sums.predicted_value != grouped_by_image_sums.value]
    )
    average_sum_error = abs(grouped_by_image_sums.predicted_value - grouped_by_image_sums.value).mean()

    return {
        'nb_of_coins_number_errors': nb_of_coins_number_errors,
        'nb_of_coins_types_errors': nb_of_coins_types_errors,
        'nb_of_sum_errors': nb_of_sum_errors,
        'average_sum_error': average_sum_error,
        'total': total,
    }


def update_business_indicators_evolution_file(
    output_folder,
    business_indicators,
    now,
    detector_key
):
    if os.path.isfile(os.path.join(output_folder, 'business_indicators_evolution.csv')):
        business_indicators_evolution_df = pd.read_csv(
            os.path.join(output_folder, 'business_indicators_evolution.csv')
        )
    else:
        business_indicators_evolution_df = pd.DataFrame(
            columns=[
                'Date',
                'Algo',
                'Errors in number of coins (%)',
                'Errors in types of coins (%)',
                'Errors in sum (%)',
                'Average sum error (€)',
            ]
        )

    business_indicators_evolution_df = business_indicators_evolution_df.append({
        'Date': now.strftime('%Y-%m-%d'),
        'Algo': detector_key,
        'Errors in number of coins (%)':
            100 * business_indicators['nb_of_coins_number_errors'] / business_indicators['total'],
        'Errors in types of coins (%)':
            100 * business_indicators['nb_of_coins_types_errors'] / business_indicators['total'],
        'Errors in sum (%)':
            100 * business_indicators['nb_of_sum_errors'] / business_indicators['total'],
        'Average sum error (€)':
            business_indicators['average_sum_error'],
    }, ignore_index=True)

    business_indicators_evolution_df.to_csv(
        os.path.join(output_folder, 'business_indicators_evolution.csv'),
        index=False
    )


if __name__ == '__main__':
    compute_results_and_indicators_command()
