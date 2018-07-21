from copy import deepcopy
import numpy as np

MATCHING_MAX_DISTANCE_THRESHOLD = 100


def match_two_coins_lists(first_coins_list, second_coins_list):
    """ Match two list of coins """
    matched_coins = {
        'matched': [],
        'first_list_isolated': [],
        'second_list_isolated': []
    }

    # We deep copy the two lists so that we can add the 'matched' keys to their elements
    first_coins_list = deepcopy(first_coins_list)
    second_coins_list = deepcopy(second_coins_list)

    distances = build_distance_matrix(first_coins_list, second_coins_list)

    for _ in range(min(len(first_coins_list), len(second_coins_list))):
        # If there is no centers close enough to each other, we break the loop
        minimal_distance = np.amin(distances)
        if minimal_distance > MATCHING_MAX_DISTANCE_THRESHOLD:
            break

        # Extract indices of minimal value and match corresponding coins
        first_coin_index, second_coin_index = np.unravel_index(
            np.argmin(distances, axis=None), distances.shape
        )

        matched_coins['matched'].append((
            first_coins_list[first_coin_index],
            second_coins_list[second_coin_index]
        ))

        # Mark coins as matched, so we can extract easily the list of unmatched coins
        first_coins_list[first_coin_index]['matched'] = True
        second_coins_list[second_coin_index]['matched'] = True

        # Replace distance value with High values so it can't be picked anymore
        distances[:, second_coin_index] = np.Infinity
        distances[first_coin_index, :] = np.Infinity

    # List unmatched coins from both lists
    matched_coins['first_list_isolated'] = [
        coin for coin in first_coins_list if not is_coin_matched(coin)
    ]
    matched_coins['second_list_isolated'] = [
        coin for coin in second_coins_list if not is_coin_matched(coin)
    ]

    return matched_coins


def is_coin_matched(coin):
    if 'matched' not in coin.keys():
        return False

    return coin['matched']


def compute_distance_between_coins(coin_1, coin_2):
    """ Compute euclidian distance between two centers """
    center_1 = np.array([coin_1['center_x'], coin_1['center_y']], dtype=np.float32)
    center_2 = np.array([coin_2['center_x'], coin_2['center_y']], dtype=np.float32)

    distance = np.linalg.norm(center_1 - center_2)

    return distance


def build_distance_matrix(first_coins_list, second_coins_list):
    """ Compute a matrix of distances between centers of coins from two lists """
    distances = np.ones((len(first_coins_list), len(second_coins_list))) * np.Infinity
    for first_idx, first_coin in enumerate(first_coins_list):
        for second_idx, second_coin in enumerate(second_coins_list):
            distance = compute_distance_between_coins(first_coin, second_coin)
            distances[first_idx][second_idx] = distance

    return distances
