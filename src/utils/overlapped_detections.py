from itertools import combinations


DETECTIONS_OVERLAP_THRESHOLD = 0.5


def filter_overlapping_detections(detections):
    """ For all overlapping detections pairs, remove the one with the smaller confidence score """

    detections_indexes_to_remove = []
    for [(first_index, first_detection), (second_index, second_detection)] in combinations(enumerate(detections), 2):
        if do_detections_overlap(first_detection, second_detection):
            first_confidence_score = first_detection['confidence_score']
            second_confidence_score = second_detection['confidence_score']
            if first_confidence_score < second_confidence_score:
                detections_indexes_to_remove.append(first_index)
            else:
                detections_indexes_to_remove.append(second_index)

    filtered_detections = [
        detection for index, detection in enumerate(detections)
        if index not in detections_indexes_to_remove
    ]

    return filtered_detections


def do_detections_overlap(first_detection, second_detection):
    first_detection_area = get_detection_area(first_detection)
    second_detection_area = get_detection_area(second_detection)
    intersection_area = get_detections_intersection_area(first_detection, second_detection)
    overlap_score = get_detections_overlap_score(intersection_area, first_detection_area, second_detection_area)

    return overlap_score >= DETECTIONS_OVERLAP_THRESHOLD


def get_detection_area(detection):
    return detection['width'] * detection['height']


def get_detections_intersection_area(first_detection, second_detection):
    first_detection_x1 = first_detection['center_x'] - first_detection['width'] / 2
    first_detection_y1 = first_detection['center_y'] - first_detection['height'] / 2
    second_detection_x1 = second_detection['center_x'] - second_detection['width'] / 2
    second_detection_y1 = second_detection['center_y'] - second_detection['height'] / 2
    first_detection_x2 = first_detection_x1 + first_detection['width']
    first_detection_y2 = first_detection_y1 + first_detection['height']
    second_detection_x2 = second_detection_x1 + second_detection['width']
    second_detection_y2 = second_detection_y1 + second_detection['height']

    x_overlap = max(
        0,
        min(first_detection_x2, second_detection_x2) -
        max(first_detection_x1, second_detection_x1)
    )
    y_overlap = max(
        0,
        min(first_detection_y2, second_detection_y2) -
        max(first_detection_y1, second_detection_y1)
    )
    return x_overlap * y_overlap


def get_detections_overlap_score(intersection_area, first_detection_area, second_detection_area):
    return intersection_area / min(first_detection_area, second_detection_area)
