""" All parameters with their default values """

PARAMETERS = {
    # Yolo
    #######

    'yolo_weights_file': 'ep417-loss53.286-val_loss45.863.h5',
    'yolo_classes_file': 'yolo_classes_07092018.txt',


    # Hough Circles
    ################

    'hough_median_blur_aperture_size': 5,
    # Aperture linear size for median blur preprocessing used before Circular Hough.
    # It must be odd and greater than 1.

    'hough_circles_dp': 1,
    # Inverse ratio of the accumulator resolution to the image resolution.
    # For example, if dp=1 , the accumulator has the same resolution as the input image.
    # If dp=2 , the accumulator has half as big width and height

    'hough_circles_min_dist': 50,
    # Minimum distance between the centers of the detected circles.
    # If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one.
    # If it is too large, some circles may be missed.

    'hough_circles_param1': 200,
    # It is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).

    'hough_circles_param2': 30,
    # It is the accumulator threshold for the circle centers at the detection stage.
    # The smaller it is, the more false circles may be detected.
    # Circles, corresponding to the larger accumulator values, will be returned first.

    'hough_circles_min_radius': 10,
    # Minimum circle radius.

    'hough_circles_max_radius': 0,
    # Maximum circle radius.
}
