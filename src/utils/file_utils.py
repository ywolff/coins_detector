import os


def remove_file_if_exists(output_path):
    """ Remove file if it exists """
    does_file_exist = os.path.isfile(output_path)
    if does_file_exist:
        os.remove(output_path)
