import click
from datetime import datetime
import numpy as np
import os
import json
import sys
sys.path.append('.')

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from src.lib.yolo.yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from src.lib.yolo.yolo3.utils import get_random_data
from src.utils.annotations_utils import get_bounding_box
from src.utils.list_utils import int_list_to_str_list


@click.command()
@click.option('--train_images_path', required=True, help='Path to train images')
@click.option('--val_images_path', required=True, help='Path to val images')
@click.option('--train_annotations_path', required=True, help='Path to train annotations')
@click.option('--val_annotations_path', required=True, help='Path to val annotations')
@click.option('--log_dir', default='logs/yolo', help='Log directory for tensorboard and weights')
@click.option('--batch_size', default=16, help='Size of the batch')
@click.option('--n_epochs', default=50, help='Number of epochs to perform')
@click.option('--load_pretrained', default=False, help='Should start from pretrained model')
@click.option('--weights_path', help='Path to weights')
def train_command(
        train_images_path,
        val_images_path,
        train_annotations_path,
        val_annotations_path,
        log_dir,
        batch_size,
        n_epochs,
        load_pretrained,
        weights_path=None):
    train(
        train_images_path,
        val_images_path,
        train_annotations_path,
        val_annotations_path,
        log_dir,
        batch_size,
        n_epochs,
        load_pretrained,
        weights_path
    )


def train(train_images_path,
          val_images_path,
          train_annotations_path,
          val_annotations_path,
          log_dir,
          batch_size,
          n_epochs,
          load_pretrained,
          weights_path=None):
    now = datetime.now()
    classes_file_name = 'yolo_classes_' + \
        now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    classes_path = os.path.join('src/weights', classes_file_name)
    yolo_train_annotations_file_name = 'yolo_train_annotations_' + \
        now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    yolo_val_annotations_file_name = 'yolo_val_annotations_' + \
        now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    yolo_train_annotations_path = os.path.join(
        '.', yolo_train_annotations_file_name)
    yolo_val_annotations_path = os.path.join(
        '.', yolo_val_annotations_file_name)

    convert_annotations(
        train_images_path,
        val_images_path,
        train_annotations_path,
        val_annotations_path,
        yolo_train_annotations_file_name,
        yolo_val_annotations_file_name,
        classes_path
    )

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors('src/constants/yolo_anchors.txt')

    input_shape = (416, 416)
    model = create_model(
        input_shape,
        anchors,
        num_classes,
        freeze_body=2,
        weights_path=weights_path,
        load_pretrained=load_pretrained,
    )

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(
        os.path.join(
            log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        period=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1
    )

    with open(yolo_train_annotations_file_name) as yolo_train_annotations_file:
        yolo_train_annotations = yolo_train_annotations_file.readlines()
    with open(yolo_val_annotations_file_name) as yolo_val_annotations_file:
        yolo_val_annotations = yolo_val_annotations_file.readlines()

    n_epochs_general_training = 5

    model.compile(
        optimizer=Adam(lr=1e-3),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred}
    )

    model.fit_generator(
        data_generator_wrapper(
            yolo_train_annotations, batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, len(yolo_train_annotations) // batch_size),
        validation_data=data_generator_wrapper(
            yolo_val_annotations, batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, len(yolo_val_annotations) // batch_size),
        epochs=n_epochs_general_training,
        initial_epoch=0,
        callbacks=[logging, checkpoint]
    )

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    model.compile(
        optimizer=Adam(lr=1e-4),
        loss={'yolo_loss': lambda y_true, y_pred: y_pred}
    )
    print('Unfreeze all of the layers.')

    model.fit_generator(
        data_generator_wrapper(
            yolo_train_annotations, batch_size, input_shape, anchors, num_classes),
        steps_per_epoch=max(1, len(yolo_train_annotations) // batch_size),
        validation_data=data_generator_wrapper(
            yolo_val_annotations, batch_size, input_shape, anchors, num_classes),
        validation_steps=max(1, len(yolo_val_annotations) // batch_size),
        epochs=n_epochs,
        initial_epoch=n_epochs_general_training + 1,
        callbacks=[logging, checkpoint, reduce_lr]
    )


def convert_annotations(train_images_path,
                        val_images_path,
                        train_annotations_path,
                        val_annotations_path,
                        yolo_train_annotations_output_path,
                        yolo_val_annotations_output_path,
                        classes_path):
    """ Merge annotation files (json) from a folder into appropriate
    annotations files for YOLO training. These files looks like:
    image_id, x0,y0,x1,y1,class_id_1 x0,y0,x1,y1,class_id_2

    # Arguments
        - {train|val}_images_path: path to the folder containing the input images for the {training|validation} set
        - {train|val}_annotations_path: path to the file containing the labels of the images
            for the {training|validation} set (contents and containers on the same line)
        - yolo_{train|val}_annotations_path: path where this script will write the yolo-formatted annotations
        - classes_path: path to the file containing the yolo classes that the training will output
    # Returns: none
    """
    annotations_table = {}
    annotations_reverse_table = {}
    annotation_index = 0

    annotations_final = {}
    for images_path, annotations_path, yolo_dataset_annotations_output_path in [
        (train_images_path, train_annotations_path,
            yolo_train_annotations_output_path),
        (val_images_path, val_annotations_path,
            yolo_val_annotations_output_path),
    ]:
        with open(yolo_dataset_annotations_output_path, 'a') as yolo_dataset_annotations_output_file:
            for file_name in sorted(os.listdir(annotations_path)):
                print(file_name)
                if '.json' not in file_name:
                    continue

                image_annotations_json = json.load(
                    open(os.path.join(annotations_path, file_name)))
                annotated_containers = image_annotations_json['annotation']['object']
            formatted_annotations = []
            for container in annotated_containers:
                container_name = container['name']
                if container_name not in annotations_table:
                    annotations_table[container_name] = annotation_index
                    annotations_reverse_table[annotation_index] = container_name
                    annotation_index += 1

                container_points = container['polygon']['pt']

                min_x, min_y, max_x, max_y = get_bounding_box(
                    container_points)

                formatted_annotations.append(
                    (min_x, min_y, max_x, max_y, annotations_table[container_name]))

                path_to_image = os.path.join(
                    images_path, os.path.splitext(file_name)[0] + ".jpg")
                annotations_final[path_to_image] = formatted_annotations

                image_annotations_strings = [
                    ",".join(int_list_to_str_list(list(annotation)))
                    for annotation in formatted_annotations
                ]
                image_annotations_line = path_to_image + \
                    " " + " ".join(image_annotations_strings)
                yolo_dataset_annotations_output_file.write(
                    image_annotations_line + '\n')

    with open(classes_path, 'a') as f:
        for key in sorted(annotations_reverse_table):
            f.write(annotations_reverse_table[key] + '\n')


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, weights_path, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [
        Input(
            shape=(
                h // {0: 32, 1: 16, 2: 8}[l],
                w // {0: 32, 1: 16, 2: 8}[l],
                num_anchors // 3,
                num_classes + 5
            )
        ) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(
        yolo_loss,
        output_shape=(1,),
        name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5}
    )([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if (n == 0) or (batch_size <= 0):
        return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    train_command()
