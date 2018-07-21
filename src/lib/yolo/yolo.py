#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
import os

import numpy as np
from keras import backend as K
from keras.layers import Input
from PIL import Image

from .yolo3.model import yolo_eval, yolo_body
from .yolo3.utils import letterbox_image

boxes = None
scores = None
classes = None
yolo_model = None
input_image_shape = None


class YOLO(object):
    def __init__(self, weights_path, anchors_path, classes_path):
        self.model_path = weights_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None), hw

        global boxes, scores, classes, yolo_model, input_image_shape
        if (
            boxes is None or
            scores is None or
            classes is None or
            yolo_model is None or
            input_image_shape is None
        ):
            self.boxes, self.scores, self.classes, self.yolo_model, self.input_image_shape = self.generate()
            boxes = self.boxes
            scores = self.scores
            classes = self.classes
            yolo_model = self.yolo_model
            input_image_shape = self.input_image_shape
        else:
            self.boxes = boxes
            self.scores = scores
            self.classes = classes
            self.yolo_model = yolo_model
            self.input_image_shape = input_image_shape

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        self.yolo_model = yolo_body(
            Input(shape=(None, None, 3)),
            num_anchors // 3,
            num_classes
        )
        self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(
            self.yolo_model.output,
            self.anchors,
            len(self.class_names),
            self.input_image_shape,
            score_threshold=self.score,
            iou_threshold=self.iou
        )

        return boxes, scores, classes, self.yolo_model, self.input_image_shape

    def preprocess(self, image_path):
        image = Image.open(image_path)
        image_data = np.array(letterbox_image(image, tuple(self.model_image_size)), dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)

        return image_data, image.size

    def detect(self, image_path):
        image_data, image_size = self.preprocess(image_path)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image_size[1], image_size[0]],
                K.learning_phase(): 0
            }
        )

        return out_boxes, out_scores, out_classes

    def close_session(self):
        self.sess.close()
