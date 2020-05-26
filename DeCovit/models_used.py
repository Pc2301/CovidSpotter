#! /usr/bin/env python
# -*- coding: utf-8 -*-
""" Run a YOLO_v3 style detection model on test images. """
import colorsys
import os
import random
import json
import numpy        as np
import tensorflow   as tf
import cv2

from keras.layers   import Input
from PIL            import Image
from keras          import backend as K
from keras.models   import load_model
from yolo3.model    import yolo_eval
from yolo3.utils    import letterbox_image, image_preprocess
from mobile_net     import backbone
from yolo3.model    import yolo_body, tiny_yolo_body
from retinanet.keras_retinanet import models

def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    (rows, cols, _) = image_shape
    smallest_side   = min(rows, cols)
    scale           = min_side / smallest_side
    largest_side    = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    return scale

def resize_image(img, min_side=800, max_side=1333):
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)
    img   = cv2.resize(img, None, fx=scale, fy=scale)
    return img, scale

class RETINANET(object):
    def __init__(self, backbone='resnet50', filename=None):
        # self.model_path = os.path.join('model_data', filename)
        if filename is None:
            raise ValueError("PROPER FILE NAME IS NOT PROVIDED")

        self.model_path = os.path.join(filename)
        model           = models.load_model(self.model_path, backbone_name=backbone)

        if filename.lower() != "model_data/mask-detector-final.h5":
            print("CONVERTING THE WEIGHTS.")
            model      = models.convert_model(model)
        else:
            print("NOT CONVERTING.")
        self.model      = model
        self.labels_to_names = {0: 'FaceMask', 1: 'No-FaceMask'}
        print("LOADING THE MASK-DETECTOR")

    def detect_mask(self, image, num_detected_person_boxes, detections):
        # image = image * 1.0/127.5
        # image -= 1.
        # image, scale = resize_image(image)
        mask_boxes, scores, mask_labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        mask_boxes  = np.squeeze(mask_boxes)
        scores      = np.squeeze(scores[0])
        mask_labels = np.squeeze(mask_labels[0])
        for i in range(num_detected_person_boxes):
            person_bbox = detections[i].tlwh
            PTL_x = person_bbox[0]
            PTL_y = person_bbox[1]
            PBR_x = PTL_x + person_bbox[2]
            PBR_y = PTL_y + person_bbox[3] / 2

            x1 = mask_boxes[:, 0]
            y1 = mask_boxes[:, 1]
            x2 = mask_boxes[:, 2] # + facial_boxes[:, 0]
            y2 = mask_boxes[:, 3] # + facial_boxes[:, 1]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)

            xx1 = np.maximum(PTL_x, x1)
            yy1 = np.maximum(PTL_y, y1)
            xx2 = np.minimum(PBR_x, x2)
            yy2 = np.minimum(PBR_y, y2)

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h * 1.0) / area
            overlap_idxs = np.argsort(-overlap)
            count = -1
            while count:
                count += 1
                if count > scores.size:
                    break
                max_overlap_idx = overlap_idxs[count]
                face_bbox   =  mask_boxes[max_overlap_idx, :]
                mid_y       = (face_bbox[1] + face_bbox[3])/2
                if (mid_y < PBR_y) and (mid_y > PTL_y):
                    detections[i].face_bbox = mask_boxes[max_overlap_idx, :]
                    if mask_labels[max_overlap_idx] == 1:
                        detections[i].danger_mask = True
                    mask_boxes = np.delete(mask_boxes, max_overlap_idx, axis=0)
                    break
        return detections

class YOLO(object):
    def __init__(self):
        self.model_path        = os.path.join('model_data', 'person_detector.h5')
        # self.mask_path         = os.path.join('model_data', 'mask_detector.h5')
        self.anchors_path      = os.path.join('model_data', 'yolo_anchors.txt')
        self.classes_path      = os.path.join('model_data', 'coco_classes.txt')
        self.mask_classes_path = os.path.join('model_data', 'mask_class_names.txt')
        self.score  = 0.5
        self.iou    = 0.5
        self.class_names        = self._get_class()
        # self.mask_class_names   = self._get_mask_class()
        self.anchors            = self._get_anchors()
        self.sess               = K.get_session()
        self.model_image_size   = (416, 416) # fixed size or (None, None)
        self.is_fixed_size      = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

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
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        print ("LOADING THE YOLO PERSON DETECTOR")
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
        self.yolo_model = load_model(model_path, compile=False)

        hsv_tuples  = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors, len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        image = Image.fromarray(image[..., ::-1])  # bgr to rgb
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size  = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image     = letterbox_image(image, new_image_size)

        image_data  = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data  = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={self.yolo_model.input: image_data,
                                                                      self.input_image_shape: [image.size[1], image.size[0]],
                                                                      K.learning_phase(): 0
                                                                      }
                                                           )
        return_boxs     = []
        return_scores   = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person' :
                continue
            box   = out_boxes[i]
            score = out_scores[i]
            x = int(box[1])
            y = int(box[0])  
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0 
            return_boxs.append([x,y,w,h])
            return_scores.append(score)

        return return_boxs, return_scores

    def close_session(self):
        self.sess.close()

class MobileNet:
    def __init__(self):
        detection_graph, self.category_index = backbone.set_model("ssd_mobilenet_v1_coco_2018_01_28", "mscoco_label_map.pbtxt")
        self.sess               = tf.InteractiveSession(graph=detection_graph)
        self.image_tensor       = detection_graph.get_tensor_by_name("image_tensor:0")
        self.detection_boxes    = detection_graph.get_tensor_by_name("detection_boxes:0")
        self.detection_scores   = detection_graph.get_tensor_by_name("detection_scores:0")
        self.detection_classes  = detection_graph.get_tensor_by_name("detection_classes:0")
        self.num_detections     = detection_graph.get_tensor_by_name("num_detections:0")

    def get_category_index(self):
        return self.category_index

    def detect_image(self, frame):
        input_frame       = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        ####  Actual detection. # input_frame = cv2.resize(frame, (350, 200))
        image_np_expanded = np.expand_dims(input_frame, axis=0)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores,
                                                       self.detection_classes, self.num_detections,
                                                       ], feed_dict={self.image_tensor: image_np_expanded}
                                                      )
        classes = np.squeeze(classes).astype(np.int32)
        boxes   = np.squeeze(boxes)
        scores  = np.squeeze(scores)

        width, height = input_frame.size
        boxes        = np.asarray(boxes)
        boxes[:,1] *= width
        boxes[:,3] *= width
        boxes[:,0] *= height
        boxes[:,2] *= height

        return_boxs   = []
        return_scores = []
        for i in range(int(num[0])):
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]["name"]
                if class_name == "person" :
                    box     = boxes[i]
                    score   = scores[i]
                    x = int(box[1])
                    y = int(box[0])
                    w = int(box[3] - box[1])
                    h = int(box[2] - box[0])
                    if x < 0:
                        w = w + x
                        x = 0
                    if y < 0:
                        h = h + y
                        y = 0
                    return_boxs.append([x, y, w, h])
                    return_scores.append(score)
        return return_boxs, return_scores


class YOLO_MASK(object):
    _defaults = {"model_path": 'model_data/mask_detector.h5',
                 "anchors_path": 'model_data/yolo_anchors.txt',
                 "classes_path": 'model_data/mask_class_names.txt',
                 "score": 0.3, "iou": 0.45, "model_image_size": (416, 416), "text_size": 3,
                 }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names    = self._get_class()
        self.anchors        = self._get_anchors()
        self.sess           = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.is_fixed_size  = self.model_image_size != (None, None)

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
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_mask(self, image):

        # if self.model_image_size != (None, None):
        #     print("ASS")
        #     assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        #     boxed_image = image_preprocess(np.copy(image), tuple(reversed(self.model_image_size)))
        #     image_data  = boxed_image

        width = image.shape[0]
        height =  image.shape[1]
        new_image_size = (width - (width % 32), height - (height % 32))
        boxed_image = image_preprocess(np.copy(image), tuple(reversed(new_image_size)))
        image_data = boxed_image
        out_boxes, out_scores, out_classes = self.sess.run([self.boxes, self.scores, self.classes],
                                                           feed_dict={self.yolo_model.input: image_data,
                                                                      self.input_image_shape: [image_data.shape[0], image_data.shape[1]],
                                                                      K.learning_phase(): 0
                                                                      })
        ObjectsList = []
        for i, c in reversed(list(enumerate(out_classes))):
            box   = out_boxes[i]
            print (box)
            score = out_scores[i]
            # if score < 0.3:
            #     continue
            # predicted_class = self.class_names[c]
            # label  = '{} {:.2f}'.format(predicted_class, score)
            # print(label)
            # scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top    = max(0, np.floor(top + 0.5).astype('int32'))
            left   = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right  = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
            ObjectsList.append([top, left, bottom, right, c])
            # print([top, left, bottom, right])

        ObjectsList = np.asarray(ObjectsList)
        return ObjectsList

    def close_session(self):
        self.sess.close()

    # def detect_img(self, image):
    #     image = cv2.imread(image, cv2.IMREAD_COLOR)
    #     original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    #
    #     r_image, ObjectsList = self.detect_image(original_image_color)
    #     return r_image, ObjectsList