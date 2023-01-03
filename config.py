#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Inference options

# YOLOv5
__C.inferYOLOV5 = edict()
__C.inferYOLOV5.weight_file = "yolov5x6.pt"  # model.pt path(s)
__C.inferYOLOV5.label_file = "coco128.yaml"  # dataset.yaml path
__C.inferYOLOV5.imgsize = (640, 640)  # inference size (height, width)
__C.inferYOLOV5.conf_thres = 0.25  # confidence threshold
__C.inferYOLOV5.iou_thres = 0.45  # NMS IOU threshold
__C.inferYOLOV5.max_det = 1000  # maximum detections per image
__C.inferYOLOV5.classes = None  # filter by class: --class 0, or --class 0 2 3
__C.inferYOLOV5.agnostic_nms = False  # class-agnostic NMS
__C.inferYOLOV5.hide_labels = False  # hide labels
__C.inferYOLOV5.hide_conf = False  # hide confidences
__C.inferYOLOV5.half = False  # use FP16 half-precision inference
__C.inferYOLOV5.dnn = False  # use OpenCV DNN for ONNX inference


# overlay Flags
__C.flags = edict()
__C.flags.image_show = False
__C.flags.video_write = False


# video inference option

__C.video = edict()
__C.video.output_folder = './output/'
__C.video.video_writer_fps = 30
__C.video.FOURCC = 'MP4V'  # 4-byte code used to specify the video codec
