"""
Run inference on images


Usage - formats:
    $ python path/to/image_inference.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import json
import os
import sys
from pathlib import Path
from config import cfg
import cv2
import numpy as np
import torch
import torchvision

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords, check_img_size


class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


class ModelFileYOLOv5:
    """
    A class to represent the files associated with a model.
    """

    def __init__(self, model_directory):
        self.labelsPath = os.path.join(model_directory, cfg.inferYOLOV5.label_file)
        self.weightsPath = os.path.join(model_directory, cfg.inferYOLOV5.weight_file)


class InferenceYOLOV5:
    def __init__(self, target_gpu_device, model_file, overlay=False):
        self.overlay = overlay
        self.model_file = model_file

        self.device_gpu = target_gpu_device
        self.weight_file = model_file.weightsPath  # model.pt path(s)
        self.data = model_file.labelsPath  # dataset.yaml path
        self.imgsz = cfg.inferYOLOV5.imgsize  # inference size (height, width)
        self.confidence_threshold = cfg.inferYOLOV5.conf_thres  # confidence threshold
        self.nms_iou_threshold = cfg.inferYOLOV5.iou_thres  # NMS IOU threshold
        self.max_detections = cfg.inferYOLOV5.max_det  # maximum detections per image
        self.classes = cfg.inferYOLOV5.classes  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = cfg.inferYOLOV5.agnostic_nms  # class-agnostic NMS
        self.hide_labels = cfg.inferYOLOV5.hide_labels  # hide labels
        self.hide_confidence = cfg.inferYOLOV5.hide_conf  # hide confidences
        self.FP16_half_precision = cfg.inferYOLOV5.half  # use FP16 half-precision inference
        self.dnn = cfg.inferYOLOV5.dnn  # use OpenCV DNN for ONNX inference

        self.colors = Colors()  # create instance for 'from utils.plots import colors'
        # Load model
        self.device = select_device(self.device_gpu)
        self.model = DetectMultiBackend(self.weight_file, device=self.device, dnn=self.dnn, data=self.data,
                                   fp16=self.FP16_half_precision)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    @torch.no_grad()
    def inferYOLOV5(self, img0):

        # img0 = cv2.imread(image)
        # img = self.letterbox(img0, self.imgsz, stride=self.stride, auto=self.pt)[0]

        # Convert
        img = img0.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        # Run inference
        # self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = self.model(im)
        # NMS
        pred = non_max_suppression(pred, self.confidence_threshold, self.nms_iou_threshold, self.classes,
                                        self.agnostic_nms, max_det=self.max_detections)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        bboxes = []
        scores = []
        class_names = []
        class_ids = []

        # Process predictions
        for i, det in enumerate(pred):  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    scores.append(float(f'{conf}'))
                    box = xyxy
                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2]) - x
                    h = int(box[3]) - y
                    bboxes.append([x, y, w, h])
                    c = int(cls)  # integer class
                    class_ids.append(c)
                    label = None if self.hide_labels else (
                        self.names[c] if self.hide_confidence else f'{self.names[c]} {conf:.2f}')
                    class_names.append(self.names[c])
                    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                    color = self.colors(c, True)
                    lw = 3 or max(round(sum(img0.shape) / 2 * 0.003), 2)  # line width
                    if label and self.overlay:
                        cv2.rectangle(img0, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                        tf = max(lw - 1, 1)  # font thickness
                        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                        outside = p1[1] - h >= 3
                        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                        cv2.rectangle(img0, p1, p2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img0,
                                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                                    0,
                                    lw / 3,
                                    (255, 255, 255),
                                    thickness=tf,
                                    lineType=cv2.LINE_AA)
        if self.overlay:
            cv2.imwrite('out.jpg', img0)

        return bboxes, scores, class_names, class_ids, img0

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


if __name__ == "__main__":
    image = 'test_img_1.jpg'
    img0 = cv2.imread(image)
    model_directory = './model_data'
    model_files = ModelFileYOLOv5(model_directory)
    target_gpu_device = 'cpu'
    inference = InferenceYOLOV5(target_gpu_device, model_files, overlay=True)
    bboxes, scores, class_names, class_ids = inference.inferYOLOV5(img0)
    output_data = {'bboxes': bboxes, 'scores': scores, 'class_names': class_names, 'class_ids': class_ids}
    print(json.dumps(output_data, indent=4))
