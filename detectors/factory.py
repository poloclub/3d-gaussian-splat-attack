from detectors.detectron2_detector import Detectron2Detector
from detectors.yolov8_detector import Yolov8Detector
from detectors.yolov3_detector import Yolov3Detector
from detectors.yolov5_detector import Yolov5Detector
# from detectors.rtdetr_detector import RTDETRDetector
# etc.

def load_detector(cfg):
    backend = cfg.scene.detector_name
    if backend == "detectron2":
        return Detectron2Detector(cfg)
    elif backend == "yolov3":
        return Yolov3Detector(cfg)
    elif backend == "yolov5":
        return Yolov5Detector(cfg)        
    elif backend == "yolov8":
        return Yolov8Detector(cfg)
    # elif backend == "rtdetr":
    #     return RTDETRDetector(cfg)
    else:
        raise ValueError(f"Unsupported detection backend: {backend}")
    