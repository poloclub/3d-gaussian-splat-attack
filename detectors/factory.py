from detectors.detectron2_detector import Detectron2Detector
#from detectors.yolov8_detector import Yolov8Detector
# from detectors.rtdetr_detector import RTDETRDetector
# etc.

def load_detector(cfg):
    backend = cfg.scene.detector_name
    if backend == "detectron2":
        return Detectron2Detector(cfg)
    # elif backend == "yolov8":
    #     return Yolov8Detector(cfg)
    # elif backend == "rtdetr":
    #     return RTDETRDetector(cfg)
    else:
        raise ValueError(f"Unsupported detection backend: {backend}")
    