from typing import List, Dict, Any
from ultralytics import YOLO
import torch as ch
from detectors.base_detector import BaseDetector
from PIL import Image, ImageDraw
from PIL import ImageFont
import torchvision.transforms as T
import numpy as np
from types import SimpleNamespace
from ultralytics.nn.tasks import DetectionModel
import torchvision.transforms.functional as TF
from ultralytics.utils import LOGGER
from torchvision.ops import box_iou
LOGGER.setLevel("WARNING")

class Yolov8Detector(BaseDetector):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None

    def load_model(self):
        import yaml

        cfg_path = "pretrained-models/yolov8/yolov8n.yaml"
        weights_path = "pretrained-models/yolov8/yolov8n.pt"

        with open(cfg_path, "r") as f:
            model_cfg = yaml.safe_load(f)

        model = DetectionModel(model_cfg)
        checkpoint = ch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"].float().state_dict(), strict=False)
        DEVICE = f"cuda:{self.cfg.device}" 
        model.to(DEVICE)
        model.train()
        model.training = True
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, task="detect")
        self.model = model

    def infer(self, x, target, bboxes, batch_size=1):
        if target.ndim == 1 and target.shape[0] == 1 and batch_size > 1:
            target = target.repeat(batch_size)        
        self.model.model.train()
        if isinstance(bboxes, np.ndarray):
            bboxes = ch.tensor(bboxes, dtype=ch.float32)
        if isinstance(target, np.ndarray):
            target = ch.tensor(target, dtype=ch.long)

        bboxes = bboxes.to(x.device)
        target = target.to(x.device)

        # Ensure bbox tensor shape is (B, N, 4) for batch processing
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0).unsqueeze(0)
        elif bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(1)

        # if x.dim() == 3:
        #     x = x.unsqueeze(0)
        # x = x.to(dtype=ch.float32)
        
        # def resize_to_multiple(img, multiple=32):
        #     h, w = img.shape[2], img.shape[3]
        #     new_h = ((h + multiple - 1) // multiple) * multiple
        #     new_w = ((w + multiple - 1) // multiple) * multiple
        #     return ch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # x = resize_to_multiple(x)

        # # Normalize bbox to 0-1 as YOLO expects (x_center, y_center, width, height)
        # height = x.shape[2]
        # width = x.shape[3]
        # boxes = bboxes.clone()
        # boxes[:, :, 0::2] /= width   # x1 and x2
        # boxes[:, :, 1::2] /= height  # y1 and y2
        # xywh = ch.zeros_like(boxes)
        # xywh[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2  # x_center
        # xywh[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2  # y_center
        # xywh[:, :, 2] = boxes[:, :, 2] - boxes[:, :, 0]        # width
        # xywh[:, :, 3] = boxes[:, :, 3] - boxes[:, :, 1]        # height

              # Ensure 3D bbox shape: (B, N, 4)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0).unsqueeze(0)
        elif bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(1)

        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(dtype=ch.float32)
        
        # Letterbox function: resizes while preserving aspect ratio and adds padding.
        def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
            # img: tensor (B, C, H, W)
            B, C, H, W = img.shape
            new_h, new_w = new_shape
            # Compute scale (do not scale up if not necessary)
            scale = min(new_h / H, new_w / W)
            resized_h = int(round(H * scale))
            resized_w = int(round(W * scale))
            # Compute padding
            pad_h = new_h - resized_h
            pad_w = new_w - resized_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # Resize image
            img_resized = ch.nn.functional.interpolate(img, size=(resized_h, resized_w), mode='bilinear', align_corners=False)
            # Pad image; color value is divided by 255 since img is expected to be [0,1]
            img_padded = ch.nn.functional.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=color[0]/255.0)
            return img_padded, scale, pad_left, pad_top

        # Apply letterbox to image tensor x
        new_shape = (640,640)  # Adjust as needed
        x, scale, pad_left, pad_top = letterbox(x, new_shape=new_shape)
        new_w, new_h = new_shape

        # Adjust bounding boxes accordingly.
        # Assume original bboxes are in pixel coordinates relative to the original image.
        boxes = bboxes.clone()
        # Scale boxes by the same factor
        boxes = boxes * ch.tensor([scale, scale, scale, scale], device=boxes.device)
        # Add padding offsets to x coordinates (indices 0 and 2) and y coordinates (indices 1 and 3)
        boxes[:, :, 0::2] += pad_left
        boxes[:, :, 1::2] += pad_top
        # Normalize the coordinates by new image dimensions
        boxes[:, :, 0::2] /= new_w
        boxes[:, :, 1::2] /= new_h

        # Convert from [x1, y1, x2, y2] to YOLO format: [x_center, y_center, width, height]
        xywh = ch.zeros_like(boxes)
        xywh[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2  # x_center
        xywh[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2  # y_center
        xywh[:, :, 2] = boxes[:, :, 2] - boxes[:, :, 0]        # width
        xywh[:, :, 3] = boxes[:, :, 3] - boxes[:, :, 1]        # height
        # output img and bbox here to debug
        losses = []
        for i in range(x.shape[0]):
            targets = ch.cat([target[i].view(-1, 1).float(), xywh[i]], dim=1)
            targets = targets.to(x.device)
            self.model.train()
            self.model.training = True  # Force training-mode behavior
            batch = {
                "img": x[i].unsqueeze(0),
                "cls": target[i].view(-1),
                "bboxes": xywh[i].to(x.device),
                "batch_idx": ch.zeros(target[i].numel(), dtype=ch.int64, device=x.device)
            }
            out = self.model(batch)
            # loss = out[0] if isinstance(out, tuple) else out
            loss = out[1] #out[1][1] 
            losses.append(loss)

        return ch.sum(losses[0]) / losses[0].shape[0]

    def predict_and_save(self, image: ch.Tensor, path: str, target: int = None, untarget: int = None, is_targeted: bool = True, threshold: float = 0.7, format: str = "RGB", gt_bbox: List[int] = None, result_dict: bool = False) -> Any:
        self.model.eval()
        with ch.no_grad():
            image_np = (image.detach().clamp(0,1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            image_pil = Image.fromarray(image_np)
            img_tensor = TF.to_tensor(image_pil).unsqueeze(0).to(next(self.model.parameters()).device)
            img_tensor = img_tensor.to(dtype=ch.float32)

            def resize_to_multiple(img, multiple=32):
                h, w = img.shape[2], img.shape[3]
                new_h = ((h + multiple - 1) // multiple) * multiple
                new_w = ((w + multiple - 1) // multiple) * multiple
                return ch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

            img_tensor = resize_to_multiple(img_tensor)
            orig_h, orig_w = image_np.shape[:2]
            resized_h, resized_w = img_tensor.shape[2], img_tensor.shape[3]
            scale_x = orig_w / resized_w
            scale_y = orig_h / resized_h
            # Lazy-load a YOLO wrapper for prediction (does not touch load_model)
            if not hasattr(self, "yolo_wrapper"):
                self.yolo_wrapper = YOLO("pretrained-models/yolov8/yolov8n.pt", task="detect")
            # Run detection using raw NumPy image and threshold
            results = self.yolo_wrapper.predict(source=image_np, conf=threshold, verbose=False)
            # Extract raw detections as arrays
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                xyxy_arr = results[0].boxes.xyxy.cpu().numpy()
                confs    = results[0].boxes.conf.cpu().numpy()
                cls_ids  = results[0].boxes.cls.cpu().numpy()
            else:
                xyxy_arr = np.empty((0, 4))
                confs    = np.array([])
                cls_ids  = np.array([])

        draw = Image.fromarray(image_np.copy())
        pred_classes = []
        pred_confs = []
        pred_boxes = []
        closest_confidence = None
        best_class = None
        best_iou = None

        if xyxy_arr.shape[0] > 0:
            draw_ctx = ImageDraw.Draw(draw)
            boxes = []
            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy_arr, confs, cls_ids):
                # Scale box back to original image size
                x1s, y1s = x1 * scale_x, y1 * scale_y
                x2s, y2s = x2 * scale_x, y2 * scale_y
                boxes.append([x1s, y1s, x2s, y2s])
                xyxy_int = [int(x1s), int(y1s), int(x2s), int(y2s)]
                draw_ctx.rectangle(xyxy_int, outline="red", width=3)
                pred_classes.append(int(cls_id))
                pred_confs.append(float(conf))
                pred_boxes.append((x1s, y1s, x2s, y2s))
                class_name = self.resolve_label_index(int(cls_id))
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14)
                except OSError:
                    font = ImageFont.load_default()
                draw_ctx.text((xyxy_int[0], xyxy_int[1] - 10), f"{class_name}, {conf:.2f}", fill="white", font=font)

            boxes_tensor = ch.tensor(boxes, dtype=ch.float32)

            if gt_bbox is not None and len(boxes_tensor) > 0:
                gt_box_tensor = ch.tensor([gt_bbox], dtype=ch.float32)
                ious = box_iou(boxes_tensor, gt_box_tensor).squeeze(1)
                best_idx = ious.argmax().item()
                best_iou = ious[best_idx].item()
                best_class = pred_classes[best_idx] if best_iou > 0.5 else None
                closest_confidence = float(confs[best_idx]) if best_iou > 0.5 else None
                target_pred_exists = (best_iou > 0.5 and best_class == target)
                untarget_pred_not_exists = not (best_iou > 0.5 and best_class == untarget)
            else:
                target_pred_exists = target in pred_classes
                untarget_pred_not_exists = all(cls != untarget for cls in pred_classes)
        else:
            target_pred_exists = False
            untarget_pred_not_exists = True

        draw.save(path)

        if result_dict:
            meets_criteria = (
                (is_targeted and target_pred_exists and (untarget is None or untarget_pred_not_exists)) or
                (not is_targeted and untarget_pred_not_exists)
            )
            # assemble structured detections list
            detections = []
            if gt_bbox is not None and 'boxes_tensor' in locals() and len(boxes_tensor) > 0:
                # use the previously computed ious and parallel lists
                for idx, iou_val in enumerate(ious.tolist()):
                    detections.append({
                        "class_name": self.resolve_label_index(pred_classes[idx]),
                        "conf": pred_confs[idx],
                        "iou": iou_val
                    })            
            return_result = {
                "detections": detections,
                "closest_class": best_class if gt_bbox is not None else None,
                "closest_class_name": self.resolve_label_index(best_class) if best_class is not None else None,
                "closest_confidence": closest_confidence if gt_bbox is not None else None,
                "untarget_pred_not_exists": untarget_pred_not_exists,
                "target_pred_exists": target_pred_exists,
            }
            return meets_criteria, return_result

        if is_targeted:
            if target_pred_exists and (untarget is None or untarget_pred_not_exists):
                return True
        elif not is_targeted and untarget_pred_not_exists:
            return True
        return False

    def preprocess_input(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        transform = T.ToTensor()  # Converts to [0,1], shape (C,H,W)
        image_tensor = transform(image)
        return {"image": image_tensor}

    def zero_grad(self):
        # YOLOv8 is not typically used with backward passes; placeholder.
        if self.model is not None:
            for param in self.model.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

    def resolve_label_index(self, label):
        def normalize(name):
            return name.replace('_', ' ').lower()

        coco_class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
            'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        if isinstance(label, int):
            if 0 <= label < len(coco_class_names):
                return coco_class_names[label]
            raise ValueError(f"Class index {label} is out of bounds.")

        label = normalize(label)
        label_lookup = {normalize(name): idx for idx, name in enumerate(coco_class_names)}
        if label not in label_lookup:
            raise ValueError(f"Label '{label}' not found in YOLOv8 COCO class list.")

        return label_lookup[label]
