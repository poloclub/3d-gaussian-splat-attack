import torch as ch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from types import SimpleNamespace
from detectors.base_detector import BaseDetector
from ultralytics.nn.tasks import attempt_load_one_weight
import yaml
from ultralytics.utils.ops import non_max_suppression

class Yolov3Detector(BaseDetector):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None

    def load_model(self):
        weights_path = "pretrained-models/yolov3/yolov3.pt"
        model, _ = attempt_load_one_weight(weights_path, device=f"cuda:{self.cfg.device}")
        model.train()
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, task="detect")
        self.model = model

    def infer(self, x, target, bboxes, batch_size=1):
        self.model.model.train()
        if isinstance(bboxes, np.ndarray):
            bboxes = ch.tensor(bboxes, dtype=ch.float32)
        if isinstance(target, np.ndarray):
            target = ch.tensor(target, dtype=ch.long)
        bboxes = bboxes.to(x.device)
        target = target.to(x.device)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0).unsqueeze(0)
        elif bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(1)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(dtype=ch.float32)

        def letterbox(img, new_shape=(960, 960), color=(114, 114, 114)):
            B, C, H, W = img.shape
            new_h, new_w = new_shape
            scale = min(new_h / H, new_w / W)
            resized_h = int(round(H * scale))
            resized_w = int(round(W * scale))
            pad_h = new_h - resized_h
            pad_w = new_w - resized_w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            img_resized = ch.nn.functional.interpolate(img, size=(resized_h, resized_w), mode='bilinear', align_corners=False)
            img_padded = ch.nn.functional.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), value=color[0]/255.0)
            return img_padded, scale, pad_left, pad_top

        x, scale, pad_left, pad_top = letterbox(x, new_shape=(960, 960))
        new_w, new_h = 960, 960

        boxes = bboxes.clone()
        boxes = boxes * ch.tensor([scale, scale, scale, scale], device=boxes.device)
        boxes[:, :, 0::2] += pad_left
        boxes[:, :, 1::2] += pad_top
        boxes[:, :, 0::2] /= new_w
        boxes[:, :, 1::2] /= new_h

        xywh = ch.zeros_like(boxes)
        xywh[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2
        xywh[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2
        xywh[:, :, 2] = boxes[:, :, 2] - boxes[:, :, 0]
        xywh[:, :, 3] = boxes[:, :, 3] - boxes[:, :, 1]

        losses = []
        for i in range(x.shape[0]):
            targets = ch.cat([target[i].view(-1, 1).float(), xywh[i]], dim=1)
            targets = targets.to(x.device)
            self.model.train()
            self.model.training = True
            batch = {
                "img": x[i].unsqueeze(0),
                "cls": target[i].view(-1),
                "bboxes": xywh[i].to(x.device),
                "batch_idx": ch.zeros(target[i].numel(), dtype=ch.int64, device=x.device)
            }
            out = self.model(batch)
            loss = out[0] if isinstance(out, tuple) else out
            losses.append(loss)

        return (sum(losses) / len(losses)).sum()

    def predict_and_save(self, image: ch.Tensor, path: str, target: int = None, untarget: int = None, is_targeted: bool = True, threshold: float = 0.7, format: str = "RGB") -> bool:
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

        self.model.eval()
        with ch.no_grad():
            results = self.model(img_tensor)
            results = non_max_suppression(results[0], conf_thres=threshold, iou_thres=0.45)[0]
            dets = results

        draw = Image.fromarray(image_np.copy())
        if dets is not None and len(dets) > 0:
            draw_ctx = ImageDraw.Draw(draw)
            for det in dets:
                if not isinstance(det, ch.Tensor):
                    continue
                det = det.squeeze()
                if det.numel() < 6:
                    continue
                det = det.flatten()
                *xyxy, conf, cls = det.tolist()[:6]
                x1, y1, x2, y2 = [float(coord) for coord in xyxy[:4]]
                if x2 < x1 or y2 < y1:
                    continue                
                x1 *= scale_x
                x2 *= scale_x
                y1 *= scale_y
                y2 *= scale_y
                xyxy = [int(x1), int(y1), int(x2), int(y2)]
                draw_ctx.rectangle(xyxy, outline="red", width=3)
                class_idx = int(cls)
                class_name = self.resolve_label_index(class_idx)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14)
                except OSError:
                    font = ImageFont.load_default()
                draw_ctx.text((xyxy[0], xyxy[1] - 10), f"{class_name}, {conf:.2f}", fill="white", font=font)
        draw.save(path)

        pred_classes = []
        if dets is not None and dets.numel() > 0 and dets.ndim >= 2:
            pred_classes = dets[:, -1].tolist()
        
        print(f'Predicted Class: {[self.resolve_label_index(int(cls)) for cls in pred_classes]}')

        target_pred_exists = target.item() in pred_classes if target is not None else False
        untarget_pred_not_exists = untarget.item() not in pred_classes if untarget is not None else True
        if is_targeted:
            return target_pred_exists and (untarget_pred_not_exists if untarget is not None else True)
        else:
            return untarget_pred_not_exists

    def preprocess_input(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        transform = T.ToTensor()
        image_tensor = transform(image)
        return {"image": image_tensor}

    def zero_grad(self):
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
            raise ValueError(f"Label '{label}' not found in YOLOv3 COCO class list.")
        return label_lookup[label]