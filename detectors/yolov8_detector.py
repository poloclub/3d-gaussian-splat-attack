from ultralytics import YOLO
import torch as ch
from detectors.base_detector import BaseDetector
from PIL import Image, ImageDraw
import torchvision.transforms as T
import numpy as np

class Yolov8Detector(BaseDetector):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = None

    def load_model(self):
        from ultralytics.nn.tasks import DetectionModel
        import yaml

        cfg_path = "pretrained-models/yolov8n.yaml"
        weights_path = "pretrained-models/yolov8n.pt"

        with open(cfg_path, "r") as f:
            model_cfg = yaml.safe_load(f)

        # Inject YOLOv8n scaling factors manually
        model_cfg['depth_multiple'] = 0.33
        model_cfg['width_multiple'] = 0.25

        model = DetectionModel(model_cfg)
        checkpoint = ch.load(weights_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"].float().state_dict(), strict=False)
        DEVICE = f"cuda:{self.cfg.device}" 
        model.to(DEVICE)
        model.train()
        model.training = True
        from types import SimpleNamespace
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

        # Ensure 3D bbox shape: (B, N, 4)
        if bboxes.dim() == 1:
            bboxes = bboxes.unsqueeze(0).unsqueeze(0)
        elif bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(1)

        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(dtype=ch.float32)
        
        def resize_to_multiple(img, multiple=32):
            h, w = img.shape[2], img.shape[3]
            new_h = ((h + multiple - 1) // multiple) * multiple
            new_w = ((w + multiple - 1) // multiple) * multiple
            return ch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

        x = resize_to_multiple(x)

        # Normalize bbox to 0-1 as YOLO expects (x_center, y_center, width, height)
        height = x.shape[2]
        width = x.shape[3]
        boxes = bboxes.clone()
        boxes[:, :, 0::2] /= width   # x1 and x2
        boxes[:, :, 1::2] /= height  # y1 and y2
        xywh = ch.zeros_like(boxes)
        xywh[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2  # x_center
        xywh[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2  # y_center
        xywh[:, :, 2] = boxes[:, :, 2] - boxes[:, :, 0]        # width
        xywh[:, :, 3] = boxes[:, :, 3] - boxes[:, :, 1]        # height

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
            loss = out[0] if isinstance(out, tuple) else out
            losses.append(loss)

        return sum(losses) / len(losses)

    def predict_and_save(self, image: ch.Tensor, path: str, target: int = None, untarget: int = None, is_targeted: bool = True, threshold: float = 0.7, format: str = "RGB") -> bool:
        from ultralytics.utils.ops import non_max_suppression
        import torchvision.transforms.functional as TF

        image_np = (image.detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        image_pil = Image.fromarray(image_np).convert("RGB")
        img_tensor = TF.to_tensor(image_pil).unsqueeze(0).to(next(self.model.parameters()).device) * 255
        img_tensor = img_tensor.to(dtype=ch.float32)

        def resize_to_multiple(img, multiple=32):
            h, w = img.shape[2], img.shape[3]
            new_h = ((h + multiple - 1) // multiple) * multiple
            new_w = ((w + multiple - 1) // multiple) * multiple
            return ch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

        img_tensor = resize_to_multiple(img_tensor)

        self.model.eval()
        with ch.no_grad():
            preds = self.model(img_tensor)
            dets = non_max_suppression(preds, conf_thres=threshold)[0]

        # Save visualization
        draw = Image.fromarray(image_np.copy())
        if dets is not None and len(dets) > 0:
            draw_ctx = ImageDraw.Draw(draw)
            for *xyxy, conf, cls in dets:
                xyxy = [int(coord.item()) for coord in xyxy]
                draw_ctx.rectangle(xyxy, outline="red", width=2)
                draw_ctx.text((xyxy[0], xyxy[1] - 10), f"cls: {int(cls.item())}, conf: {conf.item():.2f}", fill="red")

        draw.save(path)

        pred_classes = dets[:, -1].tolist() if dets is not None else []

        target_pred_exists = target.item() in pred_classes if target is not None else False
        untarget_pred_not_exists = untarget.item() not in pred_classes if untarget is not None else True

        if is_targeted:
            return target_pred_exists and (untarget_pred_not_exists if untarget is not None else True)
        else:
            return untarget_pred_not_exists

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
