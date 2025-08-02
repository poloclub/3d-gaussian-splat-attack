import torch as ch
from PIL import Image
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def test_bbox_transform(x1, y1, x2, y2, width, height):
    import torch as ch
    # Make a shape (1,1,4) tensor
    bboxes = ch.tensor([[[x1, y1, x2, y2]]], dtype=ch.float32)
    
    # The code from your 'infer()':
    boxes = bboxes.clone()
    boxes[:, :, 0::2] /= width
    boxes[:, :, 1::2] /= height
    
    xywh = ch.zeros_like(boxes)
    xywh[:, :, 0] = (boxes[:, :, 0] + boxes[:, :, 2]) / 2  # x_center
    xywh[:, :, 1] = (boxes[:, :, 1] + boxes[:, :, 3]) / 2  # y_center
    xywh[:, :, 2] = boxes[:, :, 2] - boxes[:, :, 0]        # width
    xywh[:, :, 3] = boxes[:, :, 3] - boxes[:, :, 1]        # height
    return xywh

# Example: Suppose an image of width=400, height=600
# and a pixel bounding box: [x1=100, y1=200, x2=200, y2=400]
res = test_bbox_transform(100, 200, 200, 400, 400, 600)
print("Resulting xywh =", res)


# === Load model ===
model = YOLO("pretrained-models/yolov8/yolov8n.pt")
model.eval()

# === Load image ===
img_path = "renders/render_concat_0.png"  # Update this
image = Image.open(img_path).convert("RGB")
img_tensor = TF.to_tensor(image).unsqueeze(0) * 255  # shape: (1, 3, H, W)
img_tensor = img_tensor.to(dtype=ch.float32)

def resize_to_multiple(img, multiple=32):
    h, w = img.shape[2], img.shape[3]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    return ch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

img_tensor = resize_to_multiple(img_tensor)

# === Run inference ===
with ch.no_grad():
    raw_output = model.model(img_tensor)  # raw predictions (logits)
    nms_output = non_max_suppression(raw_output, conf_thres=0.3)[0]

# === Visualize raw outputs ===
def visualize_predictions(image_np, boxes, title):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=1.5, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{int(cls.item())}: {conf.item():.2f}',
                    color='red', fontsize=8, weight='bold')
    ax.set_title(title)
    plt.axis('off')
    output_filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_filename}")

# Convert image for plotting
image_np = np.array(image)

# === Raw Predictions ===
raw_output_tensor = raw_output[0] if isinstance(raw_output, tuple) else raw_output
print("Raw model output shape:", raw_output_tensor.shape)
results = model(img_tensor, verbose=False)
filtered_raw = results[0].boxes.data  # already decoded, [x1, y1, x2, y2, conf, cls]
visualize_predictions(image_np, filtered_raw.cpu(), "Raw YOLOv8 Output (conf > 0.01)")

# === NMS Output ===
if nms_output is not None:
    visualize_predictions(image_np, nms_output.cpu(), "Post-NMS Output (conf > 0.3)")
else:
    print("No detections after NMS.")

