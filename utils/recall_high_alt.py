

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json

CATEGORY_MAP = {
    "car": 2,
    "suitcase": 28,
    "toilet": 72,
    "tv": 64,
    "cell phone": 67,
    # add more classes as needed
}

def build_coco_jsons(log_path, width, height, gt_json_path, dt_json_path):
    with open(log_path, 'r') as f:
        entries = [json.loads(line.split(' - ')[-1]) for line in f if '"cam"' in line]

    images = []
    annotations = []
    dt_results = []
    ann_id = 1

    for entry in entries:
        img_id = entry['cam']
        if img_id not in {img['id'] for img in images}:
            images.append({'id': img_id, 'width': width, 'height': height, 'file_name': ''})

        # ground truth annotation
        gt_bbox = entry.get('gt_bbox')
        if gt_bbox:
            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': CATEGORY_MAP['car'],
                'bbox': gt_bbox,
                'area': gt_bbox[2] * gt_bbox[3],
                'iscrowd': 0
            }
            annotations.append(ann)
            ann_id += 1

        # detection result
        pred_cls = entry.get('pred_class')
        pred_cat = entry.get('pred_category_id')
        conf = entry.get('confidence')
        pred_bbox = entry.get('bbox')
        if pred_cls != 'None' and pred_bbox and conf and pred_cat is not None:
            dt_results.append({
                'image_id': img_id,
                'category_id': pred_cat,
                'bbox': pred_bbox,
                'score': float(conf)
            })

    gt_json = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': cid, 'name': name} for name, cid in CATEGORY_MAP.items()]
    }
    with open(gt_json_path, 'w') as f:
        json.dump(gt_json, f)
    with open(dt_json_path, 'w') as f:
        json.dump(dt_results, f)

def run_coco_eval(gt_json_path, dt_json_path, iou_thr=0.5):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(dt_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = [iou_thr]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == '__main__':
    import sys
    import os
    # Parse command-line arguments, ignoring IPython kernel flags
    user_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    # log_file: first user arg or default
    # default_log = '/nvmescratch/mhull32/3D-Gaussian-Splat-Attack/multirun/2025-07-23/23-20-27/high_alt_test_nyc_blue_car_benign.ply_yolov3/render.log'
    # default_log = '/nvmescratch/mhull32/3D-Gaussian-Splat-Attack/multirun/2025-07-23/23-20-27/high_alt_test_nyc_blue_car_adv.ply_yolov3/render.log'
    # default_log = '/nvmescratch/mhull32/3D-Gaussian-Splat-Attack/multirun/2025-07-24/15-03-58/high_alt_test_nyc_blue_car_benign.ply_yolov11/render.log'
    default_log = '/nvmescratch/mhull32/3D-Gaussian-Splat-Attack/multirun/2025-07-24/15-06-18/high_alt_test_nyc_blue_car_adv.ply_yolov11/render.log'
    log_file = user_args[0] if len(user_args) >= 1 and os.path.isfile(user_args[0]) else default_log
    # width and height: next two args if provided
    width = int(user_args[1]) if len(user_args) >= 2 else 640
    height = int(user_args[2]) if len(user_args) >= 3 else 640
    gt_json = 'gt.json'
    dt_json = 'preds.json'

    build_coco_jsons(log_file, width, height, gt_json, dt_json)
    run_coco_eval(gt_json, dt_json)