from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import io
import contextlib
import numpy as np


# Subclass COCOeval to provide concise summary output
class MiniCOCOeval(COCOeval):
    """
    Subclass of COCOeval with a concise summary method that only prints
    AP@0.50 (area=all,maxDets=100) and AR@0.50 (area=all,maxDets=1).
    """
    def selective_summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            # stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

CATEGORY_MAP = {
    "car": 2,
    "suitcase": 28,
    "toilet": 72,
    "tv": 64,
    "cell phone": 67,
    # add more classes as needed
    "stop sign": 11,  
}

def build_coco_jsons(log_path, width, height, gt_json_path, dt_json_path, target_class):
    with open(log_path, 'r') as f:
        entries = [json.loads(line.split(' - ')[-1]) for line in f if '"cam"' in line]
    print(f"building ground truth for target class: {target_class}")
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
                'category_id': CATEGORY_MAP[target_class],
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
    coco_eval = MiniCOCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = [iou_thr]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    # Use the concise summary method
    coco_eval.selective_summarize()

if __name__ == '__main__':
    import sys
    import os
    # Parse command-line arguments, ignoring IPython kernel flags
    user_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    # width and height: next two args if provided
    width = int(user_args[1]) if len(user_args) >= 2 else 640
    height = int(user_args[2]) if len(user_args) >= 3 else 640
    gt_json = 'gt.json'
    dt_json = 'preds.json'

    model_type_dirs = ['yolov3', 'yolov5', 'yolov8', 'yolov11', 'detectron2', 'detr']

    for i in range(len(model_type_dirs)):
        base_dir = f'/nvmescratch/mhull32/3D-Gaussian-Splat-Attack/multirun/ablation_car_SH/{model_type_dirs[i]}'

        # grab all render.log files in subdirectories
        for sub in sorted(os.listdir(base_dir)):
            subdir = os.path.join(base_dir, sub)
            log_file = os.path.join(subdir, 'render.log')
            if not os.path.isfile(log_file):
                continue

            # Parse experiment name and benign/adv: high_alt_test_<scene>_<type>.ply_<model>
            dname = sub
            idx = dname.find('.ply_')
            if idx != -1:
                prefix = dname[:idx]
                model = dname[idx + len('.ply_'):]
            else:
                prefix = dname
                model = 'unknown'

            # exp_prefix = 'high_alt_test_'
            # exp_prefix = 'alt_4_'
            exp_prefix = 'SH1_high_alt_test_'
            if prefix.startswith(exp_prefix):
                prefix = prefix[len(exp_prefix):]

            parts = prefix.rsplit('_', 1)
            scene = parts[0] if len(parts) > 0 else prefix
            exp_type = parts[1] if len(parts) > 1 else ''

            print(f"\n===== Experiment: {sub} =====")
            print(f"Model: {model}, Scene: {scene}, Type: {exp_type}")

            # Build and evaluate
            build_coco_jsons(log_file, width, height, gt_json, dt_json, target_class='car')
            run_coco_eval(gt_json, dt_json)
