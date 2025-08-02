import os
import json
import sys

# parse predictions from a render.log into cam->pred_class (None if no detection)
def load_preds(log_path):
    preds = {}
    with open(log_path, 'r') as f:
        for line in f:
            if '"cam"' not in line:
                continue
            entry = json.loads(line.split(' - ')[-1])
            cls = entry.get('pred_class')
            preds[entry['cam']] = cls if cls != 'None' else None
    return preds

if __name__ == '__main__':
    # Command-line args ignored (we hardcode base path)
    base_root = '/nvmescratch/mhull32/3D-Gaussian-Splat-Attack/multirun/ablation_car_SH'
    target_class = 'car'

    model_types = ['yolov3', 'yolov5', 'yolov8', 'yolov11', 'detectron2', 'detr']
    # colors = ['blue', 'red', 'gray']
    colors = ['blue']

    for model in model_types:
        model_dir = os.path.join(base_root, model)
        for color in colors:
            # Construct subdirectory names for benign and adversarial
            benign_sub = f'high_alt_test_nyc_{color}_car_benign.ply_{model}'
            adv_sub    = f'high_alt_test_nyc_{color}_car_adv.ply_{model}'
            # benign_sub = f'nyc_street_corner_nyc_stop_sign_benign.ply_{model}'
            # adv_sub    = f'nyc_street_corner_nyc_stop_sign_clock.ply_{model}'
            benign_log = os.path.join(model_dir, benign_sub, 'render.log')
            adv_log    = os.path.join(model_dir, adv_sub,    'render.log')

            # Skip if logs are missing
            if not os.path.isfile(benign_log) or not os.path.isfile(adv_log):
                continue

            # Load predictions
            b_preds = load_preds(benign_log)
            a_preds = load_preds(adv_log)

            # Count how many cams detected as target class in benign
            total_car = sum(1 for cls in b_preds.values() if cls == target_class)
            # Count successful attacks: benign was target class but adv is not target class
            successful = sum(
                1
                for cam, cls in b_preds.items()
                if cls == target_class and a_preds.get(cam) != target_class
            )
            asr = successful / total_car if total_car else 0.0

            # Print a one-line summary per model/color
            print(f"Model: {model}, Color: {color}, ASR: {successful}/{total_car} = {asr:.2%}")
            