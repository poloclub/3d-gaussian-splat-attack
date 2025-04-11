"""
Read & render object detection predictions on a batch of images
"""
import os
import PIL
import argparse
import torch as ch
import numpy as np
# from torchvision.io import read_image
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
dir = 'red_cube'


def dt2_input(image_path:str)->dict:
    """
    Construct a Detectron2-friendly input for an image
    """
    input = {}
    filename = image_path
    adv_image = read_image(image_path, format="RGB")
    adv_image_tensor = ch.as_tensor(np.ascontiguousarray(adv_image.transpose(2, 0, 1)))

    height = adv_image_tensor.shape[1]
    width = adv_image_tensor.shape[2]
    instances = Instances(image_size=(height,width))
    instances.gt_classes = ch.Tensor([2])
    # taxi bbox
    # instances.gt_boxes = Boxes(ch.tensor([[ 50.9523, 186.4931, 437.6184, 376.7764]]))
    # stop sign bbox
    instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, height, width]]))
    # instances.gt_boxes = Boxes(ch.tensor([[ 162.0, 145.0, 364.0, 324.0]])) # for 512x512 img
    input['image'] = adv_image_tensor    
    input['filename'] = filename
    input['height'] = height
    input['width'] = width
    input['instances'] = instances
    return input

def save_adv_image_preds(model, dt2_config, input, instance_mask_thresh=0.7, target:int=None, format="RGB", path:str=None):
    """
    Helper fn to save the predictions on an adversarial image
    attacked_image:ch.Tensor An attacked image
    instance_mask_thresh:float threshold pred boxes on confidence score
    path:str where to save image
    """ 
    model.train = False
    model.training = False
    model.proposal_generator.training = False
    model.roi_heads.training = False    
    with ch.no_grad():
        adv_outputs = model([input])
        perturbed_image = input['image'].data.permute((1,2,0)).detach().cpu().numpy()
        pbi = ch.tensor(perturbed_image, requires_grad=False).detach().cpu().numpy()
        if format=="BGR":
            pbi = pbi[:, :, ::-1]
        v = Visualizer(pbi, MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]),scale=1.0)
        instances = adv_outputs[0]['instances']
        things = np.array(MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]).thing_classes) # holds class labels
        predicted_classes = things[instances.pred_classes.cpu().numpy().tolist()] 
        print(f'Predicted Class: {predicted_classes}')        
        mask = instances.scores > instance_mask_thresh
        instances = instances[mask]
        out = v.draw_instance_predictions(instances.to("cpu"))
        target_pred_exists = target in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True  
    PIL.Image.fromarray(pred).save(path)
    if target_pred_exists:
        return True
    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser( \
        description='Example script with default values' \
        ,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--input-dir", help="Directory of images to predict on", type=str, default=dir, required=True)
    parser.add_argument("-st", "--scores-thresh", help="ROI scores threshold", type=float, default=0.3)
    args = parser.parse_args()

    # specify an object detector model
    dt2_cfg = get_cfg()
    dt2_cfg.merge_from_file("pretrained-models/faster_rcnn_R_50_FPN_3x/config.yaml")
    dt2_cfg.MODEL.WEIGHTS = "pretrained-models/faster_rcnn_R_50_FPN_3x/model_final.pth"
    dt2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.scores_thresh
    dt2_cfg.MODEL.DEVICE=1
    model = build_model(dt2_cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_cfg.MODEL.WEIGHTS)    

    # source directory with images we want to predict on
    directory_in_str = f'renders/{args.input_dir}/'
    directory = os.fsencode(directory_in_str)


    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith('.png'):
            # print(filename)
            im_path = os.path.join(directory_in_str, filename)
            input = dt2_input(im_path)
            save_adv_image_preds(model=model, dt2_config=dt2_cfg, input=input, instance_mask_thresh=args.scores_thresh, path=f'preds/{args.input_dir}/{filename}')
