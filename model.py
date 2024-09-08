import os, PIL
import numpy as np
import torch as ch
from torchvision.io import read_image
import logging
import os
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.structures import Boxes, Instances
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
from detectron2.data.detection_utils import *
from detectron2.modeling import build_model
from detectron2.utils.events import EventStorage

def dt2_input(image_path:str)->dict:
    """
    Construct a Detectron2-friendly input for an image
    #FIXME - bboxes are hardcoded and they shouldn't be. 
    """
    input = {}
    filename = image_path
    adv_image = read_image(image_path, format="RGB")
    adv_image_tensor = ch.as_tensor(np.ascontiguousarray(adv_image.transpose(2, 0, 1)))

    height = adv_image_tensor.shape[1]
    width = adv_image_tensor.shape[2]
    instances = Instances(image_size=(height,width))
    # TODO - review this class setting... why is it needed?
    instances.gt_classes = ch.Tensor([2])
    # taxi bbox
    # instances.gt_boxes = Boxes(ch.tensor([[ 50.9523, 186.4931, 437.6184, 376.7764]]))
    # stop sign bbox
    # instances.gt_boxes = Boxes(ch.tensor([[ 162.0, 145.0, 364.0, 324.0]])) # for 512x512 img
    # instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, height, width]]))
    instances.gt_boxes = Boxes(ch.tensor([[0.0, 0.0, float(height), float(width)]]))
    input['image'] = adv_image_tensor    
    input['filename'] = filename
    input['height'] = height
    input['width'] = width
    input['instances'] = instances
    return input

def save_dt2_image_preds(model \
    , dt2_config \
    , input \
    , instance_mask_thresh=0.2 \
    , target:int=None \
    , untarget:int=None
    , is_targeted:bool=True \
    , format="RGB" \
    , path:str=None):
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
        untarget_pred_not_exists = untarget not in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True  
    PIL.Image.fromarray(pred).save(path)
    if is_targeted and target_pred_exists:
        return True
    elif (not is_targeted) and (untarget_pred_not_exists):
        return True
    return False

def detectron2_model():
    """
    Initializes and configures a Detectron2 model for object detection.

    This function sets up a Detectron2 model using a specified configuration file and weights file.
    It also sets the score threshold for the Region of Interest (ROI) heads and configures the device
    to be used for computation (CPU or GPU).

    Returns:
        model (detectron2.modeling.meta_arch.GeneralizedRCNN): The initialized and configured Detectron2 model.
        dt2_config (detectron2.config.CfgNode): The configuration object used to set up the model.
    """    
    model_config = "pretrained-models/faster_rcnn_R_50_FPN_3x/config.yaml"
    weights_file = "pretrained-models/faster_rcnn_R_50_FPN_3x/model_final.pth"
    score_thresh = 0.2
    
    cuda_visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", default=1)
    DEVICE = f"cuda:{cuda_visible_device}"
    
    logging.basicConfig(level=logging.INFO)
    dt2_config = get_cfg()
    dt2_config.merge_from_file(model_config)
    dt2_config.MODEL.WEIGHTS = weights_file
    dt2_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    dt2_config.MODEL.DEVICE = DEVICE
    model = build_model(dt2_config)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(dt2_config.MODEL.WEIGHTS)
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True
    return model, dt2_config
    

if __name__ == "__main__":
    pass    
    # model, dt2_config = detectron2_model()
    # img = dt2_input("renders/render_1.png")    
    # save_dt2_image_preds(model, dt2_config, img, path="renders/render_1_preds.png")
    # print('done')
    
    