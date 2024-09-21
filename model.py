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


def model_eval_mode(model):
    model.train = False
    model.training = False
    model.proposal_generator.training = False
    model.roi_heads.training = False
    return model

def model_train_mode(model):
    model.train = True
    model.training = True
    model.proposal_generator.training = True
    model.roi_heads.training = True
    return model

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



def save_adv_image_preds(model \
    , dt2_config \
    , input \
    , instance_mask_thresh=0.7 \
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
    model = model_eval_mode(model)  
    with ch.no_grad():
        adv_outputs = model([input])
        perturbed_image = input['image'].data.permute((1,2,0)).detach().cpu().numpy()
        pbi = ch.tensor(perturbed_image, requires_grad=False).detach().cpu().numpy()
        if format=="BGR":
            pbi = pbi[:, :, ::-1]
        v = Visualizer(pbi, MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]),scale=1.0)
        instances = adv_outputs[0]['instances']
        things = np.array(MetadataCatalog.get(dt2_config.DATASETS.TRAIN[0]).thing_classes) # holds class labels
        mask = instances.scores > instance_mask_thresh
        instances = instances[mask]
        predicted_classes = things[instances.pred_classes.cpu().numpy().tolist()] 
        print(f'Predicted Class: {predicted_classes}')        
        out = v.draw_instance_predictions(instances.to("cpu"))
        target_pred_exists = target in instances.pred_classes.cpu().numpy().tolist()
        untarget_pred_not_exists = untarget not in instances.pred_classes.cpu().numpy().tolist()
        pred = out.get_image()

    model = model_train_mode(model)
    
    PIL.Image.fromarray(pred).save(path)
    if is_targeted and target_pred_exists:
        return True
    elif (not is_targeted) and (untarget_pred_not_exists):
        return True
    return False

def get_instances_bboxes(model, input, target=None, threshold=0.7):
    """
    Get the bounding boxes from the model's predictions
    """
    model = model_eval_mode(model)
    with ch.no_grad():
        outputs = model([input])
        instances = outputs[0]['instances']
        mask = instances.scores > threshold
        if target is not None:
            mask = mask & (instances.pred_classes == target)
        instances = instances[mask]
        bboxes = instances.pred_boxes.tensor.detach().cpu().numpy()
        if bboxes.size == 0:
            return np.array([[0.0, 0.0, input['height'], input['width']]])
    model = model_train_mode(model)
    return bboxes 

def model_input(model, x, target, bboxes, batch_size=1):
    """
    To get the losses using DT2, we must supply the Ground Truth w/ the input dict
    as an Instances object. This includes the ground truth boxes (gt_boxes)
    and ground truth classes (gt_classes).  There should be a class & box for 
    each GT object in the scene.
    """
    
    model = model_train_mode(model)

    if x.dim() == 3:
        x = x.unsqueeze(0).requires_grad_()  
    x.retain_grad()
    # visualize x
    # z = x[0].detach().cpu().numpy()
    # PIL.Image.fromarray(((z - z.min()) / (z.max() - z.min())*255).clip(0, 255).astype(np.uint8)).save("renders/bw/tensor.png")
    
    
    # incoming tensor is N, H, W, C
    losses_name = ["loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc"]            
    target_loss_idx = [0] # this targets es only `loss_cls` loss
    # detectron2 wants images as RGB 0-255 range
    x = ch.clip(x * 255 + 0.5, 0, 255).requires_grad_()
    # x = ch.permute(x, (0, 3, 1, 2)).requires_grad_()
    x.retain_grad()
    height = x.shape[2]
    width = x.shape[3]
    if ch.tensor(bboxes).dim() == 1:
        # pad tensor if only dealing w/ single bbox
        gt_boxes = ch.tensor(bboxes).unsqueeze(0)
    else :
        gt_boxes = ch.tensor(bboxes)

    inputs = list()
    for i in  range(0, x.shape[0]):                
        instances = Instances(image_size=(height,width))
        instances.gt_classes = target.long()
        instances.gt_boxes = Boxes(gt_boxes[i])
        input = {}
        input['image']  = x[i]    
        input['filename'] = ''
        input['height'] = height
        input['width'] = width
        input['instances'] = instances      
        inputs.append(input)
    with EventStorage(0) as storage:            
        # loss = model([input])[losses_name[target_loss_idx[0]]].requires_grad_()
        losses = model(inputs)
        loss = sum([losses[losses_name[tgt_idx]] for tgt_idx in target_loss_idx]).requires_grad_()
    del x
    return loss

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
    score_thresh = 0.5
    
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
    model, dt2_config = detectron2_model()
    img = dt2_input("renders/render_1.png")    
    save_adv_image_preds(model, dt2_config, img, path="renders/render_1_preds.png")
    print('done')
    
    