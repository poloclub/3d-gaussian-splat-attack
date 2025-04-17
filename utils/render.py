import sys
import os
import copy
import hydra
import argparse
import torch
import subprocess
import numpy as np
from tqdm import tqdm
import PIL
from PIL import Image, ImageDraw
from random import randint
from omegaconf import DictConfig, OmegaConf
from detectors.factory import load_detector
from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera  
from arguments import GroupParams
# from edit_object_removal import points_inside_convex_hull
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import logging
log = logging.getLogger(__name__)

select_thresh = 0.5 # selected threshold for the gaussian group



# def main(args):
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))    
    # Initialize dataset parameters
    dataset = GroupParams()
    dataset.data_device = cfg.data_device
    dataset.eval = cfg.eval
    dataset.images = cfg.images
    dataset.model_path = cfg.scene.model_path
    dataset.source_path = cfg.scene.source_path
    dataset.combine_splats = cfg.combine_splats
    dataset.cam_indices = None #cfg.scene.cam_indices  # select specific cameras instead of loading all of them.
    dataset.n_views = cfg.n_views
    dataset.num_classes = cfg.num_classes
    dataset.object_path = cfg.object_path
    dataset.random_init = cfg.random_init
    dataset.resolution = cfg.resolution
    dataset.sh_degree = cfg.sh_degree
    dataset.train_split = cfg.train_split
    dataset.white_background = cfg.white_background
    dataset.device = cfg.device
    dataset.no_groups = cfg.no_groups
    if dataset.no_groups:
        log.warning("no_groups is not supported in rendering and will be ignored.")

    # Initialize optimization parameters
    opt = GroupParams()
    opt.densification_interval = cfg.densification_interval
    opt.density_From_iter = cfg.density_From_iter
    opt.densify_grad_threshold = cfg.densify_grad_threshold
    opt.density_until_iter = cfg.density_until_iter
    opt.feature_lr = cfg.feature_lr
    opt.iterations = cfg.iterations
    opt.lambda_dssim = cfg.lambda_dssim
    opt.opacity_lr = cfg.opacity_lr
    opt.opacity_reset_interval = cfg.opacity_reset_interval
    opt.percent_dense = cfg.percent_dense
    opt.position_lr_delay_mult = cfg.position_lr_delay_mult
    opt.position_lr_final = cfg.position_lr_final
    opt.position_lr_init = cfg.position_lr_init
    opt.position_lr_max_steps = cfg.position_lr_max_steps
    opt.reg3d_interval = cfg.reg3d_interval
    opt.reg3d_k = cfg.reg3d_k
    opt.reg3d_lambda_val = cfg.reg3d_lambda_val
    opt.reg3d_max_points = cfg.reg3d_max_points
    opt.reg3d_sample_size = cfg.reg3d_sample_size
    opt.rotation_lr = cfg.rotation_lr
    opt.scaling_lr = cfg.scaling_lr

    # Pipeline parameters
    pipe = GroupParams()
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False
    pipe.debug = False
    # set the cuda device to the args.device
    DEVICE = f"cuda:{cfg.device}" 
    torch.cuda.set_device(DEVICE)
    # Additional attack setup

    start_cam, end_cam, add_cams = cfg.start_cam, cfg.end_cam, cfg.add_cams
    shift_amount = cfg.shift_amount
    attack_conf_thresh = cfg.attack_conf_thresh
    batch_mode = cfg.batch_mode 
    if batch_mode:
        raise NotImplementedError("Batch mode is not supported in rendering and will be ignored.")


    # cleanup render and preds directories
    subprocess.run(["make", "clean"], shell=True)        
    # load detector
    detector = load_detector(cfg)
    detector.load_model()
    
    if isinstance(cfg.scene.target, str):
        cfg.scene.target = [detector.resolve_label_index(cfg.scene.target)]
        target = torch.tensor(cfg.scene.target, device=cfg.data_device)
    
    if dataset.combine_splats == False:
        # Without groups, the perturbation will be applied to the entire scene.
        # this is ideal scenes with a single object.
        # Load Gaussian Splat model
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.training_setup(opt)
        scene = Scene(args=dataset, gaussians=gaussians,load_iteration=30000, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)
        # copy gaussians variable to new object
        gaussians_original = copy.deepcopy(gaussians)

    else:
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.training_setup(opt)
        scene = Scene(args=dataset, gaussians=gaussians,load_iteration=-2, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
        # List of .ply file paths to be combined
        ply_paths = [os.path.join(cfg.splat_asset_path,cfg.scene.target_splat), 
                     os.path.join(cfg.splat_asset_path,cfg.scene.background_splat)]

        if ply_paths is None or len(ply_paths) < 2:
            raise ValueError("At least two .ply paths must be provided for combine_splats mode (target + background).")                     

        # Combine the .ply files
        gaussians.combine_splats(ply_paths)
        # Demonstrate how to extract obj_1 and obj_2 using self.masks
        obj_1_mask = gaussians.masks[0]
        obj_2_mask = gaussians.masks[1]

        # Pad obj_1_mask with the shape of obj_2_mask
        pad_size = obj_2_mask.shape[0]
        if pad_size > 0:
            padding = torch.zeros(pad_size, dtype=torch.bool, device=obj_1_mask.device)
            obj_1_mask = torch.cat((obj_1_mask, padding), dim=0)

        # make mask 3D
        obj_1_mask3d = obj_1_mask.view(1, obj_1_mask.shape[0], 1)
        obj_1_mask3d = obj_1_mask3d.any(dim=0).squeeze()
        obj_1_mask3d = obj_1_mask3d.float()[:, None, None]

        gaussians_original = copy.deepcopy(gaussians)
        
        # Updated apply_mask function to handle mask correctly
        gaussians_original.removal_setup(opt, obj_1_mask3d) # inverse 
        gaussians.removal_setup(opt, ~obj_1_mask3d.bool())
             
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # single camera or range of cameras
        # viewpoint_stack = scene.getTrainCameras().copy()[0:] 
        # for i, camera in enumerate(viewpoint_stack):
        #     render_pkg = render(camera, gaussians_original, pipe, bg)
        #     img_path = f"renders/combined_splats/combined_splats_{i}.png"
        #     Image.fromarray((torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
        #                 .byte()
        #                 .permute(1, 2, 0)
        #                 .contiguous()
        #                 .cpu()
        #                 .numpy()).save(img_path)        

    
    print("Scene setup complete. Starting render...")
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0, 0 ]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # viewpoint_stack = scene.getTrainCameras().copy()  # use all cameras

    # single camera or range of cameras
    viewpoint_stack = scene.getTrainCameras().copy()[start_cam:end_cam] 
    

    # for i in range(1, add_cams):

    #     # make a new camera with a view that we choose. 
    #     camera = copy.deepcopy(viewpoint_stack[0])
    #     # Shift left
    #     # T = camera.T
    #     # T[0] += shift_amount * i
    #     # camera.transform(T)
    #     # yaw right 
    #     camera.yaw(7*i)

    #     viewpoint_stack.append(camera)

    total_views = len(viewpoint_stack)

    # get benign render bboxes - would be better if you could SOLO render the target!
    # for each benign render, get the bbox w/ detection of target class.
    bboxes = []
    for i, cam in enumerate(tqdm(viewpoint_stack, desc="Rendering GT bboxes...", unit="camera")):
        render_pkg = render(cam, gaussians, pipe, torch.tensor([0,0,0],dtype=torch.float32, device="cuda")) # always use black background for detection
        img_path = f"renders/render_{i}.png"
        np_img = (torch.clamp(render_pkg["render"], min=0, max=1.0) * 255) \
                        .byte() \
                        .permute(1, 2, 0) \
                        .contiguous() \
                        .cpu() \
                        .numpy() 
        pil_img = Image.fromarray(np_img)
        pil_img_bw = pil_img.convert('L')
        bw_tresh = 20
        pil_img_bw = pil_img_bw.point(lambda p: p > bw_tresh and 255)
        # pil_img_bw = PIL.ImageOps.invert(pil_img_bw)
        bbox = pil_img_bw.getbbox()
        
        pil_img.save(img_path)

        rendered_img_input = detector.preprocess_input(img_path)
        # bbox = get_instances_bboxes(model, rendered_img_input, target = target.detach().cpu().numpy(), threshold=0.2)
        bboxes.append(bbox)

        draw = PIL.ImageDraw.Draw(pil_img_bw)
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), "object", fill="red")
        pil_img_bw.save(f'renders/bw/bbox_render_{i}.jpg')    
        bbox = np.expand_dims(np.array(bbox), axis=0)

    gt_bboxes = np.array(bboxes)    


    for it in tqdm(range(0, total_views), desc="Rendering", unit="it"):
        renders = []
        cam = viewpoint_stack[0]
        render_pkg = render(cam, gaussians, pipe, bg)
        renders.append(render_pkg["render"])
        renders = torch.stack(renders)
        
        combined_gaussians = copy.deepcopy(gaussians)
        # combine the gaussians .plys together in one scene.
        combined_gaussians.concat_setup("features_rest", gaussians_original._features_rest, True)
        combined_gaussians.concat_setup("features_dc", gaussians_original._features_dc, True)
        combined_gaussians.concat_setup("xyz", gaussians_original._xyz, True)
        combined_gaussians.concat_setup("scaling", gaussians_original._scaling, True)
        combined_gaussians.concat_setup("opacity", gaussians_original._opacity, True)
        combined_gaussians.concat_setup("rotation", gaussians_original._rotation, True)
        combined_gaussians.concat_setup("objects_dc", gaussians_original._objects_dc, True)        
        concat_renders = []
        cam = viewpoint_stack[0]
        render_pkg = render(cam, combined_gaussians, pipe, bg)
        concat_renders.append(render_pkg["render"]) 

        # img_path = f"renders/nyc_block_maserati/{it}.png"   
        img_path = f"renders/nyc_block_car_test/{it}.png"
        cr = concat_renders[0]
        preds_path = "preds"
        Image.fromarray((torch.clamp(cr, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()).save(img_path)

        rendered_img_input = detector.preprocess_input(img_path)
        success, result = detector.predict_and_save(
                image=cr,
                path=os.path.join(preds_path, f'render_c{total_views-len(viewpoint_stack)}.png'),
                target=target,
                untarget=None,
                is_targeted=True,
                threshold=attack_conf_thresh,
                gt_bbox = gt_bboxes[it],
                result_dict=True)
        closest_class = result['closest_class_name'] if result['closest_class_name'] is not None else "None"
        confidence_str = f"{result['closest_confidence']:.4f}" if isinstance(result['closest_confidence'], (float, int)) else "None"
        log.info(f"[cam {it}] success: {success}, max_iou_pred: {closest_class}, conf: {confidence_str}")
        
        viewpoint_stack.pop(0)
        if len(viewpoint_stack) == 0:
            print ("finished rendering all cameras")                  

if __name__ == "__main__":
    run()
