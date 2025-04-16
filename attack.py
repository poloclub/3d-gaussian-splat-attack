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
import torch.nn as nn
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

select_thresh = 0.5 # selected threshold for the gaussian group

def gaussian_position_linf_attack(gaussian, alpha, epsilon, features_xyz):
    with torch.no_grad():
        f_xyz_eta = alpha * torch.sign(gaussian._xyz.grad)
        f_xyz_eta.mul_(-1) # comment out if desire Untargeted
        gaussian._xyz.add_(f_xyz_eta)
        gaussian._xyz.sub_(features_xyz).clamp_(-epsilon, epsilon).add_(features_xyz)

def gaussian_rotation_linf_attack(gaussian, alpha, epsilon, features_rotation):
    with torch.no_grad():
        f_rotation_eta = alpha * torch.sign(gaussian._rotation.grad)
        f_rotation_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._rotation.add_(f_rotation_eta)
        gaussian._rotation.sub_(features_rotation).clamp_(-epsilon, epsilon).add_(features_rotation)

def gaussian_opacity_linf_attack(gaussian, alpha, epsilon, features_opacity):
    with torch.no_grad():
        f_opacity_eta = alpha * torch.sign(gaussian._opacity.grad)
        f_opacity_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._opacity.add_(f_opacity_eta)
        gaussian._opacity.sub_(features_opacity).clamp_(-epsilon, epsilon).add_(features_opacity)

def gaussian_scaling_linf_attack(gaussian, alpha, epsilon, features_scaling):
    with torch.no_grad():
        f_scaling_eta = alpha * torch.sign(gaussian._scaling.grad)
        f_scaling_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._scaling.add_(f_scaling_eta)
        gaussian._scaling.sub_(features_scaling).clamp_(-epsilon, epsilon).add_(features_scaling)

def gaussian_position_l2_attack(gaussian, alpha, epsilon, features_xyz):
    with torch.no_grad():
        grad_xyz = gaussian._xyz.grad
        norm_xyz = torch.norm(grad_xyz.view(-1), p=2)

        if norm_xyz > 0:
            f_xyz_eta = alpha * (grad_xyz / norm_xyz)
        else:
            f_xyz_eta = torch.zeros_like(grad_xyz)

        f_xyz_eta.mul_(-1)
        gaussian._xyz.add_(f_xyz_eta)

        delta_xyz = gaussian._xyz - features_xyz
        delta_xyz = delta_xyz.renorm(p=2, dim=0, maxnorm=epsilon)
        gaussian._xyz.copy_(features_xyz + delta_xyz)

def gaussian_rotation_l2_attack(gaussian, alpha, epsilon, features_rotation):
    with torch.no_grad():
        grad_rotation = gaussian._rotation.grad
        norm_rotation = torch.norm(grad_rotation.view(-1), p=2)

        if norm_rotation > 0:
            f_rotation_eta = alpha * (grad_rotation / norm_rotation)
        else:
            f_rotation_eta = torch.zeros_like(grad_rotation)

        f_rotation_eta.mul_(-1)
        gaussian._rotation.add_(f_rotation_eta)

        delta_rotation = gaussian._rotation - features_rotation
        delta_rotation = delta_rotation.renorm(p=2, dim=0, maxnorm=epsilon)
        gaussian._rotation.copy_(features_rotation + delta_rotation)

def gaussian_opacity_l2_attack(gaussian, alpha, epsilon, features_opacity):
    with torch.no_grad():
        grad_opacity = gaussian._opacity.grad
        norm_opacity = torch.norm(grad_opacity.view(-1), p=2)

        if norm_opacity > 0:
            f_opacity_eta = alpha * (grad_opacity / norm_opacity)
        else:
            f_opacity_eta = torch.zeros_like(grad_opacity)

        f_opacity_eta.mul_(-1)
        gaussian._opacity.add_(f_opacity_eta)

        delta_opacity = gaussian._opacity - features_opacity
        delta_opacity = delta_opacity.renorm(p=2, dim=0, maxnorm=epsilon)
        gaussian._opacity.copy_(features_opacity + delta_opacity)

def gaussian_scaling_l2_attack(gaussian, alpha, epsilon, features_scaling):
    with torch.no_grad():
        grad_scaling = gaussian._scaling.grad
        norm_scaling = torch.norm(grad_scaling.view(-1), p=2)

        if norm_scaling > 0:
            f_scaling_eta = alpha * (grad_scaling / norm_scaling)
        else:
            f_scaling_eta = torch.zeros_like(grad_scaling)

        f_scaling_eta.mul_(-1)
        gaussian._scaling.add_(f_scaling_eta)

        delta_scaling = gaussian._scaling - features_scaling
        delta_scaling = delta_scaling.renorm(p=2, dim=0, maxnorm=epsilon)
        gaussian._scaling.copy_(features_scaling + delta_scaling)

def gaussian_color_linf_attack(gaussian, alpha, epsilon, features_rest, features_dc):
    with torch.no_grad():
        f_rest_eta = alpha * torch.sign(gaussian._features_rest.grad)
        f_dc_eta = alpha * torch.sign(gaussian._features_dc.grad)

        # Perform the adversarial update
        f_rest_eta.mul_(-1)  # Targeted attack adjustment
        f_dc_eta.mul_(-1)

        # Update features in-place
        gaussian._features_rest.add_(f_rest_eta)
        gaussian._features_dc.add_(f_dc_eta)

        # Clamp the values within the range
        gaussian._features_rest.sub_(features_rest).clamp_(-epsilon, epsilon).add_(features_rest)
        gaussian._features_dc.sub_(features_dc).clamp_(-epsilon, epsilon).add_(features_dc)

def gaussian_color_l2_attack(gaussian, alpha, epsilon, features_rest, features_dc):
    with torch.no_grad():
        # Compute the L2 norm of the gradients
        grad_rest = gaussian._features_rest.grad
        grad_dc = gaussian._features_dc.grad

        norm_rest = torch.norm(grad_rest.view(-1), p=2)
        norm_dc = torch.norm(grad_dc.view(-1), p=2)

        # Avoid division by zero
        if norm_rest > 0:
            f_rest_eta = alpha * (grad_rest / norm_rest)
        else:
            f_rest_eta = torch.zeros_like(grad_rest)

        if norm_dc > 0:
            f_dc_eta = alpha * (grad_dc / norm_dc)
        else:
            f_dc_eta = torch.zeros_like(grad_dc)

        # Targeted attack adjustment
        f_rest_eta.mul_(-1)
        f_dc_eta.mul_(-1)

        # Apply the perturbations
        gaussian._features_rest.add_(f_rest_eta)
        gaussian._features_dc.add_(f_dc_eta)

        # Clamp the values within the L2 ball
        delta_rest = gaussian._features_rest - features_rest
        delta_rest = delta_rest.renorm(p=2, dim=0, maxnorm=epsilon)
        gaussian._features_rest.copy_(features_rest + delta_rest)

        delta_dc = gaussian._features_dc - features_dc
        delta_dc = delta_dc.renorm(p=2, dim=0, maxnorm=epsilon)
        gaussian._features_dc.copy_(features_dc + delta_dc)


def gaussian_color_linf_attack_masked(gaussians, mask3d, alpha, epsilon):
    with torch.no_grad():
        mask3d = mask3d.bool()
        # Compute the perturbations for the entire feature tensors
        f_rest_eta = alpha * torch.sign(gaussians._features_rest.grad)
        f_dc_eta = alpha * torch.sign(gaussians._features_dc.grad)

        # Perform adversarial update
        f_rest_eta.mul_(-1)  # Targeted attack adjustment
        f_dc_eta.mul_(-1)

        # Apply the mask to the perturbations
        f_rest_eta_masked = f_rest_eta[mask3d]
        f_dc_eta_masked = f_dc_eta[mask3d]

        # Update only the masked features in-place
        gaussians._features_rest[mask3d].add_(f_rest_eta_masked)
        gaussians._features_dc[mask3d].add_(f_dc_eta_masked)

        # Clamp the values within the range
        gaussians._features_rest[mask3d].sub_(original_features_rest[mask3d]).clamp_(-epsilon, epsilon).add_(original_features_rest[mask3d])
        gaussians._features_dc[mask3d].sub_(original_features_dc[mask3d]).clamp_(-epsilon, epsilon).add_(original_features_dc[mask3d])

        # Optionally clamp the values to [0, 1] range to maintain valid colors
        # gaussians._features_rest.clamp_(0, 1)
        # gaussians._features_dc.clamp_(0, 1)


# def main(args):
@hydra.main(version_base=None, config_path="configs", config_name="config")
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
    dataset.cam_indices = cfg.scene.cam_indices  # select specific cameras instead of loading all of them.
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
    batch_mode = cfg.batch_mode  # Set this to False for single camera mode
    batch_size = cfg.batch_size # only used if batch_mode is True
    is_targeted = cfg.scene.is_targeted


    # cleanup render and preds directories
    subprocess.run(["make", "clean"], shell=True)
        
    # load detector
    detector = load_detector(cfg)
    detector.load_model()
    
    if isinstance(cfg.scene.target, str):
        cfg.scene.target = [detector.resolve_label_index(cfg.scene.target)]
        target = torch.tensor(cfg.scene.target, device=cfg.data_device)

    selected_obj_ids = torch.tensor(cfg.selected_obj_ids, device=cfg.data_device)
    if isinstance(cfg.scene.untarget, str):
        cfg.scene.untarget = detector.resolve_label_index(cfg.scene.untarget)
        untarget = torch.tensor(cfg.scene.untarget, device=cfg.data_device)
    elif cfg.scene.untarget is not None:
        untarget = torch.tensor(cfg.scene.untarget, device=cfg.data_device)
    else:
        untarget = None
    
    if dataset.no_groups == False and dataset.combine_splats == False:
        # Load Gaussian Splat model
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.training_setup(opt)
        scene = Scene(args=dataset, gaussians=gaussians,load_iteration=30000, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)     

        # assume groups are being used. The classifier is a pretrained model from 3DGS grouping training
        # that predicts the segmentations / groups within the scene
        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth"),map_location=DEVICE))
 
        with torch.no_grad():
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d, dim=0)
            mask = prob_obj3d[selected_obj_ids, :, :] > select_thresh
            mask3d = mask.any(dim=0).squeeze()
            print("Calculating convex hull of Gaussian group")
            mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
            print("Finished calculating convex hull")
            mask3d = torch.logical_or(mask3d,mask3d_convex)
            mask3d = mask3d.float()[:,None,None]
            #mask3d = mask3d.squeeze()
        del classifier
        torch.cuda.empty_cache()
        
        # copy gaussians variable to new object
        gaussians_original = copy.deepcopy(gaussians)
        gaussians_original.removal_setup(opt,mask3d) # inverse 
        gaussians.removal_setup(opt,~mask3d.bool())
    
    elif dataset.no_groups == True and dataset.combine_splats == False:
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

    elif dataset.combine_splats == True:
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.training_setup(opt)
        scene = Scene(args=dataset, gaussians=gaussians,load_iteration=-2, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
        # List of .ply file paths to be combined
        ply_paths = cfg.scene.combine_splats_paths
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

    
    print("Setup complete. Running the pipeline...")
    # select feature that we want to attack
    manipulable_features = ["features_rest", "features_dc", "xyz", "scaling", "opacity"]
    original_features_rest = gaussians._features_rest.clone().detach().requires_grad_(True)
    original_features_dc = gaussians._features_dc.clone().detach().requires_grad_(True)
    original_features_xyz = gaussians._xyz.clone().detach().requires_grad_(True)
    original_features_scaling = gaussians._scaling.clone().detach().requires_grad_(True)
    original_features_opacity = gaussians._opacity.clone().detach().requires_grad_(True)
    original_features_rotation = gaussians._rotation.clone().detach().requires_grad_(True)    

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0, 0 ]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # viewpoint_stack = scene.getTrainCameras().copy()  # use all cameras

    # single camera or range of cameras
    viewpoint_stack = scene.getTrainCameras().copy()#[start_cam:end_cam] 
    

    for i in range(1, add_cams):

        # make a new camera with a view that we choose. 
        camera = copy.deepcopy(viewpoint_stack[0])
        # Shift left
        # T = camera.T
        # T[0] += shift_amount * i
        # camera.transform(T)
        # yaw right 
        camera.yaw(7*i)

        viewpoint_stack.append(camera)

    total_views = len(viewpoint_stack)
    # Create a pending list of cameras for batching
    pending_views = viewpoint_stack.copy()

    # get benign render bboxes - would be better if you could SOLO render the target!
    # for each benign render, get the bbox w/ detection of target class.
    bboxes = []
    for i, cam in enumerate(viewpoint_stack):
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
        bbox = pil_img_bw.getbbox()        
        pil_img.save(img_path)

        rendered_img_input = detector.preprocess_input(img_path)
        bboxes.append(bbox)

        draw = PIL.ImageDraw.Draw(pil_img_bw)
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), "object", fill="red")
        pil_img_bw.save(f'renders/bw/bbox_render_{i}.jpg')    
        bbox = np.expand_dims(np.array(bbox), axis=0)

    gt_bboxes = np.array(bboxes)
    pending_bboxes = gt_bboxes.copy()

    for it in range(1500):
        renders = []
        
        if batch_mode:
            # Select current batch
            current_batch = pending_views[:cfg.batch_size]
            current_bboxes = pending_bboxes[:cfg.batch_size]
            renders = []
            for cam in current_batch:
                render_pkg = render(cam, gaussians, pipe, bg)
                renders.append(render_pkg["render"])
            renders = torch.stack(renders)
            loss = detector.infer(renders, target=target, bboxes=current_bboxes, batch_size=len(current_batch))
        else:
            cam = viewpoint_stack[0]
            render_pkg = render(cam, gaussians, pipe, bg)
            renders.append(render_pkg["render"])
            renders = torch.stack(renders)
            loss = detector.infer(renders, target=target, bboxes=gt_bboxes[0], batch_size=renders.shape[0])

        print(f"Iteration: {it}, Loss: {loss}")
        loss.backward()

        if gaussians._features_rest.grad is not None and gaussians._features_dc.grad is not None:
            epsilon = cfg.epsilon
            alpha = cfg.alpha
            gaussian_color_l2_attack(gaussians, alpha, epsilon, original_features_rest, original_features_dc)
            # gaussian_position_l2_attack(gaussians, alpha, epsilon, original_features_xyz)
            # gaussian_scaling_l2_attack(gaussians, alpha, epsilon, original_features_scaling)
            # gaussian_rotation_l2_attack(gaussians, alpha, epsilon, original_features_rotation)
            # gaussian_opacity_l2_attack(gaussians, alpha, epsilon, original_features_opacity)
            
            # Uncomment the following lines to apply different attacks
            # gaussian_color_linf_attack(gaussians, alpha, epsilon, original_features_rest, original_features_dc)
            # gaussian_position_linf_attack(gaussians, alpha, epsilon, original_features_xyz)
            # gaussian_scaling_linf_attack(gaussians, alpha, epsilon, original_features_scaling)
            # gaussian_rotation_linf_attack(gaussians, alpha, epsilon, original_features_rotation)
            # gaussian_opacity_linf_attack(gaussians, alpha, epsilon, original_features_opacity)
            # gaussian_scaling_linf_attack(gaussians, alpha, epsilon)

            combined_gaussians = copy.deepcopy(gaussians)
            combined_gaussians.concat_setup("features_rest", gaussians_original._features_rest, True)
            combined_gaussians.concat_setup("features_dc", gaussians_original._features_dc, True)
            combined_gaussians.concat_setup("xyz", gaussians_original._xyz, True)
            combined_gaussians.concat_setup("scaling", gaussians_original._scaling, True)
            combined_gaussians.concat_setup("opacity", gaussians_original._opacity, True)
            combined_gaussians.concat_setup("rotation", gaussians_original._rotation, True)
            combined_gaussians.concat_setup("objects_dc", gaussians_original._objects_dc, True)

            concat_renders = []
            if batch_mode:
                for cam in current_batch:
                    render_pkg = render(cam, combined_gaussians, pipe, bg)
                    concat_renders.append(render_pkg["render"])
            else:
                cam = viewpoint_stack[0]
                render_pkg = render(cam, combined_gaussians, pipe, bg)
                concat_renders.append(render_pkg["render"])

            if batch_mode:
                successes = []
                for j, cam in enumerate(current_batch):
                    img_path = f"renders/render_concat_{j}.png"
                    cr = concat_renders[j]
                    preds_path = "preds"
                    Image.fromarray((torch.clamp(cr, min=0, max=1.0) * 255)
                                    .byte()
                                    .permute(1, 2, 0)
                                    .contiguous()
                                    .cpu()
                                    .numpy()).save(img_path)

                    rendered_img_input = detector.preprocess_input(img_path)
                    success = detector.predict_and_save(
                        image=cr,
                        path=os.path.join(preds_path, f'render_it{it}_c{j}.png'),
                        target=target,
                        untarget=untarget,
                        is_targeted=is_targeted,
                        threshold=attack_conf_thresh,
                        gt_bbox=current_bboxes[j]
                    )
                    successes.append(success)
                num_successes = sum(successes)
                print(f"Successes: {num_successes}/{len(current_batch)}")
                if num_successes == len(current_batch)-1 or num_successes == len(current_batch):
                    print("Current batch attacked successfully")
                    # Remove the successful batch from pending lists
                    pending_views = pending_views[len(current_batch):]
                    pending_bboxes = pending_bboxes[len(current_batch):]
                    # If no more cameras remain, finish the attack
                    if len(pending_views) == 0:
                        print("All camera viewpoints attacked successfully")
                        print("saving gaussians")
                        combined_gaussians.save_ply(os.path.join("output", f"{cfg.scene.name}_{it}.ply"))
                        break
            else:
                img_path = f"renders/render_concat_0.png"
                cr = concat_renders[0]
                preds_path = "preds"
                Image.fromarray((torch.clamp(cr, min=0, max=1.0) * 255)
                                .byte()
                                .permute(1, 2, 0)
                                .contiguous()
                                .cpu()
                                .numpy()).save(img_path)

                rendered_img_input = detector.preprocess_input(img_path)
                success = detector.predict_and_save(
                        image=cr,
                        path=os.path.join(preds_path, f'render_it{it}_c{total_views-len(viewpoint_stack)}.png'),
                        target=target,
                        untarget=untarget,
                        is_targeted=is_targeted,
                        threshold=attack_conf_thresh,
                        gt_bbox = gt_bboxes[0]
                    )
                if not batch_mode and success:
                    viewpoint_stack.pop(0)
                    gt_bboxes = np.delete(gt_bboxes, 0, axis=0)
                    if len(viewpoint_stack) == 0:
                        print ("All camera viewpoints attacked successfully")
                        print("saving gaussians")
                        combined_gaussians.save_ply(os.path.join("output", f"{cfg.scene.name}_{it}.ply"))
                        break
                print(f"Success: {success}")
        del combined_gaussians
        gaussians.optimizer.zero_grad(set_to_none=True)
        detector.zero_grad()

if __name__ == "__main__":
    run()
