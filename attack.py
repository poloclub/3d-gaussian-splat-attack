import torch
import numpy as np
import sys
import argparse
import os
from random import randint
from model import detectron2_model, dt2_input, save_adv_image_preds, model_input, get_instances_bboxes
from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera  
from arguments import GroupParams
import subprocess
from PIL import Image
from tqdm import tqdm
from edit_object_removal import points_inside_convex_hull
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import copy
# sys.path.append("submodules/gaussian-splatting")
#from scene import Scene, GaussianModel
dataset = GroupParams()
dataset.data_device = 'cuda'
dataset.eval = False
dataset.images = 'images'
dataset.model_path = f"/raid/mhull32/gaussian-grouping/output/road_sign"
dataset.n_views = 100
dataset.num_classes = 256
dataset.object_path = 'object_mask'
dataset.random_init = False
dataset.resolution = 1
dataset.sh_degree = 3
dataset.source_path = f"/raid/mhull32/gaussian-grouping/data/road_sign"
dataset.train_split = False
dataset.white_background = False

opt = GroupParams()
opt.densification_interval = 100
opt.density_From_iter = 500
opt.densify_grad_threshold = 0.0002
opt.density_until_iter = 15000
opt.feature_lr = 0.0025
opt.iterations = 30000
opt.lambda_dssim = 0.2
opt.opacity_lr = 0.05
opt.opacity_reset_interval = 3000
opt.percent_dense = 0.01
opt.position_lr_delay_mult = 0.01
opt.position_lr_final = 1.6e-06
opt.position_lr_init = 0.00016
opt.position_lr_max_steps = 30000
opt.reg3d_interval = 2
opt.reg3d_k = 5
opt.reg3d_lambda_val = 2
opt.reg3d_max_points = 300000
opt.reg3d_sample_size = 1000
#opt.random_background = False
opt.rotation_lr = 0.001
opt.scaling_lr = 0.005

pipe = GroupParams()
pipe.compute_cov3D_python = False
pipe.convert_SHs_python = False
pipe.debug = False

def gaussian_position_linf_attack(gaussian, alpha, epsilon):
    with torch.no_grad():
        f_xyz_eta = alpha * torch.sign(gaussian._xyz.grad)
        f_xyz_eta.mul_(-1) # comment out if desire Untargeted
        gaussian._xyz.add_(f_xyz_eta)
        gaussian._xyz.sub_(original_features_xyz).clamp_(-epsilon, epsilon).add_(original_features_xyz)

def gaussian_rotation_linf_attack(gaussian, alpha, epsilon):
    with torch.no_grad():
        f_rotation_eta = alpha * torch.sign(gaussian._rotation.grad)
        f_rotation_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._rotation.add_(f_rotation_eta)
        gaussian._rotation.sub_(original_features_rotation).clamp_(-epsilon, epsilon).add_(original_features_rotation)

def gaussian_opacity_linf_attack(gaussian, alpha, epsilon):
    with torch.no_grad():
        f_opacity_eta = alpha * torch.sign(gaussian._opacity.grad)
        f_opacity_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._opacity.add_(f_opacity_eta)
        gaussian._opacity.sub_(original_features_opacity).clamp_(-epsilon, epsilon).add_(original_features_opacity)

def gaussian_scaling_linf_attack(gaussian, alpha, epsilon):
    with torch.no_grad():
        f_scaling_eta = alpha * torch.sign(gaussian._scaling.grad)
        f_scaling_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._scaling.add_(f_scaling_eta)
        gaussian._scaling.sub_(original_features_scaling).clamp_(-epsilon, epsilon).add_(original_features_scaling)

def gaussian_color_linf_attack(gaussian, alpha, epsilon):
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
        gaussian._features_rest.sub_(original_features_rest).clamp_(-epsilon, epsilon).add_(original_features_rest)
        gaussian._features_dc.sub_(original_features_dc).clamp_(-epsilon, epsilon).add_(original_features_dc)


def gaussian_color_linf_attack_masked(gaussians, mask3d, alpha, epsilon):
    with torch.no_grad():
        mask3d = mask3d.bool()
        # Compute the perturbations for the entire feature tensors
        f_rest_eta = alpha * torch.sign(gaussians._features_rest.grad)
        f_dc_eta = alpha * torch.sign(gaussians._features_dc.grad)

        # Perform the adversarial update
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

 

if __name__ == "__main__":

    # cleanup render and preds directories
    subprocess.run(["make", "clean"], shell=True)
        
    # detectron2 
    model, dt2_config = detectron2_model()
    
    # Load Gaussian Splat model
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    scene = Scene(dataset, gaussians,load_iteration=30000, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))
    selected_obj_ids = torch.tensor([175], device='cuda')
    select_thresh = 0.3
    with torch.no_grad():
        logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
        prob_obj3d = torch.softmax(logits3d, dim=0)
        mask = prob_obj3d[selected_obj_ids, :, :] > select_thresh
        mask3d = mask.any(dim=0).squeeze()
        print("calculating convex hull")
        mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(),mask3d,outlier_factor=1.0)
        print("finished calculating convex hull")
        mask3d = torch.logical_or(mask3d,mask3d_convex)
        mask3d = mask3d.float()[:,None,None]
        #mask3d = mask3d.squeeze()
    del classifier
    torch.cuda.empty_cache()
    
    # copy gaussians variable to new object
    gaussians_original = copy.deepcopy(gaussians)
    gaussians_original.removal_setup(opt,mask3d) # inverse 
    gaussians.removal_setup(opt,~mask3d.bool())
    # select feature that we want to attack
    manipulable_features = ["features_rest", "features_dc", "xyz", "scaling", "opacity"]
    original_features_rest = gaussians._features_rest.clone().detach().requires_grad_(True)
    original_features_dc = gaussians._features_dc.clone().detach().requires_grad_(True)
    # original_features_xyz = gaussians._xyz.clone().detach().requires_grad_(True)
    # original_features_scaling = gaussians._scaling.clone().detach().requires_grad_(True)
    # original_features_opacity = gaussians._opacity.clone().detach().requires_grad_(True)
    # original_features_rotation = gaussians._rotation.clone().detach().requires_grad_(True)    

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # viewpoint_stack = scene.getTrainCameras().copy()  # use all cameras
    # cam_idx = 49
    start_cam = 43
    end_cam = 44
    # single camera or range of cameras
    viewpoint_stack = scene.getTrainCameras().copy()[start_cam:end_cam] 
    
    # create sequence of cameras with nearby views
    add_cams = 5
    shift_amount = 0.35  # Adjust this value based on how far you want to shift
    for i in range(1, add_cams):
        camera = copy.deepcopy(viewpoint_stack[0])

        # Shift right
        T = camera.T
        T[0] += shift_amount * i
        camera.update_transform(T)
        
        viewpoint_stack.append(camera)

    total_views = len(viewpoint_stack)

    # get benign render bboxes - would be better if you could SOLO render the target!
    # for each benign render, get the bbox w/ detection of target class.
    bboxes = []
    for i, cam in enumerate(viewpoint_stack):
        render_pkg = render(cam, gaussians, pipe, bg)
        img_path = f"renders/render_{i}.png"
        Image.fromarray((torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()).save(img_path)
        rendered_img_input = dt2_input(img_path)
        bbox = get_instances_bboxes(model, rendered_img_input, target = 11, threshold=0.2)
        bboxes.append(bbox)

    gt_bboxes = np.array(bboxes)
    batch_mode = False  # Set this to False for single camera mode

    for it in range(1000):
        renders = []
        if batch_mode:
            for cam in viewpoint_stack:
                render_pkg = render(cam, gaussians, pipe, bg)
                renders.append(render_pkg["render"])
            renders = torch.stack(renders)
            target = torch.tensor([5])  # bus
            loss = model_input(model, renders, target=target, bboxes=gt_bboxes, batch_size=renders.shape[0])
        else:
            cam = viewpoint_stack[0]
            render_pkg = render(cam, gaussians, pipe, bg)
            renders.append(render_pkg["render"])
            renders = torch.stack(renders)
            target = torch.tensor([5])  # bus
            loss = model_input(model, renders, target=target, bboxes=np.expand_dims(gt_bboxes[0],axis=0), batch_size=renders.shape[0])

        print(f"Loss: {loss}")
        loss.backward(retain_graph=True)

        if gaussians._features_rest.grad is not None and gaussians._features_dc.grad is not None:
            epsilon = 5.0
            alpha = 0.001
            gaussian_color_linf_attack(gaussians, alpha, epsilon)

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
                for cam in viewpoint_stack:
                    render_pkg = render(cam, combined_gaussians, pipe, bg)
                    concat_renders.append(render_pkg["render"])
            else:
                cam = viewpoint_stack[0]
                render_pkg = render(cam, combined_gaussians, pipe, bg)
                concat_renders.append(render_pkg["render"])

            if batch_mode:
                for j, cam in enumerate(viewpoint_stack):
                    img_path = f"renders/render_concat_{j}.png"
                    cr = concat_renders[j]
                    preds_path = "preds"
                    Image.fromarray((torch.clamp(cr, min=0, max=1.0) * 255)
                                    .byte()
                                    .permute(1, 2, 0)
                                    .contiguous()
                                    .cpu()
                                    .numpy()).save(img_path)

                    rendered_img_input = dt2_input(img_path)
                    success = save_adv_image_preds(
                        model, dt2_config, input=rendered_img_input,
                        instance_mask_thresh=0.4,
                        target=target, untarget=None, is_targeted=True,
                        path=os.path.join(preds_path, f'render_it{it}_c{j}.png')
                    )
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

                rendered_img_input = dt2_input(img_path)
                success = save_adv_image_preds(
                    model, dt2_config, input=rendered_img_input,
                    instance_mask_thresh=0.4,
                    target=target, untarget=None, is_targeted=True,
                    path=os.path.join(preds_path, f'render_it{it}_c{total_views-len(viewpoint_stack)}.png')
                )
                if not batch_mode and success:
                    viewpoint_stack.pop(0)
                    gt_bboxes = np.delete(gt_bboxes, 0, axis=0)
                    if len(viewpoint_stack) == 0:
                        print ("All cameras attacked successfully")
                        break
        del combined_gaussians
        gaussians.optimizer.zero_grad(set_to_none=True)
        model.zero_grad()
        print(f"Success: {success}")
