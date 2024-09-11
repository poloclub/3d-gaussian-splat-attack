import torch
import numpy as np
import sys
import argparse
import os
from random import randint
from model import detectron2_model, dt2_input, save_adv_image_preds, model_input
sys.path.append("submodules/gaussian-splatting")
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from gaussian_renderer import render  # This is the rendering function from __init__.py
from scene.cameras import Camera  
from arguments import GroupParams
import subprocess
from PIL import Image
from tqdm import tqdm

splat = 'road_sign'

# copy stuff from experiment configs for now just to run the renderer.
dataset = GroupParams()
dataset.data_device = 'cuda'
dataset.eval = False
dataset.images = 'images'
dataset.model_path = f'./splats/{splat}'
dataset.resolution = -1
dataset.sh_degree = 3
dataset.source_path = f"C:\\Users\\matth\\Documents\\3D-Gaussian-Splat-Attack\\splats\\{splat}"
dataset.white_background = False

opt = GroupParams()
opt.densification_interval = 10
opt.density_From_iter = 500
opt.densify_grad_threshold = 0.0002
opt.density_until_iter = 15000
#opt.feature_lr = 0.0025
opt.feature_lr = 0.01
opt.iterations = 30000
opt.lambda_dssim = 0.2
opt.opacity_lr = 0.05
opt.opacity_reset_interval = 3000
opt.percent_dense = 0.01
opt.position_lr_delay_mult = 0.01
opt.position_lr_final = 1.6e-06
opt.position_lr_init = 0.00016
opt.position_lr_max_steps = 30000
opt.random_background = False
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
        f_scaling_eta = alpha * torch.sign(gaussian._scaling.grad)
        f_scaling_eta.mul_(-1)  # Targeted attack adjustment
        gaussian._scaling.add_(f_scaling_eta)
        gaussian._scaling.sub_(original_features_scaling).clamp_(-epsilon, epsilon).add_(original_features_scaling)

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
    original_features_rest = gaussians._features_rest.clone().detach().requires_grad_(True)
    original_features_dc = gaussians._features_dc.clone().detach().requires_grad_(True)
    original_features_xyz = gaussians._xyz.clone().detach().requires_grad_(True)
    original_features_scaling = gaussians._scaling.clone().detach().requires_grad_(True)
    original_features_opacity = gaussians._opacity.clone().detach().requires_grad_(True)
    original_features_scaling = gaussians._scaling.clone().detach().requires_grad_(True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    # viewpoint_stack = scene.getTrainCameras().copy()  # use all cameras
    cam_idx = 49
    viewpoint_stack = [scene.getTrainCameras().copy()[cam_idx]] # single camera
    total_views = len(viewpoint_stack)

    for _ in tqdm(range(total_views), desc="Rendering viewpoints"):
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        
        for i in range(1000):
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            gaussians._features_rest.requires_grad_()
            gaussians._features_dc.requires_grad_()

            # hard code the sign bbox, and target label for image at camera 49
            sign_bbox = np.array([[49.1297, 295.2162, 773.3174, 1033.7278]])
            # target = torch.tensor([32])  # sports ball
            target = torch.tensor([5])  # bus
            # target = torch.tensor([11]) # stop sign

            # loss wrt to bbox and target
            loss = model_input(model, render_pkg["render"], target=target, bboxes=sign_bbox, batch_size=1)
            print(f"Loss: {loss}")
            loss.backward(retain_graph=True)

            if gaussians._features_rest.grad is not None and gaussians._features_dc.grad is not None:
                epsilon = 5.0
                alpha = 0.01
                gaussian_color_linf_attack(gaussians, alpha, epsilon)
                # gaussian_position_linf_attack(gaussians, alpha, epsilon)
                # gaussian_rotation_linf_attack(gaussians, alpha, epsilon)
                # gaussian_opacity_linf_attack(gaussians, alpha, epsilon)
                # gaussian_scaling_linf_attack(gaussians, alpha, epsilon)

            # Render the splat from the chosen camera and predict. Save the image and preds
            img_path = f"renders/render_{total_views - len(viewpoint_stack)}.png"
            preds_path = "preds"
            Image.fromarray((torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
                            .byte()
                            .permute(1, 2, 0)
                            .contiguous()
                            .cpu()
                            .numpy()).save(img_path)

            rendered_img_input = dt2_input(img_path)
            success = save_adv_image_preds(
                model, dt2_config, input=rendered_img_input,
                instance_mask_thresh=0.9,
                target=target, untarget=None, is_targeted=True,
                path=os.path.join(preds_path, f'render_c{cam_idx}_it{i}.png')
            )

            print(f"Success: {success}")
