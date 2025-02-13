import torch
import numpy as np
import sys
import argparse
import os
from random import randint
from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera  
from arguments import GroupParams
import subprocess
import PIL
from PIL import Image
from tqdm import tqdm
from edit_object_removal import points_inside_convex_hull
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import copy

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
opt.random_background = False
opt.rotation_lr = 0.001
opt.scaling_lr = 0.005


def parse_args():
    parser = argparse.ArgumentParser(description="Refactor Gaussian Adversarial Attack Script")

    # Dataset parameters
    parser.add_argument("--data_device", type=str, default="cuda", help="Device to use for data processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
    parser.add_argument("--eval", action="store_true", help="Set evaluation mode")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views")
    parser.add_argument("--num_classes", type=int, default=256, help="Number of classes")
    parser.add_argument("--object_path", type=str, required=True, help="Path to object masks")
    parser.add_argument("--random_init", action="store_true", help="Enable random initialization")
    parser.add_argument("--resolution", type=int, default=1, help="Resolution factor")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree")
    parser.add_argument("--source_path", type=str, required=True, help="Path to data source")
    parser.add_argument("--train_split", action="store_true", help="Use train split")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--no-groups", action="store_true", help="Model without Gaussian groups")

    # Optimization parameters
    parser.add_argument("--densification_interval", type=int, default=100, help="Interval for densification")
    parser.add_argument("--density_From_iter", type=int, default=500, help="Starting iteration for density")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002, help="Densify gradient threshold")
    parser.add_argument("--density_until_iter", type=int, default=15000, help="Density until iteration")
    parser.add_argument("--feature_lr", type=float, default=0.0025, help="Feature learning rate")
    parser.add_argument("--iterations", type=int, default=30000, help="Number of optimization iterations")
    parser.add_argument("--lambda_dssim", type=float, default=0.2, help="Lambda for DSSIM loss")
    parser.add_argument("--opacity_lr", type=float, default=0.05, help="Learning rate for opacity")
    parser.add_argument("--opacity_reset_interval", type=int, default=3000, help="Interval to reset opacity")
    parser.add_argument("--percent_dense", type=float, default=0.01, help="Percent density")
    parser.add_argument("--position_lr_delay_mult", type=float, default=0.01, help="Position learning rate delay multiplier")
    parser.add_argument("--position_lr_final", type=float, default=1.6e-6, help="Final position learning rate")
    parser.add_argument("--position_lr_init", type=float, default=0.00016, help="Initial position learning rate")
    parser.add_argument("--position_lr_max_steps", type=int, default=30000, help="Max steps for position learning rate")
    parser.add_argument("--reg3d_interval", type=int, default=2, help="Interval for 3D regularization")
    parser.add_argument("--reg3d_k", type=int, default=5, help="K value for 3D regularization")
    parser.add_argument("--reg3d_lambda_val", type=int, default=2, help="Lambda value for 3D regularization")
    parser.add_argument("--reg3d_max_points", type=int, default=300000, help="Max points for 3D regularization")
    parser.add_argument("--reg3d_sample_size", type=int, default=1000, help="Sample size for 3D regularization")
    parser.add_argument("--rotation_lr", type=float, default=0.001, help="Rotation learning rate")
    parser.add_argument("--scaling_lr", type=float, default=0.005, help="Scaling learning rate")

    # Pipeline parameters
    parser.add_argument("--compute_cov3D_python", action="store_true", help="Enable computation of covariance in Python")
    parser.add_argument("--convert_SHs_python", action="store_true", help="Enable SH conversion in Python")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode")

    # Attack options
    parser.add_argument("--select_thresh", type=float, default=0.5, help="Selection threshold for Gaussian group")
    parser.add_argument("--start_cam", type=int, default=0, help="Start index for camera views")
    parser.add_argument("--end_cam", type=int, default=1, help="End index for camera views")
    parser.add_argument("--add_cams", type=int, default=1, help="Number of additional cameras")
    parser.add_argument("--shift_amount", type=float, default=0.15, help="Shift amount for additional cameras")
    parser.add_argument("--attack_conf_thresh", type=float, default=0.7, help="Confidence threshold for attack")
    parser.add_argument("--batch_mode", action="store_true", help="Enable batch mode")
    parser.add_argument("--cam_indices", type=int, nargs="+", required=False, help="Select specific cameras")

    return parser.parse_args()


def main(args):
    DEVICE = f"cuda:{args.device}" 
    torch.cuda.set_device(DEVICE)    
    
    dataset = GroupParams()
    dataset.data_device = args.data_device
    dataset.eval = args.eval
    dataset.images = args.images
    dataset.model_path = args.model_path
    dataset.n_views = args.n_views
    dataset.num_classes = args.num_classes
    dataset.object_path = args.object_path
    dataset.random_init = args.random_init
    dataset.resolution = args.resolution
    dataset.sh_degree = args.sh_degree
    dataset.source_path = args.source_path
    dataset.train_split = args.train_split
    dataset.white_background = args.white_background
    dataset.device = args.device
    dataset.cam_indices = args.cam_indices # select specific cameras instead of loading all of them.
    dataset.no_groups = args.no_groups
    # Pipeline parameters
    pipe = GroupParams()
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False
    pipe.debug = False

    # Initialize the GaussianModel with a specific SH degree    
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    scene = Scene(args=dataset, gaussians=gaussians,load_iteration=-2, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
    # List of .ply file paths to be combined
    # ply_paths = [
    #     "output/bike/point_cloud_302_5.ply",
    #     "output/bike/point_cloud_109.ply",
    # ]
    ply_paths = [
        "output/room/plain_room.ply",
        "output/room/truck.ply",
    ]    

    # Combine the .ply files
    gaussians.combine_splats(ply_paths)

    # Demonstrate how to extract obj_1 and obj_2 using self.masks
    # obj_1_mask = gaussians.masks[0]
    # obj_2_mask = gaussians.masks[1]

    # # Pad obj_1_mask with the shape of obj_2_mask
    # pad_size = obj_2_mask.shape[0]
    # if pad_size > 0:
    #     padding = torch.zeros(pad_size, dtype=torch.bool, device=obj_1_mask.device)
    #     obj_1_mask = torch.cat((obj_1_mask, padding), dim=0)

    # # make a mask3D
    # obj_1_mask3d = obj_1_mask.view(1, obj_1_mask.shape[0], 1)
    # obj_1_mask3d = obj_1_mask3d.any(dim=0).squeeze()
    
    # obj_1_mask3d = obj_1_mask3d.float()[:, None, None]

    # original_gaussians = copy.deepcopy(gaussians)
    # # Updated apply_mask function to handle mask correctly
    # original_gaussians.removal_setup(opt, obj_1_mask3d) # inverse 
    # gaussians.removal_setup(opt, ~obj_1_mask3d.bool())

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # single camera or range of cameras
    viewpoint_stack = scene.getTrainCameras().copy()[0:] 
    render_pkg = render(viewpoint_stack[0], gaussians, pipe, bg)
    img_path = "renders/combined_splats/combined_splats.png"
    Image.fromarray((torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()).save(img_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
