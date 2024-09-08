import torch
import numpy as np
import sys
import argparse
from random import randint
sys.path.append("submodules/gaussian-splatting")
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from gaussian_renderer import render  # This is the rendering function from __init__.py
from arguments import GroupParams
from PIL import Image
from tqdm import tqdm

# copy stuff from experiment configs for now just to run the renderer.
dataset = GroupParams()
dataset.data_device = 'cuda'
dataset.eval = False
dataset.images = 'images'
dataset.model_path = './splats/fencer'
dataset.resolution = -1
dataset.sh_degree = 3
dataset.source_path = 'C:\\Users\\matth\\Documents\\3D-Gaussian-Splat-Attack\\splats\\fencer'
dataset.white_background = False

opt = GroupParams()
opt.densification_interval = 10
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
opt.random_background = False
opt.rotation_lr = 0.001
opt.scaling_lr = 0.005

pipe = GroupParams()
pipe.compute_cov3D_python = False
pipe.convert_SHs_python = False
pipe.debug = False

# Load a Gaussian splat model from a .ply file
def load_gaussian_model(ply_file_path):
    gaussian_model = GaussianModel(sh_degree=3)  # Initialize Gaussian model with SH degree 3
    gaussian_model.load_ply(ply_file_path)       # Load the splat model from .ply file
    return gaussian_model

# Setup the camera parameters for rendering
def setup_camera():
    camera = {
        "FoVx": 60,  # Field of view in x direction
        "FoVy": 60,  # Field of view in y direction
        "image_height": 512,
        "image_width": 512,
        "camera_center": [0, 0, 5],  # Camera position in world space
        "world_view_transform": torch.eye(4),  # Identity matrix for simplicity
        "full_proj_transform": torch.eye(4)    # Identity matrix for simplicity
    }
    return camera

# Render the Gaussian splat and handle the output
def render_gaussian_model(gaussian_model, output_image_path):
    camera = setup_camera()  # Set up the camera
    bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).cuda()  # White background

    # Render the Gaussian splat
    render_pkg = render(viewpoint_camera=camera, pc=gaussian_model, pipe=None, bg_color=bg_color)
    
    # Extract data from render output
    image = render_pkg["render"]                 # The rendered image
    viewspace_points = render_pkg["viewspace_points"]  # Points in view space
    visibility_filter = render_pkg["visibility_filter"]  # Which Gaussians are visible
    radii = render_pkg["radii"]                  # Radii of Gaussians in view space

    # Now you can process or save the image as needed (placeholder for image saving)
    # save_image(output_image_path, image)

    print("Rendering completed.")
    return image

# Main script
if __name__ == "__main__":

    # Load Gaussian Splat model
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    scene = Scene(dataset, gaussians,load_iteration=30000) # very important to specify iteration to load! 

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    viewpoint_stack = scene.getTrainCameras().copy()
    total_views = len(viewpoint_stack)

    for _ in tqdm(range(total_views), desc="Rendering viewpoints"):
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)

        Image.fromarray((torch.clamp(render_pkg["render"], min=0, max=1.0) * 255)\
            .byte() \
            .permute(1, 2, 0)\
            .contiguous()\
            .cpu()\
            .numpy())\
            .save(f"renders/render_{total_views - len(viewpoint_stack)}.png")

