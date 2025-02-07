import torch
import copy
from torch import nn
from scene.gaussian_model import GaussianModel
from arguments import GroupParams

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


def main():
    DEVICE = f"cuda:7" 
    torch.cuda.set_device(DEVICE)    
    # Initialize the GaussianModel with a specific SH degree
    gaussians = GaussianModel(sh_degree=2)

    # List of .ply file paths to be combined
    ply_paths = [
        "output/bike/point_cloud_109.ply",
        "output/bike/point_cloud_302_5.ply"
    ]

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

    # make a mask3D
    obj_1_mask3d = obj_1_mask.view(1, obj_1_mask.shape[0], 1)
    obj_1_mask3d = obj_1_mask3d.any(dim=0).squeeze()
    
    obj_1_mask3d = obj_1_mask3d.float()[:, None, None]

    original_gaussians = copy.deepcopy(gaussians)
    # Updated apply_mask function to handle mask correctly
    original_gaussians.removal_setup(opt, obj_1_mask3d) # inverse 
    gaussians.removal_setup(opt, ~obj_1_mask3d.bool())

if __name__ == "__main__":
    main()
