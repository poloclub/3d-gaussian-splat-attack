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
import PIL
from PIL import Image
from tqdm import tqdm
from edit_object_removal import points_inside_convex_hull
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import copy


import torch
import argparse
import os
import copy
from random import randint
from model import detectron2_model, dt2_input, save_adv_image_preds, model_input, get_instances_bboxes
from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera  
from arguments import GroupParams
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from edit_object_removal import points_inside_convex_hull
from PIL import Image, ImageDraw
from tqdm import tqdm


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
    parser.add_argument("--selected_obj_ids", type=int, nargs="+", required=True, help="IDs of selected objects")
    parser.add_argument("--select_thresh", type=float, default=0.5, help="Selection threshold for Gaussian group")
    parser.add_argument("--target", type=int, nargs="+", required=True, help="Target object IDs")
    parser.add_argument("--start_cam", type=int, default=0, help="Start index for camera views")
    parser.add_argument("--end_cam", type=int, default=1, help="End index for camera views")
    parser.add_argument("--add_cams", type=int, default=1, help="Number of additional cameras")
    parser.add_argument("--shift_amount", type=float, default=0.15, help="Shift amount for additional cameras")
    parser.add_argument("--attack_conf_thresh", type=float, default=0.7, help="Confidence threshold for attack")

    return parser.parse_args()


# sys.path.append("submodules/gaussian-splatting")
#from scene import Scene, GaussianModel
# dataset = GroupParams()
# dataset.data_device = 'cuda'
# dataset.eval = False
# dataset.images = 'images'
# # dataset.model_path = f"/raid/mhull32/gaussian-grouping/output/road_sign"
# dataset.model_path = f"/raid/mhull32/gaussian-grouping/output/living_room"
# dataset.n_views = 100
# dataset.num_classes = 256
# dataset.object_path = 'object_mask'
# dataset.random_init = False
# dataset.resolution = 1
# dataset.sh_degree = 3
# # dataset.source_path = f"/raid/mhull32/gaussian-grouping/data/road_sign"
# dataset.source_path = f"/raid/mhull32/gaussian-grouping/data/living_room"
# dataset.train_split = False
# dataset.white_background = False

# opt = GroupParams()
# opt.densification_interval = 100
# opt.density_From_iter = 500
# opt.densify_grad_threshold = 0.0002
# opt.density_until_iter = 15000
# opt.feature_lr = 0.0025
# opt.iterations = 30000
# opt.lambda_dssim = 0.2
# opt.opacity_lr = 0.05
# opt.opacity_reset_interval = 3000
# opt.percent_dense = 0.01
# opt.position_lr_delay_mult = 0.01
# opt.position_lr_final = 1.6e-06
# opt.position_lr_init = 0.00016
# opt.position_lr_max_steps = 30000
# opt.reg3d_interval = 2
# opt.reg3d_k = 5
# opt.reg3d_lambda_val = 2
# opt.reg3d_max_points = 300000
# opt.reg3d_sample_size = 1000
# #opt.random_background = False
# opt.rotation_lr = 0.001
# opt.scaling_lr = 0.005

# pipe = GroupParams()
# pipe.compute_cov3D_python = False
# pipe.convert_SHs_python = False
# pipe.debug = False


# attack options:
# modify the object iD to select another item in the scene.
# obj 175 is the road sign in the neighborhood scene
# obj 221 is one of the trucks
# selected_obj_ids = torch.tensor([35], device='cuda')
select_thresh = 0.5 # selected threshold for the gaussian group
# target = torch.tensor([19]) 
# cam_idx = 49
# start_cam = 0
# end_cam = 1
# create sequence of cameras with nearby views
# add_cams = 1
# shift_amount = 0.15  # Adjust this value based on how far you want to shift

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

 

def main(args):
    # Initialize dataset parameters
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
    dataset.white_background = False # args.white_background
    dataset.device = args.device

    # Initialize optimization parameters
    opt = GroupParams()
    opt.densification_interval = args.densification_interval
    opt.density_From_iter = args.density_From_iter
    opt.densify_grad_threshold = args.densify_grad_threshold
    opt.density_until_iter = args.density_until_iter
    opt.feature_lr = args.feature_lr
    opt.iterations = args.iterations
    opt.lambda_dssim = args.lambda_dssim
    opt.opacity_lr = args.opacity_lr
    opt.opacity_reset_interval = args.opacity_reset_interval
    opt.percent_dense = args.percent_dense
    opt.position_lr_delay_mult = args.position_lr_delay_mult
    opt.position_lr_final = args.position_lr_final
    opt.position_lr_init = args.position_lr_init
    opt.position_lr_max_steps = args.position_lr_max_steps
    opt.reg3d_interval = args.reg3d_interval
    opt.reg3d_k = args.reg3d_k
    opt.reg3d_lambda_val = args.reg3d_lambda_val
    opt.reg3d_max_points = args.reg3d_max_points
    opt.reg3d_sample_size = args.reg3d_sample_size
    opt.rotation_lr = args.rotation_lr
    opt.scaling_lr = args.scaling_lr

    # Pipeline parameters
    pipe = GroupParams()
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False
    pipe.debug = False
    # set the cuda device to the args.device
    DEVICE = f"cuda:{args.device}" 
    torch.cuda.set_device(DEVICE)
    # Additional attack setup

    selected_obj_ids = torch.tensor(args.selected_obj_ids, device=args.data_device)
    target = torch.tensor(args.target, device=args.data_device)
    start_cam, end_cam, add_cams = args.start_cam, args.end_cam, args.add_cams
    shift_amount = args.shift_amount
    attack_conf_thresh = args.attack_conf_thresh

    print("Setup complete. Running the pipeline...")

    # cleanup render and preds directories
    subprocess.run(["make", "clean"], shell=True)
        
    # detectron2 
    model, dt2_config = detectron2_model(device=args.device)
    
    # Load Gaussian Splat model
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    scene = Scene(dataset, gaussians,load_iteration=30000, shuffle=False) # very important to specify iteration to load! use -1 for highest iteration
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    classifier.cuda()
    classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth"),map_location=DEVICE))

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
    original_features_xyz = gaussians._xyz.clone().detach().requires_grad_(True)
    original_features_scaling = gaussians._scaling.clone().detach().requires_grad_(True)
    # original_features_opacity = gaussians._opacity.clone().detach().requires_grad_(True)
    # original_features_rotation = gaussians._rotation.clone().detach().requires_grad_(True)    

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # viewpoint_stack = scene.getTrainCameras().copy()  # use all cameras

    # single camera or range of cameras
    viewpoint_stack = scene.getTrainCameras().copy()[start_cam:end_cam] 
    

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

    # get benign render bboxes - would be better if you could SOLO render the target!
    # for each benign render, get the bbox w/ detection of target class.
    bboxes = []
    for i, cam in enumerate(viewpoint_stack):
        render_pkg = render(cam, gaussians, pipe, bg)
        img_path = f"renders/render_{i}.png"
        np_img = (torch.clamp(render_pkg["render"], min=0, max=1.0) * 255) \
                        .byte() \
                        .permute(1, 2, 0) \
                        .contiguous() \
                        .cpu() \
                        .numpy() 
        pil_img = Image.fromarray(np_img)
        pil_img_bw = pil_img.convert('L')
        bw_tresh = 128
        pil_img_bw = pil_img_bw.point(lambda p: p > bw_tresh and 255)
        # pil_img_bw = PIL.ImageOps.invert(pil_img_bw)
        bbox = pil_img_bw.getbbox()
        
        pil_img.save(img_path)

        rendered_img_input = dt2_input(img_path)
        # bbox = get_instances_bboxes(model, rendered_img_input, target = target.detach().cpu().numpy(), threshold=0.2)
        bboxes.append(bbox)

        draw = PIL.ImageDraw.Draw(pil_img_bw)
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), "object", fill="red")
        pil_img_bw.save(f'renders/bw/bbox_render_{i}.jpg')    
        bbox = np.expand_dims(np.array(bbox), axis=0)

    gt_bboxes = np.array(bboxes)
    batch_mode = False  # Set this to False for single camera mode

    for it in range(1000):
        renders = []
        
        if batch_mode:
            for cam in viewpoint_stack:
                render_pkg = render(cam, gaussians, pipe, bg)
                renders.append(render_pkg["render"])
            renders = torch.stack(renders)
           
            loss = model_input(model, renders, target=target, bboxes=gt_bboxes, batch_size=renders.shape[0])
        else:
            cam = viewpoint_stack[0]
            render_pkg = render(cam, gaussians, pipe, bg)
            renders.append(render_pkg["render"])
            renders = torch.stack(renders)

            loss = model_input(model, renders, target=target, bboxes=gt_bboxes[0], batch_size=renders.shape[0])
 
        print(f"Loss: {loss}")
        loss.backward(retain_graph=True)

        if gaussians._features_rest.grad is not None and gaussians._features_dc.grad is not None:
            epsilon = 5.0
            alpha = 0.001
            # gaussian_color_linf_attack(gaussians, alpha, epsilon, original_features_rest, original_features_dc)
            gaussian_position_linf_attack(gaussians, alpha, epsilon, original_features_xyz)
            # gaussian_scaling_linf_attack(gaussians, alpha, epsilon, original_features_scaling)

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
                        instance_mask_thresh=attack_conf_thresh,
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
                    instance_mask_thresh=attack_conf_thresh,
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


if __name__ == "__main__":
    args = parse_args()
    main(args)