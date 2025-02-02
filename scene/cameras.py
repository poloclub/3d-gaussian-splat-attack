#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


    def update_world_view_projection_transforms(self):
        # Recompute world_view_transform with updated T
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)
        ).transpose(0, 1).cuda()

        # Update the full projection transform with the new world_view_transform
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)        


    def transform(self, T):
        """
        Example: 
                T[0] += 1.0 (shift camera +1 unit in x direction - right) 
                T[1] += 1.0 (shift camera +1 unit in y direction - up)
                T[2] += 1.0 (shift camera +1 unit in z direction - forward)
        """
        assert isinstance(T, np.ndarray), "T must be a numpy array"
        assert T.shape == (3,), "T must be of shape (3,)"
        self.T = T

        self.update_world_view_projection_transforms()
        
    def yaw(self, angle):
        
        def get_yaw_rotation_matrix(theta):
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            return torch.tensor([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])

        # Define yaw angle in radians (e.g., 10 degrees left)
        yaw_angle = np.radians(angle)

        # Get the yaw rotation matrix
        yaw_rotation = get_yaw_rotation_matrix(yaw_angle)

        # Apply the yaw rotation to the camera's rotation matrix R
        self.R = yaw_rotation.mm(torch.tensor(self.R)).numpy().astype(np.float64)

        self.update_world_view_projection_transforms()


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

