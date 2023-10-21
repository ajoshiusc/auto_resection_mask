# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from monai.config.deviceconfig import USE_COMPILED
from monai.networks.layers.spatial_transforms import grid_pull
from monai.networks.utils import meshgrid_ij
from torch.nn.functional import grid_sample

import numpy as np

def jacobian_determinant(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.
    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.
    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis): return np.diff(
        array, axis=axis)[:, :(H-1), :(W-1), :(D-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2] -
                                                   dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det




def jacobian_determinant_torch(vf):
    """
    Given a displacement vector field vf, compute the jacobian determinant scalar field.
    vf is assumed to be a vector field of shape (3,H,W,D),
    and it is interpreted as the displacement field.
    So it is defining a discretely sampled map from a subset of 3-space into 3-space,
    namely the map that sends point (x,y,z) to the point (x,y,z)+vf[:,x,y,z].
    This function computes a jacobian determinant by taking discrete differences in each spatial direction.
    Returns a numpy array of shape (H-1,W-1,D-1).
    """

    _, H, W, D = vf.shape

    # Compute discrete spatial derivatives
    def diff_and_trim(array, axis): return torch.diff(
        array, axis=axis)[:, :(H-1), :(W-1), :(D-1)]
    dx = diff_and_trim(vf, 1)
    dy = diff_and_trim(vf, 2)
    dz = diff_and_trim(vf, 3)

    # Add derivative of identity map
    dx[0] += 1
    dy[1] += 1
    dz[2] += 1

    # Compute determinant at each spatial location
    det = dx[0]*(dy[1]*dz[2]-dz[1]*dy[2]) - dy[0]*(dx[1]*dz[2] -
                                                   dz[1]*dx[2]) + dz[0]*(dx[1]*dy[2]-dy[1]*dx[2])

    return det




def get_grid(moving_shape, target_shape, requires_grad=False):

    mesh_points = [torch.arange(0, dim) for dim in target_shape]
    ref_grid = torch.stack(meshgrid_ij(*mesh_points),
                           dim=0)  # (spatial_dims, ...)
    # grid = torch.stack([grid] * ddf.shape[0], dim=0)  # (batch, spatial_dims, ...)
    #ref_grid = grid.to(ddf)
    ref_grid = ref_grid.float()
    ref_grid[0] *= (moving_shape[0]/target_shape[0])
    ref_grid[1] *= (moving_shape[1]/target_shape[1])
    ref_grid[2] *= (moving_shape[2]/target_shape[2])
    ref_grid.requires_grad = False
    return ref_grid


def apply_warp(disp_field, moving_image, target_image, interp_mode='bilinear'):
    # disp_field size should be Bx3xWxHxD
    # moving_image and target_image size should be BxCxWxHxD
    # B=batch size, C=num channels, W=width, H=height, and D=depth

    ref_grid = get_grid(moving_image.shape[2:], target_image.shape[2:])

    grid = ref_grid[None,].to(disp_field) + disp_field
    grid = torch.permute(grid, (0, 2, 3, 4, 1))

    for i, dim in enumerate(moving_image.shape[2:]):
        grid[..., i] = grid[..., i] * 2 / (dim - 1) - 1

    spatial_dims = 3
    index_ordering: List[int] = list(range(spatial_dims - 1, -1, -1))
    grid = grid[..., index_ordering]  # z, y, x -> x, y, z

    warped_image = grid_sample(moving_image.to(
        disp_field), grid=grid, align_corners=True, mode=interp_mode)

    return warped_image
